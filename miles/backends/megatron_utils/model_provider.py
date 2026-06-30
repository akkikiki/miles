# Adapt from https://github.com/NVIDIA/Megatron-LM/blob/b1efb3c7126ef7615e8c333432d76e08038e17ff/pretrain_gpt.py
import argparse
import inspect
import logging
from contextlib import nullcontext
from typing import Literal

import torch
from megatron.core import tensor_parallel
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.transformer.spec_utils import import_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.arguments import core_transformer_config_from_args

from miles.utils.misc import load_function
from miles.utils.replay_base import routing_replay_manager

logger = logging.getLogger(__name__)


# Adapt from https://github.com/volcengine/verl/blob/c3b20575d2bc815fcccd84bddb4c0401fc4b632b/verl/models/llama/megatron/layers/parallel_linear.py#L82
class LinearForLastLayer(torch.nn.Linear):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: TransformerConfig,
        bias: bool = True,
    ) -> None:
        super().__init__(in_features=input_size, out_features=output_size, bias=bias)
        self.sequence_parallel = config.sequence_parallel
        if self.sequence_parallel:
            self.weight.sequence_parallel = True

        self.weight.data.normal_(mean=0.0, std=0.02)
        if bias:
            self.bias.data.zero_()

    def forward(
        self,
        input_: torch.Tensor,
        weight: torch.Tensor | None = None,
        runtime_gather_output: bool | None = None,
    ) -> tuple[torch.Tensor, None]:
        logits = super().forward(input_)
        logits = logits.float()
        if self.sequence_parallel:
            logits = tensor_parallel.gather_from_sequence_parallel_region(logits, tensor_parallel_output_grad=False)
        return logits, None


def _qlora_base_recipe(config, base_prec: str):
    """TE recipe for building the frozen LoRA base in fp8/fp4 (QLoRA).

    Reuses the compute-side recipe builders, so 'fp4' requires the nvfp4 compute
    config (i.e. FP4=1 / --fp4-recipe nvfp4) and 'fp8' requires an fp8 format
    (--fp8-format). Raises a clear error otherwise.
    """
    if base_prec == "fp4":
        from megatron.core.fp4_utils import get_fp4_recipe

        if getattr(config, "fp4", None) is None:
            raise ValueError(
                "--lora-base-precision=fp4 needs the nvfp4 compute config set "
                "(run with FP4=1 / --fp4-format e2m1 --fp4-recipe nvfp4)."
            )
        return get_fp4_recipe(config)
    # fp8
    from megatron.core.fp8_utils import get_fp8_recipe

    if getattr(config, "fp8", None) is None:
        raise ValueError(
            "--lora-base-precision=fp8 needs an fp8 compute format set "
            "(e.g. --fp8-format e4m3)."
        )
    return get_fp8_recipe(config)


def get_model_provider_func(
    args: argparse.Namespace,
    role: Literal["actor", "critic"] = "actor",
):
    # Support custom model provider path (similar to --custom-rm-path for reward models)
    if getattr(args, "custom_model_provider_path", None):

        def wrapped_model_provider(
            pre_process: bool = True,
            post_process: bool = True,
            vp_stage: int | None = None,
            config: TransformerConfig | None = None,
            pg_collection=None,
        ) -> GPTModel:
            assert config is None, "miles builds the config from args, so it expects config to be None"
            custom_model_provider = load_function(args.custom_model_provider_path)
            # Check if the custom provider supports vp_stage parameter
            has_vp_stage = "vp_stage" in inspect.signature(custom_model_provider).parameters
            if has_vp_stage:
                model = custom_model_provider(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
            else:
                model = custom_model_provider(pre_process=pre_process, post_process=post_process)
            # Apply critic output layer if needed
            if post_process and role == "critic":
                model.output_layer = LinearForLastLayer(
                    input_size=model.config.hidden_size, output_size=1, config=model.config
                )
            return model

        return wrapped_model_provider

    if args.megatron_to_hf_mode == "bridge":
        from megatron.bridge import AutoBridge

        bridge = AutoBridge.from_hf_pretrained(args.hf_checkpoint, trust_remote_code=True)
        provider = bridge.to_megatron_provider(load_weights=False)
        # TODO: we should not manually set this...
        provider.tensor_model_parallel_size = args.tensor_model_parallel_size
        provider.pipeline_model_parallel_size = args.pipeline_model_parallel_size
        provider.expert_model_parallel_size = args.expert_model_parallel_size
        provider.expert_tensor_parallel_size = args.expert_tensor_parallel_size
        provider.sequence_parallel = args.sequence_parallel
        provider.context_parallel_size = args.context_parallel_size
        # CP>1 VL models assert this; bridge configs skip core_transformer_config_from_args.
        provider.calculate_per_token_loss = args.calculate_per_token_loss
        provider.attention_softmax_in_fp32 = args.attention_softmax_in_fp32
        provider.variable_seq_lengths = args.variable_seq_lengths
        if hasattr(args, "moe_token_dispatcher_type"):
            provider.moe_token_dispatcher_type = args.moe_token_dispatcher_type
        if getattr(args, "decoder_first_pipeline_num_layers", None) is not None:
            provider.num_layers_in_first_pipeline_stage = args.decoder_first_pipeline_num_layers
        if getattr(args, "decoder_last_pipeline_num_layers", None) is not None:
            provider.num_layers_in_last_pipeline_stage = args.decoder_last_pipeline_num_layers
        if getattr(args, "moe_router_bias_update_rate", None) is not None:
            provider.moe_router_bias_update_rate = args.moe_router_bias_update_rate
        if getattr(args, "moe_aux_loss_coeff", None) is not None:
            provider.moe_aux_loss_coeff = args.moe_aux_loss_coeff
        provider.finalize()

        def wrapped_bridge_provider(
            pre_process: bool = True,
            post_process: bool = True,
            vp_stage: int | None = None,
            config: TransformerConfig | None = None,
            pg_collection=None,
        ) -> GPTModel:
            assert config is None, "miles builds the config from args, so it expects config to be None"
            # PP>1 paths in megatron.bridge providers (e.g. mamba_provider) read
            # self._pg_collection.pp during provide(); without forwarding the
            # caller's pg_collection here, those code paths hit AttributeError.
            if pg_collection is not None:
                provider._pg_collection = pg_collection
            model = provider.provide(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
            # Gemma-4 forward returns (logits, loss_mask); keep logits only.
            _bridge_forward = model.forward

            def _logits_only_forward(*args, **kwargs):
                out = _bridge_forward(*args, **kwargs)
                return out[0] if isinstance(out, tuple) else out

            model.forward = _logits_only_forward
            return model

        return wrapped_bridge_provider

    def model_provider(
        pre_process: bool = True,
        post_process: bool = True,
        vp_stage: int | None = None,
        config: TransformerConfig | None = None,
        pg_collection=None,
    ) -> GPTModel:
        """Builds the model.

        If you set the use_legacy_models to True, it will return the legacy GPT model and if not the mcore GPT model.

        Args:
            pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
            post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


        Returns:
            Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
        """
        use_te = args.transformer_impl == "transformer_engine"

        # Experimental loading arguments from yaml
        assert config is None, "miles builds the config from args, so it expects config to be None"
        config = core_transformer_config_from_args(args)

        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
            # Allow the spec to be a function so that user can use customized Megatron easier.
            if callable(transformer_layer_spec):
                transformer_layer_spec = transformer_layer_spec(args, config, vp_stage)
        else:
            if args.num_experts:
                # Define the decoder block spec
                kwargs = {
                    "use_transformer_engine": use_te,
                }
                if vp_stage is not None:
                    kwargs["vp_stage"] = vp_stage
                transformer_layer_spec = get_gpt_decoder_block_spec(config, **kwargs)
            else:
                # Define the decoder layer spec
                if use_te:
                    transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                        num_experts=args.num_experts,
                        moe_grouped_gemm=args.moe_grouped_gemm,
                        qk_layernorm=args.qk_layernorm,
                        multi_latent_attention=args.multi_latent_attention,
                        moe_use_legacy_grouped_gemm=args.moe_use_legacy_grouped_gemm,
                    )
                else:
                    transformer_layer_spec = get_gpt_layer_local_spec(
                        num_experts=args.num_experts,
                        moe_grouped_gemm=args.moe_grouped_gemm,
                        qk_layernorm=args.qk_layernorm,
                        multi_latent_attention=args.multi_latent_attention,
                        moe_use_legacy_grouped_gemm=args.moe_use_legacy_grouped_gemm,
                        normalization=args.normalization,
                        use_kitchen=config.use_kitchen,
                        use_true_on_policy_backend=config.true_on_policy_contract is not None,
                        use_kitchen_attention=config.use_kitchen_attention,
                        kitchen_attention_backend=config.kitchen_attention_backend,
                    )

        build_model_context = nullcontext
        build_model_context_args = {}

        # QLoRA: store the FROZEN LoRA base in fp8/fp4 with NO bf16 master.
        # --lora-base-precision {fp8,fp4} builds the base inside TE fp8_model_init
        # with preserve_high_precision_init_val=False (the base is frozen, so no
        # master is needed). This is what lets the 744B trainer fit on fewer GPUs.
        # Distinct from --fp8-param-gather (full-FT fp8 training, which KEEPS the
        # master). Default 'bf16' == legacy behavior.
        from miles.backends.megatron_utils.lora_utils import is_lora_enabled

        base_prec = getattr(args, "lora_base_precision", "bf16")
        qlora_lowp = base_prec in ("fp8", "fp4") and is_lora_enabled(args)
        if base_prec in ("fp8", "fp4") and not is_lora_enabled(args):
            raise ValueError(
                f"--lora-base-precision={base_prec} requires LoRA to be enabled "
                "(the quantized frozen base only makes sense with a bf16 adapter)."
            )

        if qlora_lowp or args.fp8_param_gather:
            try:
                from transformer_engine.pytorch import fp8_model_init
            except Exception as e:
                raise RuntimeError(
                    "fp8/fp4 base init requires `fp8_model_init` from TransformerEngine, but not found."
                ) from e

            build_model_context = fp8_model_init
            build_model_context_args["enabled"] = True
            sig = inspect.signature(fp8_model_init).parameters

            if qlora_lowp:
                # frozen base -> drop the high-precision master (the memory win)
                if "preserve_high_precision_init_val" in sig:
                    build_model_context_args["preserve_high_precision_init_val"] = False
                if "recipe" in sig:
                    build_model_context_args["recipe"] = _qlora_base_recipe(config, base_prec)
            else:
                # legacy --fp8-param-gather (fp8 training): keep the master
                if "preserve_high_precision_init_val" in sig:
                    build_model_context_args["preserve_high_precision_init_val"] = True
            logger.info(
                f"Building frozen base under fp8_model_init: base_precision={base_prec}, "
                f"qlora_lowp={qlora_lowp}, args={ {k: v for k, v in build_model_context_args.items() if k != 'recipe'} }"
            )

        kwargs = {
            "config": config,
            "transformer_layer_spec": transformer_layer_spec,
            "vocab_size": args.padded_vocab_size,
            "max_sequence_length": args.max_position_embeddings,
            "pre_process": pre_process,
            "post_process": post_process,
            "fp16_lm_cross_entropy": args.fp16_lm_cross_entropy,
            "parallel_output": True,
            "share_embeddings_and_output_weights": not args.untie_embeddings_and_output_weights,
            "position_embedding_type": args.position_embedding_type,
            "rotary_percent": args.rotary_percent,
            "rotary_base": args.rotary_base,
            "rope_scaling": args.use_rope_scaling,
        }

        if vp_stage is not None:
            kwargs["vp_stage"] = vp_stage

        if args.mtp_num_layers:
            from megatron.core.models.gpt.gpt_layer_specs import get_gpt_mtp_block_spec

            mtp_kwargs = {
                "use_transformer_engine": use_te,
            }
            if vp_stage is not None:
                mtp_kwargs["vp_stage"] = vp_stage

            # hard code here to skip r3 registration for mtp layers
            # getattr is required to avoid ckpt conversion errors
            if getattr(args, "use_rollout_routing_replay", False):
                routing_replay_manager.enabled = False
                logger.warning(
                    "Rollout routing replay is not applicable for MTP modules, so skipped replay registration"
                )
            mtp_block_spec = get_gpt_mtp_block_spec(config, transformer_layer_spec, **mtp_kwargs)
            kwargs["mtp_block_spec"] = mtp_block_spec
            if getattr(args, "use_rollout_routing_replay", False):
                routing_replay_manager.enabled = True

        with build_model_context(**build_model_context_args):
            model = GPTModel(**kwargs)

        if post_process and role == "critic":
            model.output_layer = LinearForLastLayer(input_size=config.hidden_size, output_size=1, config=config)

        return model

    return model_provider
