import re

import idr_torch
import torch
from peft import LoraConfig, get_peft_model
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import AutoModelForCausalLM, PreTrainedModel
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaModel
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2Model

from .config import GlobalConfig
from .utils import get_auto_wrap_policy

DEVICE = torch.device("cuda", idr_torch.local_rank)


def get_hf_model(
    config: GlobalConfig
) -> tuple[PreTrainedModel, torch.nn.Module]:
    if config.smart_loading and idr_torch.local_rank != 0:
        with torch.device("meta"):
            model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
                config.model_path,
                trust_remote_code=False,
                torch_dtype=torch.bfloat16,
            )
    else:
        model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            trust_remote_code=False,
            torch_dtype=torch.bfloat16,
        )

    if type(model.base_model) is LlamaModel:
        transformer_layer_cls = LlamaDecoderLayer
        target_modules = ["q_proj", "k_proj", "v_proj"]
    elif type(model.base_model) is Qwen2Model:
        transformer_layer_cls = Qwen2DecoderLayer
        target_modules = ["q_proj", "k_proj", "v_proj"]
    else:
        raise ValueError(f"Model type not implemented: {type(model.base_model)}")

    if config.peft == "lora":
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, lora_config)
        model.config.use_cache = False

    elif config.nb_layer_freezed > 0:
        for name, params in model.named_parameters():
            if re.search(r"embed", name) is not None:
                params.requires_grad = False
            elif re.search(r"\.(\d+)\.", name) is not None:
                if (
                    int(re.search(r"\.(\d+)\.", name).group(1))
                    < config.nb_layer_freezed
                ):
                    params.requires_grad = False

    return model, transformer_layer_cls


def get_fsdp_model(config: GlobalConfig) -> FSDP:
    torch.distributed.init_process_group(
        "nccl", rank=idr_torch.rank, world_size=idr_torch.world_size
    )
    torch.cuda.set_device(DEVICE)

    model, transformer_layer_cls = get_hf_model(config)

    auto_wrap_policy = get_auto_wrap_policy(config, transformer_layer_cls)
    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        device_id=idr_torch.local_rank,
        sync_module_states=True if config.smart_loading else False,
        param_init_fn=lambda module: module.to_empty(
            device=torch.device("cuda"), recurse=False
        ) if idr_torch.local_rank != 0 and config.smart_loading else None,
    )

    # optimizer, lr_scheduler = get_optimizer_scheduler(config, model)

    return model  # , optimizer, lr_scheduler


def get_model(config: GlobalConfig) -> PreTrainedModel:
    if config.training_dist is None:
        model, _ = get_hf_model(config)
        model.to(DEVICE)
        return model
    elif config.training_dist == "fsdp":
        return get_fsdp_model(config)
    else:
        raise ValueError(f"Unknown distribution type: {config.training_dist}")
