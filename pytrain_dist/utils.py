import functools

import idr_torch
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy


def print_rank_0(*args, **kwargs):
    """Print only on rank 0"""
    if idr_torch.rank == 0:
        print(*args, **kwargs)


def auto_wrap_policy_for_peft(transformer_layer_name):
    from peft.tuners import PrefixEncoder, PromptEmbedding, PromptEncoder
    from torch.distributed.fsdp.wrap import _or_policy, lambda_auto_wrap_policy

    def lambda_policy_fn(module):
        if (
            len(list(module.named_children())) == 0
            and getattr(module, "weight", None) is not None
            and module.weight.requires_grad
        ):
            return True
        return False

    lambda_policy = functools.partial(
        lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn
    )
    transformer_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=(
            PrefixEncoder,
            PromptEncoder,
            PromptEmbedding,
            transformer_layer_name,
            # FullyShardedDataParallelPlugin.get_module_class_from_name(
            #     model, transformer_layer_name
            # ),
        ),
    )

    auto_wrap_policy = functools.partial(
        _or_policy, policies=[lambda_policy, transformer_wrap_policy]
    )
    return auto_wrap_policy


def get_auto_wrap_policy(config, transformer_layer_cls):
    if config.peft is not None:
        return auto_wrap_policy_for_peft(transformer_layer_cls)
    else:
        return functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={transformer_layer_cls},
        )