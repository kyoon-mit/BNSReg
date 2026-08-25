import torch.nn as nn


def s4_param_groups(
    module: nn.Module,
    base_lr: float,
    weight_decay: float,
) -> list[dict]:
    """Build AdamW parameter groups honouring the S4 `_optim` convention.

    `S4DKernel.register()` tags the SSM parameters (log_dt, log_A_real, A_imag)
    with a `_optim` dict carrying `weight_decay=0.0` and, optionally, a
    kernel-specific `lr`. Passing `module.parameters()` straight to AdamW
    discards those overrides, which applies weight decay to the SSM poles and
    timescales — exactly what the S4 recipe forbids.

    Args:
        module:       the task/model whose parameters are being optimised.
        base_lr:      learning rate for parameters without an `_optim` tag, and
                      fallback for tagged parameters that specify no `lr`.
        weight_decay: weight decay for untagged parameters only.

    Returns:
        A list of param-group dicts ready for `optim.AdamW(...)`.
    """
    all_params = list(module.parameters())
    default_params = [p for p in all_params if not hasattr(p, '_optim')]
    optim_params = [p for p in all_params if hasattr(p, '_optim')]

    param_groups = [{'params': default_params,
                     'lr': base_lr,
                     'weight_decay': weight_decay}]

    # Collect the unique _optim dicts and create one group for each.
    hps = [getattr(p, '_optim') for p in optim_params]
    unique_hps = [dict(s) for s in sorted(set(frozenset(hp.items()) for hp in hps))]
    for hp in unique_hps:
        group = {
            'params': [p for p in optim_params if getattr(p, '_optim') == hp],
            'lr': hp.get('lr', base_lr),
        }
        group.update(hp)  # applies weight_decay=0.0 and overrides lr if present
        param_groups.append(group)

    return param_groups
