from optimization.schedulers import CosineWDSchedule, LinearDecaySchedule, WarmupCosineSchedule
import torch


def init_ft_opt(
    encoder,
    predictor,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    num_epochs,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    mixed_precision=False,
    ipe_scale=1.0,
    betas=(0.9, 0.999),
    eps=1e-8,
    frozen_encoder=False,
    zero_init_bias_wd=True,
):
    """Initialize fine-tuning optimization without annealing support.

    ``predictor`` is the downstream task head (classifier or regressor).
    When ``frozen_encoder`` is true, encoder parameters are excluded from the
    optimizer and marked non-trainable; the task head remains trainable.
    """
    if frozen_encoder:
        for parameter in encoder.parameters():
            parameter.requires_grad = False

    trainable_modules = [predictor]
    if not frozen_encoder:
        trainable_modules.insert(0, encoder)

    decay_params = []
    no_decay_params = []
    for module in trainable_modules:
        for name, parameter in module.named_parameters():
            if not parameter.requires_grad:
                continue
            if "bias" in name or len(parameter.shape) == 1:
                no_decay_params.append(parameter)
            else:
                decay_params.append(parameter)

    param_groups = []
    if decay_params:
        param_groups.append({"params": decay_params})
    if no_decay_params:
        param_groups.append(
            {
                "params": no_decay_params,
                "WD_exclude": zero_init_bias_wd,
                "weight_decay": 0,
            }
        )
    if not param_groups:
        raise ValueError("No trainable fine-tuning parameters were supplied.")

    optimizer = torch.optim.AdamW(param_groups, betas=betas, eps=eps)
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(warmup * iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
    )
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
    )
    scaler = torch.cuda.amp.GradScaler() if mixed_precision else None
    return optimizer, scaler, scheduler, wd_scheduler


def init_opt(
    is_anneal,
    encoder,
    predictor,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    num_epochs,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    mixed_precision=False,
    ipe_scale=1.25,
    betas=(0.9, 0.999),
    eps=1e-8,
    zero_init_bias_wd=True,
):
    param_groups = [
        {"params": (p for n, p in encoder.named_parameters() if ("bias" not in n) and (len(p.shape) != 1))},
        {"params": (p for n, p in predictor.named_parameters() if ("bias" not in n) and (len(p.shape) != 1))},
        {
            "params": (p for n, p in encoder.named_parameters() if ("bias" in n) or (len(p.shape) == 1)),
            "WD_exclude": zero_init_bias_wd,
            "weight_decay": 0,
        },
        {
            "params": (p for n, p in predictor.named_parameters() if ("bias" in n) or (len(p.shape) == 1)),
            "WD_exclude": zero_init_bias_wd,
            "weight_decay": 0,
        },
    ]

    optimizer = torch.optim.AdamW(param_groups, betas=betas, eps=eps)
    if not is_anneal:
        scheduler = WarmupCosineSchedule(
            optimizer,
            warmup_steps=int(warmup * iterations_per_epoch),
            start_lr=start_lr,
            ref_lr=ref_lr,
            final_lr=final_lr,
            T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
        )
    else:
        scheduler = LinearDecaySchedule(
            optimizer,
            ref_lr=ref_lr,
            final_lr=final_lr,
            T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
        )
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
    )
    scaler = torch.cuda.amp.GradScaler() if mixed_precision else None
    return optimizer, scaler, scheduler, wd_scheduler
