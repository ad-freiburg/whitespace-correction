import math
from typing import Any, Union

from torch import optim

from whitespace_correction.utils.config import LRSchedulerConfig

LR_SCHEDULER_TYPE = Union[optim.lr_scheduler.LambdaLR,
                          optim.lr_scheduler.StepLR,
                          optim.lr_scheduler.MultiStepLR]


def get_lr_scheduler_from_config(config: LRSchedulerConfig,
                                 num_training_steps: int,
                                 optimizer: optim.Optimizer) -> LR_SCHEDULER_TYPE:
    return get_lr_scheduler(optimizer=optimizer,
                            name=config.type,
                            num_training_steps=num_training_steps,
                            **config.arguments)


def get_lr_scheduler(optimizer: optim.Optimizer,
                     name: str,
                     num_training_steps: int,
                     **kwargs: Any) -> LR_SCHEDULER_TYPE:
    if name == "linear_with_warmup":
        warmup_steps: Union[int, float] = kwargs["warmup_steps"]
        if isinstance(warmup_steps, float):
            warmup_steps = num_training_steps * warmup_steps

        def _linear(step: int) -> float:
            if step < warmup_steps:
                return step / max(1.0, warmup_steps)
            frac = (num_training_steps - step) / max(1.0, num_training_steps - warmup_steps)
            return max(0.0, frac)

        return optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                           lr_lambda=_linear)

    elif name == "cosine_with_warmup":
        warmup_steps = kwargs["warmup_steps"]
        if isinstance(warmup_steps, float):
            warmup_steps = num_training_steps * warmup_steps

        def _cosine(step: int) -> float:
            if step < warmup_steps:
                return step / max(1.0, warmup_steps)
            frac = (step - warmup_steps) / max(1.0, num_training_steps - warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * frac)))

        return optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                           lr_lambda=_cosine)

    elif name == "invsqrt_with_warmup":
        warmup_steps = kwargs["warmup_steps"]
        if isinstance(warmup_steps, float):
            warmup_steps = num_training_steps * warmup_steps

        def _inv_sqrt(step: int) -> float:
            if step < warmup_steps:
                return step / max(1.0, warmup_steps)
            return math.sqrt(warmup_steps) / math.sqrt(step)

        return optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                           lr_lambda=_inv_sqrt)

    elif name == "step":
        step_size: Union[int, float] = kwargs["step_size"]
        factor = kwargs["factor"]

        if isinstance(step_size, float):
            step_size = int(num_training_steps * step_size)
        return optim.lr_scheduler.StepLR(optimizer=optimizer,
                                         step_size=step_size,
                                         gamma=factor)

    elif name == "multi_step":
        steps = kwargs["steps"]
        factor = kwargs["factor"]

        if isinstance(steps[0], float):
            steps = [num_training_steps * step for step in steps]
        return optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                              milestones=steps,
                                              gamma=factor)

    elif name == "multi_step_with_warmup":
        warmup_steps = kwargs["warmup_steps"]
        if isinstance(warmup_steps, float):
            warmup_steps = num_training_steps * warmup_steps

        steps = kwargs["steps"]
        factor = kwargs["factor"]

        if isinstance(steps[0], float):
            steps = [num_training_steps * step for step in steps]

        steps = sorted(steps)

        def _multi_step_with_warmup(step: int) -> float:
            if step < warmup_steps:
                return step / max(1.0, warmup_steps)

            power = 0
            for step_at in steps:
                if step >= step_at:
                    power += 1

            return factor ** power

        return optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                           lr_lambda=_multi_step_with_warmup)

    else:
        raise ValueError(f"Unknown learning rate scheduler {name}")
