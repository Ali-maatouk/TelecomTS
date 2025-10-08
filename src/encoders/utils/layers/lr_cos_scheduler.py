import math
from torch.optim.lr_scheduler import LambdaLR

def get_cosine_with_warmup_scheduler(
    optimizer,
    *,
    total_steps: int,
    warmup_steps: int
) -> LambdaLR:
    """
    Returns a LambdaLR scheduler that
    - linearly warms up from 0 → base LR over `warmup_steps`
    - then cosine-decays from base → 0 over the remaining steps.
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            # linear warmup: 0→1
            return float(current_step) / float(max(1, warmup_steps))
        # cosine decay: 1→0
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)
