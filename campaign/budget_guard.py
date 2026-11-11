"""
Allocation budget guard for batch OCR campaigns on an HPC allocation.

Before a campaign is submitted, the scheduler checks that it fits in what is
left of the allocation. The accounting system (on Leonardo: `saldo -b`) lags
the scheduler by up to a day or two, so jobs that already ran but are not yet
billed ("unsettled") are subtracted explicitly:

    remaining - unsettled - estimate >= floor

The floor keeps a reserve for re-runs of failed documents. A second check
spreads use evenly over the allocation window (linearised monthly quota), so a
three-month allocation is not burned in the first week.

Wall time per job is sized from page counts: a fixed warm-up (vLLM start and
model load) plus a per-page allowance, clamped to sane bounds.

Accounting convention on Leonardo Booster: 1 GPU hour = 8 local core hours
(a node is 4 GPUs and 32 cores).
"""

from __future__ import annotations

from dataclasses import dataclass

CORE_HOURS_PER_GPU_HOUR = 8
GPUS_PER_NODE = 4

# Measured on Leonardo Booster with Chandra OCR 2 (vLLM, bf16, 200 DPI),
# averaged over ~7.1M pages including failed and re-run jobs.
DEFAULT_CORE_HOURS_PER_PAGE = 0.0056


@dataclass(frozen=True)
class BudgetDecision:
    passed: bool
    headroom_core_h: float
    reason: str


def estimate_core_hours(pages: int, core_hours_per_page: float = DEFAULT_CORE_HOURS_PER_PAGE) -> float:
    """Core hours a campaign of `pages` pages is expected to cost."""
    if pages < 0:
        raise ValueError("pages must be >= 0")
    return pages * core_hours_per_page


def check_allocation(
    remaining_core_h: float,
    unsettled_core_h: float,
    estimate_core_h: float,
    floor_core_h: float = 300.0,
) -> BudgetDecision:
    """Refuse a launch unless remaining - unsettled - estimate >= floor."""
    headroom = remaining_core_h - unsettled_core_h - estimate_core_h
    if headroom >= floor_core_h:
        return BudgetDecision(True, round(headroom, 1), "fits in the allocation above the floor")
    return BudgetDecision(
        False,
        round(headroom, 1),
        f"remaining {remaining_core_h:.0f} - unsettled {unsettled_core_h:.0f} - estimate "
        f"{estimate_core_h:.0f} = {headroom:.0f} core-h, below the {floor_core_h:.0f} core-h floor",
    )


def check_monthly_quota(
    estimate_core_h: float,
    month_used_core_h: float,
    month_committed_core_h: float,
    total_core_h: float,
    allocation_months: int,
) -> BudgetDecision:
    """Refuse a launch that would push this month's use past the linearised quota."""
    if allocation_months <= 0:
        raise ValueError("allocation_months must be > 0")
    cap = total_core_h / allocation_months
    projected = estimate_core_h + month_used_core_h + month_committed_core_h
    headroom = cap - projected
    if projected <= cap:
        return BudgetDecision(True, round(headroom, 1), "within the monthly linearised quota")
    return BudgetDecision(
        False,
        round(headroom, 1),
        f"estimate {estimate_core_h:.0f} + used {month_used_core_h:.0f} + committed "
        f"{month_committed_core_h:.0f} exceeds the monthly quota of {cap:.0f} core-h",
    )


def walltime_seconds(
    pages: int,
    fixed_seconds: int = 1800,
    per_page_seconds: float = 0.75,
    floor_seconds: int = 3600,
    cap_seconds: int = 24 * 3600 - 60,
) -> int:
    """SLURM wall time for a one-node job over `pages` pages, clamped to [floor, cap]."""
    seconds = fixed_seconds + per_page_seconds * max(pages, 0)
    return int(min(max(seconds, floor_seconds), cap_seconds))


def worst_case_core_hours(walltime_s: int, nodes: int = 1, cores_per_node: int = 32) -> float:
    """Core hours billed if the job runs to its wall-time limit."""
    return walltime_s / 3600 * nodes * cores_per_node


def format_walltime(seconds: int) -> str:
    """Seconds as the HH:MM:SS string sbatch --time expects."""
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
