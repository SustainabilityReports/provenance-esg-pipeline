import pytest

from campaign.budget_guard import (
    check_allocation,
    check_monthly_quota,
    estimate_core_hours,
    format_walltime,
    walltime_seconds,
    worst_case_core_hours,
)


def test_estimate_scales_with_pages():
    assert estimate_core_hours(10_000, 0.005) == pytest.approx(50.0)
    with pytest.raises(ValueError):
        estimate_core_hours(-1)


def test_allocation_passes_above_floor():
    d = check_allocation(remaining_core_h=1000, unsettled_core_h=200, estimate_core_h=400, floor_core_h=300)
    assert d.passed
    assert d.headroom_core_h == 400


def test_allocation_refuses_below_floor():
    d = check_allocation(remaining_core_h=1000, unsettled_core_h=400, estimate_core_h=400, floor_core_h=300)
    assert not d.passed
    assert "below the 300 core-h floor" in d.reason


def test_unsettled_hours_count_against_the_budget():
    # The same campaign fits only while the lagging accounting is taken into account.
    assert check_allocation(1000, 0, 600, 300).passed
    assert not check_allocation(1000, 200, 600, 300).passed


def test_monthly_quota():
    # 160,000 core-h over 3 months -> ~53,333 per month
    assert check_monthly_quota(5000, 40_000, 5000, 160_000, 3).passed
    d = check_monthly_quota(5000, 45_000, 5000, 160_000, 3)
    assert not d.passed
    assert "monthly quota" in d.reason


def test_walltime_is_clamped():
    assert walltime_seconds(0) == 3600
    assert walltime_seconds(9_742) == 1800 + int(0.75 * 9_742)
    assert walltime_seconds(10_000_000) == 24 * 3600 - 60


def test_worst_case_and_format():
    assert worst_case_core_hours(3600) == 32
    assert format_walltime(9_106) == "02:31:46"
