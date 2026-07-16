from __future__ import annotations

from cwipss.analysis.veto import VetoConfig, VetoContext, evaluate_vetoes


def _row(**overrides):
    row = {
        "t0_rec": 100,
        "t1_rec": 150,
        "dur_rec": 50,
        "freq_mhz": 35.0,
    }
    row.update(overrides)
    return row


def _context() -> VetoContext:
    return VetoContext(record_start=0, record_stop=1000, freq_start_mhz=30.0, freq_stop_mhz=40.0)


def test_veto_keeps_unflagged_candidate_for_validation() -> None:
    decision = evaluate_vetoes(_row(), _context())

    assert decision["candidate_status"] == "needs_validation"
    assert decision["veto_flags"] == ""
    assert decision["veto_rule_count"] == 0


def test_channel_local_candidate_has_zero_bandwidth() -> None:
    decision = evaluate_vetoes(_row(), _context())

    assert "broadband" not in decision["veto_flags"].split("|")


def test_edge_veto_flags_time_and_frequency_boundaries() -> None:
    decision = evaluate_vetoes(
        _row(t0_rec=0, t1_rec=20, dur_rec=20, freq_mhz=30.0),
        _context(),
        VetoConfig(edge_time_records=0),
    )

    flags = decision["veto_flags"].split("|")
    assert "time_edge" in flags
    assert "freq_edge" in flags


def test_fixed_channel_veto_flags_narrow_long_candidate() -> None:
    decision = evaluate_vetoes(
        _row(t0_rec=100, t1_rec=500, dur_rec=400, freq_mhz=35.0),
        _context(),
        VetoConfig(max_fixed_channel_bandwidth_fraction=0.01),
    )

    assert "fixed_channel" in decision["veto_flags"].split("|")


def test_channel_local_candidate_is_not_a_broadband_burst() -> None:
    decision = evaluate_vetoes(
        _row(t0_rec=100, t1_rec=110, dur_rec=10, freq_mhz=35.0),
        _context(),
    )

    assert "burst_train" not in decision["veto_flags"].split("|")


def test_disabled_veto_adds_review_fields_without_flags() -> None:
    decision = evaluate_vetoes(_row(), _context(), VetoConfig(enabled=False))

    assert decision["candidate_status"] == "needs_validation"
    assert decision["veto_flags"] == ""
    assert decision["veto_details_json"] == "{}"
