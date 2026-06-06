from __future__ import annotations

from cwipss.analysis.veto import VetoConfig, VetoContext, evaluate_vetoes


def _row(**overrides):
    row = {
        "record_start": 100,
        "record_stop": 150,
        "duration_records": 50,
        "freq_start_mhz": 35.0,
        "freq_stop_mhz": 36.0,
        "bandwidth_mhz": 1.0,
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


def test_broadband_veto_flags_large_bandwidth_fraction() -> None:
    decision = evaluate_vetoes(_row(freq_start_mhz=31.0, freq_stop_mhz=39.0), _context())

    assert decision["candidate_status"] == "vetoed"
    assert "broadband" in decision["veto_flags"].split("|")


def test_edge_veto_flags_time_and_frequency_boundaries() -> None:
    decision = evaluate_vetoes(
        _row(record_start=0, record_stop=20, duration_records=20, freq_start_mhz=30.0, freq_stop_mhz=31.0),
        _context(),
        VetoConfig(edge_time_records=0),
    )

    flags = decision["veto_flags"].split("|")
    assert "time_edge" in flags
    assert "freq_edge" in flags


def test_fixed_channel_veto_flags_narrow_long_candidate() -> None:
    decision = evaluate_vetoes(
        _row(record_start=100, record_stop=500, duration_records=400, freq_start_mhz=35.0, freq_stop_mhz=35.0),
        _context(),
        VetoConfig(max_fixed_channel_bandwidth_fraction=0.01),
    )

    assert "fixed_channel" in decision["veto_flags"].split("|")


def test_burst_train_veto_flags_short_wide_candidate() -> None:
    decision = evaluate_vetoes(
        _row(record_start=100, record_stop=110, duration_records=10, freq_start_mhz=34.0, freq_stop_mhz=38.0),
        _context(),
    )

    assert "burst_train" in decision["veto_flags"].split("|")


def test_disabled_veto_adds_review_fields_without_flags() -> None:
    decision = evaluate_vetoes(_row(freq_start_mhz=31.0, freq_stop_mhz=39.0), _context(), VetoConfig(enabled=False))

    assert decision["candidate_status"] == "needs_validation"
    assert decision["veto_flags"] == ""
    assert decision["veto_details_json"] == "{}"
