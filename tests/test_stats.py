from __future__ import annotations

from cwipss.stats import benjamini_hochberg, review_validation_rows


def test_benjamini_hochberg_preserves_order_and_monotonic_adjustment() -> None:
    qvalues = benjamini_hochberg([0.01, 0.04, 0.03, 0.20])

    assert qvalues == [0.04, 0.05333333333333334, 0.05333333333333334, 0.2]


def test_review_validation_rows_adds_run_and_global_q_values() -> None:
    rows = [
        {
            "run_id": "run-a",
            "candidate_id": "1",
            "validation_status": "evaluated",
            "shuffle_pvalue": "0.01",
            "observed_metric": "4.0",
            "fold_profile_snr": "4.0",
        },
        {
            "run_id": "run-a",
            "candidate_id": "2",
            "validation_status": "evaluated",
            "shuffle_pvalue": "0.04",
            "observed_metric": "5.0",
            "fold_profile_snr": "5.0",
        },
        {
            "run_id": "run-b",
            "candidate_id": "3",
            "validation_status": "evaluated",
            "shuffle_pvalue": "0.03",
            "observed_metric": "6.0",
            "fold_profile_snr": "6.0",
        },
        {
            "run_id": "run-b",
            "candidate_id": "4",
            "validation_status": "error",
            "shuffle_pvalue": "",
            "observed_metric": "",
            "fold_profile_snr": "",
        },
    ]

    reviewed = review_validation_rows(rows)

    assert reviewed[0]["p_value"] == 0.01
    assert reviewed[0]["q_value"] == 0.02
    assert reviewed[0]["global_q_value"] == 0.03
    assert reviewed[0]["evidence_rank"] == 1
    assert reviewed[1]["q_value"] == 0.04
    assert reviewed[1]["global_q_value"] == 0.04
    assert reviewed[2]["q_value"] == 0.03
    assert reviewed[2]["global_q_value"] == 0.04
    assert reviewed[3]["stats_status"] == "missing_pvalue"
    assert reviewed[3]["evidence_rank"] == ""


def test_evidence_rank_tie_breaks_on_observed_metric() -> None:
    rows = [
        {
            "run_id": "run-a",
            "candidate_id": "1",
            "validation_status": "evaluated",
            "shuffle_pvalue": "0.05",
            "observed_metric": "3.0",
            "fold_profile_snr": "3.0",
        },
        {
            "run_id": "run-a",
            "candidate_id": "2",
            "validation_status": "evaluated",
            "shuffle_pvalue": "0.05",
            "observed_metric": "6.0",
            "fold_profile_snr": "6.0",
        },
    ]

    reviewed = review_validation_rows(rows)

    assert reviewed[1]["evidence_rank"] == 1
    assert reviewed[0]["evidence_rank"] == 2
