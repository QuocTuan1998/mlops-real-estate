def decide_promotion(
    candidate_value: float,
    production_value: float,
    metric_name: str = "MAE",
    min_improvement: float = 0.0,
) -> bool:
    if metric_name == "R2":
        improvement = candidate_value - production_value
    else:
        improvement = production_value - candidate_value
    return improvement > min_improvement


def test_candidate_better_mae_is_promoted():
    assert decide_promotion(candidate_value=420_000_000, production_value=500_000_000) is True


def test_candidate_worse_mae_is_rejected():
    assert decide_promotion(candidate_value=550_000_000, production_value=500_000_000) is False


def test_candidate_equal_mae_is_rejected_with_zero_threshold():
    assert decide_promotion(candidate_value=500_000_000, production_value=500_000_000) is False


def test_candidate_better_r2_is_promoted():
    assert decide_promotion(candidate_value=0.85, production_value=0.80, metric_name="R2") is True


def test_candidate_worse_r2_is_rejected():
    assert decide_promotion(candidate_value=0.70, production_value=0.80, metric_name="R2") is False


def test_min_improvement_threshold_blocks_marginal_gains():
    # Candidate is only marginally better than production; threshold requires more.
    assert (
        decide_promotion(
            candidate_value=499_000_000,
            production_value=500_000_000,
            min_improvement=5_000_000,
        )
        is False
    )


def test_min_improvement_threshold_allows_significant_gains():
    assert (
        decide_promotion(
            candidate_value=400_000_000,
            production_value=500_000_000,
            min_improvement=5_000_000,
        )
        is True
    )