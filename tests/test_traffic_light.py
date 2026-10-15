from llm_bench.metrics import summarize_predictions
from llm_bench.schemas import RowPrediction
from llm_bench.settings import Settings
from llm_bench.traffic_light import (
    TrafficLight,
    classify_traffic_light,
    composite_score_100,
    row_score_100,
    score_from_abs_error,
)


def test_classify_thresholds():
    cfg = Settings(traffic_green_min=80, traffic_yellow_min=60)
    assert classify_traffic_light(90, cfg) == TrafficLight.green
    assert classify_traffic_light(70, cfg) == TrafficLight.yellow
    assert classify_traffic_light(40, cfg) == TrafficLight.red
    assert classify_traffic_light(None, cfg) == TrafficLight.unknown


def test_score_from_mae_scale():
    # MAE=0 on 0..5 scale → 100; MAE=2.5 → 50; MAE=5 → 0
    assert score_from_abs_error(0.0, score_min=0, score_max=5) == 100.0
    assert score_from_abs_error(2.5, score_min=0, score_max=5) == 50.0
    assert score_from_abs_error(5.0, score_min=0, score_max=5) == 0.0


def test_composite_and_summary():
    cfg = Settings(score_min=0, score_max=5, traffic_green_min=80, traffic_yellow_min=60)
    preds = [
        RowPrediction(
            row_id="1",
            model_id="m1",
            model_name="Good",
            metrics={"abs_error": 0.2},
            weight=1.0,
        ),
        RowPrediction(
            row_id="2",
            model_id="m1",
            model_name="Good",
            metrics={"abs_error": 0.3},
            weight=1.0,
        ),
        RowPrediction(
            row_id="3",
            model_id="m2",
            model_name="Bad",
            metrics={"abs_error": 4.0},
            weight=1.0,
        ),
    ]
    # enrich rows like runner
    for p in preds:
        p.score_100 = row_score_100(p.metrics, cfg=cfg)
        p.traffic_light = classify_traffic_light(p.score_100, cfg).value

    summaries = summarize_predictions(preds, cfg=cfg)
    assert summaries[0].model_name == "Good"
    assert summaries[0].score_100 is not None and summaries[0].score_100 >= 80
    assert summaries[0].traffic_light == "green"
    assert summaries[1].traffic_light == "red"


def test_composite_error_penalty():
    cfg = Settings(score_min=0, score_max=5)
    perfect = composite_score_100(mae=0.0, error_rate=0.0, cfg=cfg)
    penalized = composite_score_100(mae=0.0, error_rate=0.5, cfg=cfg)
    assert perfect == 100.0
    assert penalized == 50.0


def test_token_efficiency_in_summary():
    preds = [
        RowPrediction(
            row_id="1",
            model_id="m1",
            model_name="Verbose",
            metrics={"token_f1": 1.0},
            prompt_tokens=100,
            completion_tokens=400,
            weight=1.0,
        ),
        RowPrediction(
            row_id="2",
            model_id="m2",
            model_name="Compact",
            metrics={"token_f1": 1.0},
            prompt_tokens=100,
            completion_tokens=50,
            weight=1.0,
        ),
    ]
    summaries = {s.model_name: s for s in summarize_predictions(preds)}
    assert summaries["Verbose"].total_tokens == 500
    assert summaries["Compact"].total_tokens == 150
    assert summaries["Compact"].score_per_1k_tokens > summaries["Verbose"].score_per_1k_tokens
