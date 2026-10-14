from io import BytesIO

from llm_bench.calibration import calibrate_dataset
from llm_bench.dataset_io import parse_dataset_file
from llm_bench.schemas import DatasetRow
from llm_bench.scorers import extract_score, token_f1


def test_calibrate_ready_with_scores():
    rows = [
        DatasetRow(id="1", question="Q1", gold_score=1.0),
        DatasetRow(id="2", question="Q2", gold_score=4.0, reference_answer="A"),
    ]
    valid, report = calibrate_dataset(rows)
    assert len(valid) == 2
    assert report.ready_to_run
    assert report.score_present == 2


def test_calibrate_rejects_empty():
    _, report = calibrate_dataset([])
    assert not report.ready_to_run
    assert report.error_count >= 1


def test_parse_sample_csv():
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "data/datasets/sample_qa_scores.csv"
    bundle = parse_dataset_file(path, path.name)
    assert bundle.calibration.ready_to_run
    assert bundle.calibration.n_valid == 10
    assert bundle.calibration.score_bins


def test_extract_score_json():
    assert extract_score('{"answer":"x","score":3.5}') == 3.5
    assert extract_score("Score: 4") == 4.0


def test_token_f1():
    assert token_f1("Paris", "Paris") == 1.0
    assert token_f1("a b", "b c") == 0.5
