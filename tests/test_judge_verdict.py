from llm_bench.scorers import extract_judge_verdict


def test_extract_judge_json():
    text = '{"analysis": "Missing density values.", "score": 3.5}'
    analysis, score = extract_judge_verdict(text)
    assert analysis == "Missing density values."
    assert score == 3.5


def test_extract_judge_score_only():
    analysis, score = extract_judge_verdict("4")
    assert score == 4.0
    assert analysis is None
