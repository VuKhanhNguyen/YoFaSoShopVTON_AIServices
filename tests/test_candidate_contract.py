"""Quy tắc ranking là hàm thuần, không cần MySQL/GPU."""
import pytest
from api.services.pipeline import candidate_score, gate_candidate


@pytest.mark.parametrize("category,expected", [("upper_body", .35), ("lower_body", 0), ("dress", 0)])
def test_ranking_limb_weight(category, expected):
    assert candidate_score(category, 0, 0, 0, 1, 0, 0) == pytest.approx(expected)


def test_long_sleeve_alias_does_not_trigger_short_sleeve_limb_gate():
    assert gate_candidate("dress", "long sleeve dress", .9, .9, .1, 3)
    assert not gate_candidate("dress", "sling dress", .9, .9, .1, 3)
