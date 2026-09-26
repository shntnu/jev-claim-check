# /// script
# requires-python = ">=3.12"
# dependencies = ["altair>=5", "dspy[typesafe]>=3.4.0", "marimo>=0.25.0"]
# ///
"""Run the notebook with synthetic predictions: uv run check_demo.py."""

from collections import Counter
from types import SimpleNamespace
from unittest.mock import Mock, patch

import dspy

from claims import app


def check_demo():
    labels = ["supports", "contradicts", "not_enough_info"]
    examples = [
        dspy.Example(claim=f"Claim {i}", abstract="Original abstract", verdict=label)
        for i, label in enumerate(labels)
    ]
    predictions = [
        dspy.Prediction(verdict=SimpleNamespace(
            value="supports", confidence=confidence,
            probabilities=dict(zip(labels, [0.6, 0.3, 0.1])),
        ))
        for confidence in [0.9, 0.2, 0.5]
    ]
    row = dict(
        backend="Jev", accuracy=1 / 3, seconds=1, dollars=0.01,
        correct=Counter(supports=1), total=Counter(labels), unanswered=0,
        results=list(zip(examples, predictions, [1, 0, 0])),
    )
    overrides = {"dev": examples, "jev": None, "jev_row": row}
    outputs, state = app.run(defs=overrides)
    assert outputs
    assert [r["claim"] for r in state["claim_rows"]] == ["Claim 1", "Claim 2", "Claim 0"]
    assert state["edited_pred"] is state["picked_pred"]
    assert state["triage"].value == 1
    chart = state["confidence_chart"].to_dict()
    assert chart["mark"]["type"] == "bar"
    assert len(chart["data"]["values"]) == len(examples)
    assert {row["result"] for row in chart["data"]["values"]} == {"Correct", "Incorrect"}

    for cut in [0, 1, 3]:
        _, state = app.run(defs={**overrides, "triage": SimpleNamespace(value=cut)})
        assert len(state["sent"]) == cut
        assert state["caught"] == min(cut, 2)
        assert state["remaining_errors"] == 2 - min(cut, 2)

    changed = dspy.Prediction(verdict=SimpleNamespace(
        value="contradicts", confidence=0.8,
        probabilities=dict(zip(labels, [0.1, 0.8, 0.1])),
    ))
    predictor = Mock(return_value=changed)
    with patch.object(dspy, "Predict", return_value=predictor):
        outputs, state = app.run(defs={
            **overrides, "edited_claim": SimpleNamespace(value="Reversed claim"),
            "picked_example": examples[0], "picked_pred": predictions[0],
        })
    predictor.assert_called_once_with(claim="Reversed claim", abstract="Original abstract")
    assert state["edited_pred"] is changed
    assert any("verdict flipped" in output.text for output in outputs if hasattr(output, "text"))
    print("Demo checks passed: sorting, review endpoints, result reuse, and edited-claim prediction.")


if __name__ == "__main__":
    check_demo()
