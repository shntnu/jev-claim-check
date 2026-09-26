# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "altair>=5",
#     "dspy[typesafe]>=3.4.0",
#     "marimo>=0.25.0",
# ]
# ///

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import html
    import io
    import json
    import os
    import tarfile
    import time
    import urllib.request
    from collections import Counter
    from typing import Literal, get_args

    import dspy
    import marimo as mo
    from dspy.experimental import Choice, TypeSafe


@app.cell(hide_code=True)
def _():
    Verdict = Literal["supports", "contradicts", "not_enough_info"]
    VERDICTS = get_args(Verdict)


    class CheckClaim(dspy.Signature):
        """Judge a scientific claim against the abstract of a paper cited for it.

        supports: the abstract reports findings that support the claim.
        contradicts: the abstract reports findings that contradict the claim.
        not_enough_info: the abstract neither supports nor contradicts the claim."""

        claim: str = dspy.InputField()
        abstract: str = dspy.InputField(desc="Title and abstract of the cited paper")
        verdict: Verdict = dspy.OutputField(desc="Does the abstract support or contradict the claim?")


    # The same question with the verdict typed as a Choice, so Jev's answer also carries its confidence.
    # Jev sees an identical request, because the answer meanings live in the docstring above.
    CheckClaimWithConfidence = CheckClaim.with_updated_fields(
        "verdict", type_=Choice[tuple((verdict, "") for verdict in VERDICTS)]
    )
    return CheckClaimWithConfidence, VERDICTS


@app.cell(hide_code=True)
def _():
    def scifact_folder():
        """SciFact's data folder next to this notebook, downloaded once from AllenAI (CC BY-NC 2.0)."""
        folder = mo.notebook_dir() / "data" / "scifact"
        if not (folder / "data").exists():
            url = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
            with urllib.request.urlopen(url) as response:
                archive = io.BytesIO(response.read())
            with tarfile.open(fileobj=archive, mode="r:gz") as tar:
                tar.extractall(folder, filter="data")
        return folder / "data"


    def read_jsonl(name):
        return [json.loads(line) for line in (scifact_folder() / f"{name}.jsonl").read_text().splitlines()]


    def load_dev():
        """One example per claim and cited abstract in SciFact's dev split, labeled from its evidence annotations."""
        abstracts = {}
        for doc in read_jsonl("corpus"):
            abstracts[doc["doc_id"]] = doc["title"] + ". " + " ".join(sentence.strip() for sentence in doc["abstract"])

        examples = []
        for claim in read_jsonl("claims_dev"):
            for doc_id in claim["cited_doc_ids"]:
                evidence = claim["evidence"].get(str(doc_id))
                if not evidence:
                    verdict = "not_enough_info"
                elif evidence[0]["label"] == "SUPPORT":
                    verdict = "supports"
                else:
                    verdict = "contradicts"
                example = dspy.Example(claim=claim["claim"], abstract=abstracts[doc_id], verdict=verdict)
                examples.append(example.with_inputs("claim", "abstract"))
        return examples

    return (load_dev,)


@app.cell(hide_code=True)
def _(load_dev):
    dev = load_dev()
    return (dev,)


@app.cell(hide_code=True)
def _():
    def is_correct(example, pred, trace=None):
        verdict = getattr(pred.verdict, "value", pred.verdict)  # a Choice answer keeps the verdict in .value
        return verdict == example.verdict


    def jev_dollars(history):
        return sum(call["usage"]["prompt_tokens"] for call in history) * 0.042 / 1e6  # $0.042 per million input tokens


    def compare(name, program, lm, dollars, examples):
        """Run a CheckClaim program on every example with one backend; return accuracy, time, cost, and answers."""
        evaluate = dspy.Evaluate(devset=examples, metric=is_correct, num_threads=12, max_errors=len(examples))
        start = time.time()
        with dspy.context(lm=lm):
            result = evaluate(program)
        return {
            "backend": name,
            "accuracy": result.score / 100,
            "seconds": time.time() - start,
            "dollars": dollars(lm.history),
            "correct": Counter(example.verdict for example, _, score in result.results if score),
            "total": Counter(example.verdict for example in examples),
            "unanswered": sum("verdict" not in pred for _, pred, _ in result.results),  # refusals and errors
            "results": result.results,
        }

    return compare, jev_dollars


@app.cell(hide_code=True)
def _(CheckClaimWithConfidence, compare, dev, jev_dollars):
    mo.stop(
        not os.environ.get("TYPESAFE_API_KEY"),
        mo.md(
            "Add your TypeSafe API key as `TYPESAFE_API_KEY` to run Jev. "
            "On molab, add it as a notebook secret. Locally, put it in `.env` and run "
            "`uvx --env-file .env marimo edit --sandbox claims.py`."
        ),
    )
    jev = TypeSafe("jev-latest", timeout=30, cache=False)
    jev_row = compare("Jev", dspy.Predict(CheckClaimWithConfidence), jev, jev_dollars, dev)
    return jev, jev_row


@app.cell(hide_code=True)
def _(jev_row):
    least_confident_first = sorted(
        ((example, pred) for example, pred, _ in jev_row["results"] if "verdict" in pred),
        key=lambda pair: pair[1].verdict.confidence,
    )
    claim_rows = [
        {
            "#": i,
            "claim": example.claim,
            "true verdict": example.verdict,
            "Jev": pred.verdict.value,
            "confidence": round(pred.verdict.confidence, 2),
        }
        for i, (example, pred) in enumerate(least_confident_first)
    ]
    return claim_rows, least_confident_first


@app.cell(hide_code=True)
def _(claim_rows):
    import altair as alt

    confidence_chart = alt.Chart(alt.Data(values=[
        {**row, "result": "Correct" if row["Jev"] == row["true verdict"] else "Incorrect"}
        for row in claim_rows
    ])).mark_bar(size=2).encode(
        x=alt.X("#:Q", title="Claims, from least to most confident", axis=alt.Axis(tickCount=5)),
        y=alt.Y("confidence:Q", title="Confidence", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("result:N", title=None, scale=alt.Scale(
            domain=["Correct", "Incorrect"], range=["#64748b", "#c4493d"],
        )),
        tooltip=[alt.Tooltip("claim:N", title="Claim"),
                 alt.Tooltip("Jev:N", title="Jev"),
                 alt.Tooltip("true verdict:N", title="Dataset label"),
                 alt.Tooltip("confidence:Q", title="Confidence", format=".2f")],
    ).properties(height=140, width="container")
    chart_view = mo.ui.altair_chart(confidence_chart, chart_selection=False, legend_selection=False)
    return (chart_view,)


@app.cell(hide_code=True)
def _(claim_rows):
    claim_picker = mo.ui.table(
        claim_rows,
        selection="single", initial_selection=[0], page_size=5,
        visible_columns=["claim", "confidence"],
        show_column_summaries=False, show_data_types=False,
        show_download=False, show_search=False,
        label="Select a claim (least confident first)",
    )
    return (claim_picker,)


@app.cell(hide_code=True)
def _(claim_picker, least_confident_first):
    _picked = claim_picker.value[0]["#"] if claim_picker.value else 0
    picked_example, picked_pred = least_confident_first[_picked]
    edited_claim = mo.ui.text_area(
        value=picked_example.claim, label="Edit the claim; click outside to evaluate", rows=2, full_width=True
    )
    return edited_claim, picked_example, picked_pred


@app.cell(hide_code=True)
def _(
    CheckClaimWithConfidence,
    edited_claim,
    jev,
    picked_example,
    picked_pred,
):
    if edited_claim.value.strip() == picked_example.claim.strip():
        edited_pred = picked_pred  # unchanged, so reuse Jev's answer from the initial run instead of asking again
    else:
        with dspy.context(lm=jev):
            edited_pred = dspy.Predict(CheckClaimWithConfidence)(claim=edited_claim.value, abstract=picked_example.abstract)
    return (edited_pred,)


@app.cell(hide_code=True)
def _(VERDICTS, edited_claim, edited_pred, picked_example, picked_pred):
    _before, _after = picked_pred.verdict, edited_pred.verdict
    _status = "Original answer" if edited_pred is picked_pred else "Edited answer"
    probability_table = mo.md(
        "| Verdict | Original | Edited |\n| :-- | --: | --: |\n" + "\n".join(
            f"| {verdict.replace('_', ' ')} | {_before.probabilities[verdict]:.1%} | {_after.probabilities[verdict]:.1%} |"
            for verdict in VERDICTS
        )
    )
    editor_panel = mo.vstack([
        edited_claim,
        mo.md(f"{_status}: **{_after.value.replace('_', ' ')}**. Original label: **{picked_example.verdict.replace('_', ' ')}**."),
        probability_table,
        mo.md(f"Current answer: ({max(_after.probabilities.values()):.1%} - 33.3%) / 66.7% "
              f"gives confidence **{_after.confidence:.2f}** (rounded)."),
        mo.accordion({"Source abstract": mo.Html(f"<p>{html.escape(picked_example.abstract)}</p>")}),
    ], gap=0.5)
    return (editor_panel,)


@app.cell(hide_code=True)
def _(chart_view, claim_picker, dev, editor_panel, jev_row):
    mo.vstack([
        mo.md(f"### Checking claims against abstracts\nJev checks {len(dev)} [SciFact](https://github.com/allenai/scifact) claim-abstract pairs through DSPy. Agreement with expert labels: **{jev_row['accuracy']:.1%}**; unanswered: {jev_row['unanswered']}."),
        chart_view,
        mo.md("Confidence rescales the largest of the three probabilities: **(largest probability - 1/3) / (2/3)**. "
              "An equal split gives 0; certainty gives 1. It is not the probability that the answer is correct."),
        mo.hstack([mo.vstack([claim_picker]).style({"min-width": "0", "overflow-x": "auto"}), editor_panel.style({"min-width": "0"})], widths=[1, 1], align="start", gap=1),
    ], gap=0.5)
    return


if __name__ == "__main__":
    app.run()
