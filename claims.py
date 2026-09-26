# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "anywidget>=0.11.0",
#     "dspy[typesafe]>=3.4.0",
#     "marimo>=0.25.0",
#     "traitlets>=5.16.1",
# ]
# ///

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")

with app.setup:
    import html
    import io
    import json
    import os
    import tarfile
    import time
    import urllib.request
    from collections import Counter
    from typing import Literal, get_args

    import anywidget
    import dspy
    import marimo as mo
    import traitlets
    from dspy.experimental import Choice, TypeSafe


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Does this abstract support this claim?

    Jev, TypeSafe's System One model, judges whether a paper's abstract supports, contradicts, or says nothing about a scientific claim, through one DSPy signature.
    The data is [SciFact](https://github.com/allenai/scifact) (Wadden et al. 2020): expert-labeled biomedical claims paired with the abstracts cited for them.
    The notebook downloads it, about 3 MB, the first time it runs.

    Jev runs live on all 340 pairs when the notebook opens, which takes seconds and costs about a cent on your TypeSafe key.
    An LLM's result on the same pairs appears as a recorded reference, so the notebook never pays for an LLM.

    To run it, provide your TypeSafe API key as `TYPESAFE_API_KEY`.
    On molab, add it as a notebook secret.
    Locally, put it in a `.env` file next to this notebook and run `uvx --env-file .env marimo edit --sandbox claims.py`.
    """)
    return


@app.cell
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


@app.cell
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


@app.cell
def _(VERDICTS, load_dev):
    dev = load_dev()
    _counts = Counter(example.verdict for example in dev)
    mo.md(f"SciFact dev: **{len(dev)}** claim-abstract pairs, " + ", ".join(f"{_counts[v]} {v}" for v in VERDICTS) + ".")
    return (dev,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. Jev on every pair, next to a recorded LLM

    `CheckClaim` defines the task, and `compare` runs it on every pair and records accuracy, time, and cost.
    For Jev, the verdict is typed as a `Choice`, so each answer also carries Jev's confidence.
    The question Jev sees stays the same.
    Claude Haiku 4.5 answered the same pairs through the plain `CheckClaim` on 2026-09-25, and its numbers below are fixed from that run.
    """)
    return


@app.cell
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


@app.cell
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


@app.cell
def _():
    # Claude Haiku 4.5 answered the same 340 pairs through OpenRouter on 2026-09-25, using the plain CheckClaim
    # signature at temperature 0. Its result is kept as a fixed reference so the notebook never pays for an LLM.
    haiku_recorded = {
        "backend": "Claude Haiku 4.5 (recorded)",
        "accuracy": 278 / 340,
        "seconds": 49,
        "dollars": 0.439,
        "correct": Counter(supports=125, contradicts=64, not_enough_info=89),
        "total": Counter(supports=138, contradicts=71, not_enough_info=131),
        "unanswered": 0,
    }
    return (haiku_recorded,)


@app.cell
def _(dev, haiku_recorded, jev_row):
    _ref = haiku_recorded
    _gap = (jev_row["accuracy"] - _ref["accuracy"]) * 100
    mo.hstack(
        [
            mo.stat(
                f"{jev_row['accuracy']:.1%}",
                label="Jev accuracy",
                caption=f"{_gap:+.1f} pts vs {_ref['backend']} at {_ref['accuracy']:.1%}",
                direction="increase" if _gap >= 0 else "decrease",
                bordered=True,
            ),
            mo.stat(
                f"{jev_row['seconds']:.0f} s",
                label=f"Jev time for {len(dev)} checks",
                caption=f"{_ref['backend']} took {_ref['seconds']:.0f} s, {_ref['seconds'] / jev_row['seconds']:.1f}x as long",
                direction="decrease" if jev_row["seconds"] <= _ref["seconds"] else "increase",
                target_direction="decrease",
                bordered=True,
            ),
            mo.stat(
                f"${jev_row['dollars']:.3f}",
                label=f"Jev cost for {len(dev)} checks",
                caption=f"{_ref['backend']} cost ${_ref['dollars']:.2f}, {_ref['dollars'] / jev_row['dollars']:.0f}x as much",
                direction="decrease" if jev_row["dollars"] <= _ref["dollars"] else "increase",
                target_direction="decrease",
                bordered=True,
            ),
        ],
        widths="equal",
    )
    return


@app.cell
def _(VERDICTS, VerdictDumbbell, haiku_recorded, jev_row):
    _rows = [jev_row, haiku_recorded]
    mo.ui.anywidget(
        VerdictDumbbell(
            backends=[row["backend"] for row in _rows],
            rows=[
                {
                    "verdict": verdict,
                    "total": jev_row["total"][verdict],
                    "correct": {row["backend"]: row["correct"][verdict] for row in _rows},
                }
                for verdict in VERDICTS
            ],
        )
    )
    return


@app.cell
def _(VERDICTS, haiku_recorded, jev_row):
    mo.accordion(
        {
            "Table view": mo.ui.table(
                [
                    {
                        "backend": row["backend"],
                        "accuracy": f"{row['accuracy']:.1%}",
                        "seconds": round(row["seconds"]),
                        "cost": f"${row['dollars']:.3f}",
                        **{verdict: f"{row['correct'][verdict]}/{row['total'][verdict]}" for verdict in VERDICTS},
                        "unanswered": row["unanswered"],
                    }
                    for row in [jev_row, haiku_recorded]
                ],
                selection=None,
            )
        }
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. Jev's confidence decides what a person checks

    Each thin column below is one of Jev's verdicts from section 1, sorted from least to most confident, and its height is Jev's confidence.
    Red marks the verdicts Jev got wrong.
    Drag the line to choose how many of the least confident verdicts go to a person, and hover a column to read its claim.
    """)
    return


@app.cell
def _(jev_row):
    least_confident_first = sorted(
        ((example, pred) for example, pred, _ in jev_row["results"] if "verdict" in pred),
        key=lambda pair: pair[1].verdict.confidence,
    )
    return (least_confident_first,)


@app.cell
def _(TriageStrip, least_confident_first):
    triage = mo.ui.anywidget(
        TriageStrip(
            items=[
                {
                    "confidence": pred.verdict.confidence,
                    "wrong": pred.verdict.value != example.verdict,
                    "claim": example.claim,
                    "label": example.verdict,
                    "verdict": pred.verdict.value,
                }
                for example, pred in least_confident_first
            ],
            cut=round(0.2 * len(least_confident_first)),
        )
    )
    triage
    return (triage,)


@app.cell
def _(least_confident_first, triage):
    _sent = least_confident_first[: triage.value["cut"]]
    mo.ui.table(
        [
            {
                "claim": example.claim,
                "true verdict": example.verdict,
                "Jev": pred.verdict.value,
                "confidence": round(pred.verdict.confidence, 2),
            }
            for example, pred in _sent
        ],
        selection=None,
        label=f"The {len(_sent)} claims sent to a person, least confident first",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. Edit a claim and watch Jev's verdict

    Pick a claim, then change it in the box: reverse the finding, swap the population, drug, or an identifier, or paraphrase it.
    Jev judges the edited claim against the same abstract, and the chart shows how its probability for each verdict moved.
    A good checker flips when the meaning changes and holds steady when it doesn't.
    Jev's probabilities vary a little between calls, so treat a shift of a few points as noise.
    """)
    return


@app.cell
def _(least_confident_first):
    _rows = [
        {
            "#": i,
            "claim": example.claim,
            "true verdict": example.verdict,
            "Jev": pred.verdict.value,
            "confidence": round(pred.verdict.confidence, 2),
        }
        for i, (example, pred) in enumerate(least_confident_first)
    ]
    # Start on the metastatic colorectal cancer example when it is in this run's answers.
    _start = next((row["#"] for row in _rows if row["claim"].startswith("Metastatic colorectal cancer")), 0)
    claim_picker = mo.ui.table(
        _rows, selection="single", initial_selection=[_start], page_size=5, label="Pick a claim, least confident first"
    )
    claim_picker
    return (claim_picker,)


@app.cell
def _(claim_picker, least_confident_first):
    _picked = claim_picker.value[0]["#"] if claim_picker.value else 0
    picked_example, picked_pred = least_confident_first[_picked]
    edited_claim = mo.ui.text_area(
        value=picked_example.claim, label="Edit the claim, then click outside the box", rows=3, full_width=True
    )
    edited_claim
    return edited_claim, picked_example, picked_pred


@app.cell
def _(
    CheckClaimWithConfidence,
    edited_claim,
    jev,
    picked_example,
    picked_pred,
):
    if edited_claim.value.strip() == picked_example.claim.strip():
        edited_pred = picked_pred  # unchanged, so reuse Jev's answer from section 1 instead of asking again
    else:
        with dspy.context(lm=jev):
            edited_pred = dspy.Predict(CheckClaimWithConfidence)(claim=edited_claim.value, abstract=picked_example.abstract)
    return (edited_pred,)


@app.cell
def _(VERDICTS, VerdictShift, edited_pred, picked_example, picked_pred):
    _before, _after = picked_pred.verdict, edited_pred.verdict
    if edited_pred is picked_pred:
        _status = "Edit the claim above to see whether Jev's verdict moves."
    elif _after.value != _before.value:
        _status = f"For the edited claim, Jev's verdict flipped to **{_after.value}** with confidence {_after.confidence:.2f}."
    else:
        _status = f"For the edited claim, Jev's verdict held at **{_after.value}** with confidence {_after.confidence:.2f}."
    mo.vstack(
        [
            mo.md(
                f"The original claim is labeled **{picked_example.verdict}**, and Jev said **{_before.value}** "
                f"with confidence {_before.confidence:.2f}. {_status}"
            ),
            mo.ui.anywidget(VerdictShift(verdicts=list(VERDICTS), original=_before.probabilities, edited=_after.probabilities)),
            mo.accordion({"Abstract Jev reads": mo.Html(f"<p>{html.escape(picked_example.abstract)}</p>")}),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Appendix: widgets

    Three small [anywidget](https://anywidget.dev) views draw the charts above.
    Their JavaScript and CSS live in the cells below, so this notebook runs as a single file, including on molab.
    """)
    return


@app.cell(hide_code=True)
def _():
    VIZ_CSS = r"""
    /* Shared look for the notebook's widgets. Colors come from the dataviz reference palette. */

    .viz {
      color-scheme: light;
      --surface: #fcfcfb;
      --ink: #0b0b0b;
      --ink-2: #52514e;
      --muted: #898781;
      --grid: #e1e0d9;
      --axis: #c3c2b7;
      --border: rgba(11, 11, 11, 0.1);
      --region: rgba(11, 11, 11, 0.045);
      --wash: rgba(137, 135, 129, 0.16);
      --series-1: #2a78d6;
      --series-2: #eb6834;
      --critical: #d03b3b;

      position: relative;
      padding: 12px 14px 8px;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: var(--surface);
      color: var(--ink);
      font: 13px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif;
    }

    .viz[data-theme="dark"] {
      color-scheme: dark;
      --surface: #1a1a19;
      --ink: #ffffff;
      --ink-2: #c3c2b7;
      --grid: #2c2c2a;
      --axis: #383835;
      --border: rgba(255, 255, 255, 0.1);
      --region: rgba(255, 255, 255, 0.06);
      --wash: rgba(137, 135, 129, 0.22);
      --series-1: #3987e5;
      --series-2: #d95926;
    }

    @media (prefers-color-scheme: dark) {
      .viz:not([data-theme]) {
        color-scheme: dark;
        --surface: #1a1a19;
        --ink: #ffffff;
        --ink-2: #c3c2b7;
        --grid: #2c2c2a;
        --axis: #383835;
        --border: rgba(255, 255, 255, 0.1);
        --region: rgba(255, 255, 255, 0.06);
        --wash: rgba(137, 135, 129, 0.22);
        --series-1: #3987e5;
        --series-2: #d95926;
      }
    }

    .viz .title { font-weight: 600; margin-bottom: 4px; }
    .viz .legend { display: flex; flex-wrap: wrap; gap: 4px 16px; color: var(--ink-2); font-size: 12px; margin-bottom: 6px; }
    .viz .legend span { display: inline-flex; align-items: center; gap: 6px; }
    .viz .swatch { width: 10px; height: 10px; border-radius: 2px; }
    .viz .swatch.dot { border-radius: 50%; }
    .viz .swatch.area { box-shadow: inset 0 2px 0 var(--muted); }
    .viz .swatch.cut-key { width: 2px; height: 12px; border-radius: 1px; background: var(--ink); }

    .viz .stats { display: flex; justify-content: space-between; gap: 16px; margin-bottom: 6px; }
    .viz .stats > :last-child { text-align: right; }
    .viz .stat-label, .viz .stat-detail { color: var(--ink-2); font-size: 12px; }
    .viz .stat-value { font-size: 20px; font-weight: 600; }

    .viz svg { display: block; width: 100%; height: auto; overflow: visible; }
    .viz svg text { fill: var(--ink); font-size: 12px; }
    .viz svg .muted { fill: var(--muted); font-size: 11px; font-variant-numeric: tabular-nums; }
    .viz svg .secondary { fill: var(--ink-2); }
    .viz svg .grid { stroke: var(--grid); stroke-width: 1; }
    .viz svg .axis { stroke: var(--axis); stroke-width: 1; }
    .viz svg [tabindex]:focus { outline: none; }
    .viz svg [tabindex]:focus-visible { outline: 2px solid var(--series-1); outline-offset: 2px; }

    .viz .tip {
      position: absolute;
      z-index: 1;
      max-width: 340px;
      padding: 6px 8px;
      border: 1px solid var(--border);
      border-radius: 6px;
      background: var(--surface);
      box-shadow: 0 2px 10px rgba(0, 0, 0, 0.14);
      font-size: 12px;
      pointer-events: none;
    }
    .viz .tip[hidden] { display: none; }
    .viz .tip strong { display: block; font-size: 13px; }
    .viz .tip .sub { color: var(--ink-2); }
    .viz .tip .muted { color: var(--muted); }
    """
    return (VIZ_CSS,)


@app.cell(hide_code=True)
def _(VIZ_CSS):
    class VerdictDumbbell(anywidget.AnyWidget):
        """Accuracy by true verdict: one row per verdict, one dot per backend."""

        backends = traitlets.List().tag(sync=True)
        rows = traitlets.List().tag(sync=True)
        _css = VIZ_CSS
        _esm = r"""
    // Accuracy by true verdict: one row per verdict, one dot per backend.
    const SVG = "http://www.w3.org/2000/svg";

    function svg(tag, attrs, parent) {
      const node = document.createElementNS(SVG, tag);
      for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
      parent.append(node);
      return node;
    }

    // marimo records its theme on <body data-theme>. Copy it onto the widget, because the widget's CSS
    // lives in a shadow root and cannot see <body>. Outside marimo, the CSS falls back to the OS setting.
    function followTheme(root) {
      const apply = () => {
        const theme = document.body.dataset.theme;
        if (theme === "dark" || theme === "light") root.dataset.theme = theme;
        else delete root.dataset.theme;
      };
      apply();
      const observer = new MutationObserver(apply);
      observer.observe(document.body, { attributes: true, attributeFilter: ["data-theme"] });
      return () => observer.disconnect();
    }

    function render({ model, el }) {
      const root = document.createElement("div");
      root.className = "viz";
      el.append(root);
      const stopFollowingTheme = followTheme(root);

      function draw() {
        const backends = model.get("backends");
        const rows = model.get("rows");
        root.replaceChildren();

        const title = document.createElement("div");
        title.className = "title";
        title.textContent = "Accuracy by true verdict";
        root.append(title);

        if (backends.length > 1) {
          const legend = document.createElement("div");
          legend.className = "legend";
          backends.forEach((name, i) => {
            const swatch = document.createElement("i");
            swatch.className = "swatch dot";
            swatch.style.background = `var(--series-${i + 1})`;
            const item = document.createElement("span");
            item.append(swatch, document.createTextNode(name));
            legend.append(item);
          });
          root.append(legend);
        }

        const tip = document.createElement("div");
        tip.className = "tip";
        tip.hidden = true;

        const width = 640;
        const left = 150;
        const right = width - (backends.length > 1 ? 96 : 12);
        const top = 4;
        const rowHeight = 36;
        const plotBottom = top + rows.length * rowHeight;
        const height = plotBottom + 22;

        const shares = rows.flatMap((row) => backends.map((name) => row.correct[name] / row.total));
        const lo = Math.min(0.5, Math.floor(Math.min(...shares) * 10) / 10);
        const x = (share) => left + ((share - lo) / (1 - lo)) * (right - left);

        const chart = svg("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": "Accuracy by true verdict" }, root);

        for (let tick = lo; tick <= 1.0001; tick += 0.1) {
          svg("line", { class: "grid", x1: x(tick), x2: x(tick), y1: top, y2: plotBottom }, chart);
          const label = svg("text", { class: "muted", x: x(tick), y: plotBottom + 16, "text-anchor": "middle" }, chart);
          label.textContent = `${Math.round(tick * 100)}%`;
        }

        function showTip(target, value, detail) {
          const strong = document.createElement("strong");
          strong.textContent = value;
          const sub = document.createElement("div");
          sub.className = "sub";
          sub.textContent = detail;
          tip.replaceChildren(strong, sub);
          tip.hidden = false;
          const box = target.getBoundingClientRect();
          const frame = root.getBoundingClientRect();
          const tipLeft = box.left - frame.left + box.width / 2 - tip.offsetWidth / 2;
          tip.style.left = `${Math.max(4, Math.min(tipLeft, frame.width - tip.offsetWidth - 4))}px`;
          const above = box.top - frame.top - tip.offsetHeight - 6;
          tip.style.top = `${above > 0 ? above : box.bottom - frame.top + 6}px`;
        }
        const hideTip = () => (tip.hidden = true);

        rows.forEach((row, r) => {
          const y = top + r * rowHeight + rowHeight / 2;
          svg("text", { x: 0, y: y - 2 }, chart).textContent = row.verdict;
          svg("text", { class: "muted", x: 0, y: y + 12 }, chart).textContent = `${row.total} pairs`;

          const points = backends.map((backend, i) => ({ backend, i, correct: row.correct[backend], share: row.correct[backend] / row.total }));
          const xs = points.map((point) => x(point.share));
          svg("line", { x1: Math.min(...xs), x2: Math.max(...xs), y1: y, y2: y, stroke: "var(--axis)", "stroke-width": 2 }, chart);

          for (const point of points) {
            const cx = x(point.share);
            svg("circle", { cx, cy: y, r: 5, fill: `var(--series-${point.i + 1})`, stroke: "var(--surface)", "stroke-width": 2 }, chart);
            const hit = svg("circle", { cx, cy: y, r: 12, fill: "transparent", tabindex: 0 }, chart);
            const value = `${(point.share * 100).toFixed(1)}%`;
            const detail = `${point.backend}, ${point.correct} of ${row.total} ${row.verdict} pairs`;
            hit.addEventListener("pointerenter", () => showTip(hit, value, detail));
            hit.addEventListener("focus", () => showTip(hit, value, detail));
            hit.addEventListener("pointerleave", hideTip);
            hit.addEventListener("blur", hideTip);
          }

          if (points.length > 1) {
            const lead = (points[0].share - points[1].share) * 100;
            const text = svg("text", { class: "secondary", x: right + 14, y: y + 4 }, chart);
            text.textContent = `${points[0].backend} ${lead >= 0 ? "+" : ""}${lead.toFixed(1)} pts`;
          }
        });

        root.append(tip);
      }

      draw();
      model.on("change:rows", draw);
      model.on("change:backends", draw);
      return () => stopFollowingTheme();
    }

    export default { render };
    """

    return (VerdictDumbbell,)


@app.cell(hide_code=True)
def _(VIZ_CSS):
    class TriageStrip(anywidget.AnyWidget):
        """Jev's verdicts sorted by confidence; `cut` counts the least confident ones sent to a person."""

        items = traitlets.List().tag(sync=True)
        cut = traitlets.Int(0).tag(sync=True)
        _css = VIZ_CSS
        _esm = r"""
    // Jev's verdicts sorted from least to most confident. The cut sends the least confident ones to a person.
    const SVG = "http://www.w3.org/2000/svg";

    function svg(tag, attrs, parent) {
      const node = document.createElementNS(SVG, tag);
      for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
      parent.append(node);
      return node;
    }

    // marimo records its theme on <body data-theme>. Copy it onto the widget, because the widget's CSS
    // lives in a shadow root and cannot see <body>. Outside marimo, the CSS falls back to the OS setting.
    function followTheme(root) {
      const apply = () => {
        const theme = document.body.dataset.theme;
        if (theme === "dark" || theme === "light") root.dataset.theme = theme;
        else delete root.dataset.theme;
      };
      apply();
      const observer = new MutationObserver(apply);
      observer.observe(document.body, { attributes: true, attributeFilter: ["data-theme"] });
      return () => observer.disconnect();
    }

    function html(tag, className, text) {
      const node = document.createElement(tag);
      if (className) node.className = className;
      if (text) node.textContent = text;
      return node;
    }

    function stat(label) {
      const block = html("div", "stat");
      const value = html("div", "stat-value");
      const detail = html("div", "stat-detail");
      block.append(html("div", "stat-label", label), value, detail);
      return { block, value, detail };
    }

    function legendItem(color, label, className = "swatch") {
      const swatch = html("i", className);
      swatch.style.background = color;
      const item = html("span");
      item.append(swatch, document.createTextNode(label));
      return item;
    }

    function render({ model, el }) {
      const items = model.get("items");
      const n = items.length;
      const wrongBefore = [0];
      for (const item of items) wrongBefore.push(wrongBefore.at(-1) + (item.wrong ? 1 : 0));
      const mistakes = wrongBefore[n];

      const controller = new AbortController();
      const { signal } = controller;
      const root = html("div", "viz");
      el.append(root);
      const stopFollowingTheme = followTheme(root);

      const sent = stat("Sent to a person");
      const kept = stat("Jev decides alone");
      const stats = html("div", "stats");
      stats.append(sent.block, kept.block);

      const legend = html("div", "legend");
      const handleKey = html("span");
      handleKey.append(html("i", "swatch cut-key"), document.createTextNode("cut: drag it, or focus it and use the arrow keys"));
      legend.append(legendItem("var(--wash)", "Jev right", "swatch area"), legendItem("var(--critical)", `Jev wrong (${mistakes})`), handleKey);
      root.append(stats, legend);

      const width = 680;
      const left = 36;
      const right = width - 10;
      const top = 16;
      const bottom = 156;
      const height = 192;
      const x = (i) => left + (i / n) * (right - left);
      const y = (confidence) => bottom - confidence * (bottom - top);
      const chart = svg("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": "Jev's verdicts sorted by confidence" }, root);

      const region = svg("rect", { x: left, y: top, width: 0, height: bottom - top, fill: "var(--region)" }, chart);
      for (const confidence of [0.5, 1]) {
        svg("line", { class: "grid", x1: left, x2: right, y1: y(confidence), y2: y(confidence) }, chart);
      }
      for (const confidence of [0, 0.5, 1]) {
        const label = svg("text", { class: "muted", x: left - 8, y: y(confidence) + 4, "text-anchor": "end" }, chart);
        label.textContent = confidence.toFixed(1);
      }
      const yCaption = svg("text", { class: "muted", transform: `translate(8 ${(top + bottom) / 2}) rotate(-90)`, "text-anchor": "middle" }, chart);
      yCaption.textContent = "confidence";

      // Every verdict as a step of the confidence curve, then Jev's mistakes on top in red.
      let curve = "";
      items.forEach((item, i) => (curve += ` L ${x(i)} ${y(item.confidence)} L ${x(i + 1)} ${y(item.confidence)}`));
      svg("path", { d: `M ${x(0)} ${bottom}${curve} L ${x(n)} ${bottom} Z`, fill: "var(--wash)" }, chart);
      svg("path", { d: `M${curve.slice(2)}`, fill: "none", stroke: "var(--muted)", "stroke-width": 1.5 }, chart);
      const barWidth = Math.max(1.5, x(1) - x(0));
      items.forEach((item, i) => {
        if (item.wrong) {
          svg("rect", { x: x(i), y: y(item.confidence), width: barWidth, height: bottom - y(item.confidence), fill: "var(--critical)" }, chart);
        }
      });
      svg("line", { class: "axis", x1: left, x2: right, y1: bottom, y2: bottom }, chart);

      for (const share of [0, 0.25, 0.5, 0.75, 1]) {
        const anchor = share === 0 ? "start" : share === 1 ? "end" : "middle";
        const label = svg("text", { class: "muted", x: x(share * n), y: bottom + 14, "text-anchor": anchor }, chart);
        label.textContent = `${share * 100}%`;
      }
      const xCaption = svg("text", { class: "muted", x: (left + right) / 2, y: bottom + 30, "text-anchor": "middle" }, chart);
      xCaption.textContent = "share of verdicts, least confident first";

      const hoverLine = svg("line", { y1: top, y2: bottom, stroke: "var(--ink-2)", "stroke-width": 1, visibility: "hidden" }, chart);
      const cutLine = svg("line", { y1: top - 8, y2: bottom, stroke: "var(--ink)", "stroke-width": 2 }, chart);
      const handle = svg("circle", {
        cy: top - 8, r: 6, fill: "var(--ink)", stroke: "var(--surface)", "stroke-width": 2, tabindex: 0,
        role: "slider", "aria-label": "Verdicts sent to a person", "aria-valuemin": 0, "aria-valuemax": n,
      }, chart);
      chart.style.cursor = "ew-resize";
      chart.style.touchAction = "none";

      const tip = html("div", "tip");
      tip.hidden = true;
      root.append(tip);

      let cut = model.get("cut");

      function update(next) {
        cut = Math.max(0, Math.min(n, Math.round(next)));
        const cx = x(cut);
        region.setAttribute("width", cx - left);
        cutLine.setAttribute("x1", cx);
        cutLine.setAttribute("x2", cx);
        handle.setAttribute("cx", cx);
        handle.setAttribute("aria-valuenow", cut);

        const caught = wrongBefore[cut];
        sent.value.textContent = `${cut} verdicts`;
        sent.detail.textContent = `${Math.round((100 * cut) / n)}% of all verdicts, holding ${caught} of Jev's ${mistakes} mistakes`;
        const rest = n - cut;
        if (rest > 0) {
          kept.value.textContent = `${((100 * (rest - (mistakes - caught))) / rest).toFixed(1)}% correct`;
          kept.detail.textContent = `${rest} verdicts with confidence ${items[cut].confidence.toFixed(2)} or higher`;
        } else {
          kept.value.textContent = "none";
          kept.detail.textContent = "every verdict goes to a person";
        }
      }

      function commit() {
        if (model.get("cut") !== cut) {
          model.set("cut", cut);
          model.save_changes();
        }
      }

      // Pointer position as a fractional verdict index.
      function indexAt(event) {
        const box = chart.getBoundingClientRect();
        const svgX = ((event.clientX - box.left) / box.width) * width;
        return ((svgX - left) / (right - left)) * n;
      }

      function showItem(i, event) {
        const item = items[i];
        const center = x(i) + barWidth / 2;
        hoverLine.setAttribute("x1", center);
        hoverLine.setAttribute("x2", center);
        hoverLine.setAttribute("visibility", "visible");
        tip.replaceChildren(
          html("strong", "", `Jev: ${item.verdict}, ${item.wrong ? "wrong" : "right"}`),
          html("div", "sub", item.claim),
          html("div", "muted", `True verdict: ${item.label}. Confidence ${item.confidence.toFixed(2)}.`),
        );
        tip.hidden = false;
        const frame = root.getBoundingClientRect();
        const px = event.clientX - frame.left;
        const py = event.clientY - frame.top;
        const tipLeft = px + 14 + tip.offsetWidth > frame.width ? px - 14 - tip.offsetWidth : px + 14;
        tip.style.left = `${Math.max(4, tipLeft)}px`;
        tip.style.top = `${Math.max(4, py - tip.offsetHeight - 12)}px`;
      }

      function hideItem() {
        tip.hidden = true;
        hoverLine.setAttribute("visibility", "hidden");
      }

      let dragging = false;
      chart.addEventListener("pointerdown", (event) => {
        dragging = true;
        chart.setPointerCapture(event.pointerId);
        hideItem();
        update(indexAt(event));
      }, { signal });
      chart.addEventListener("pointermove", (event) => {
        if (dragging) return update(indexAt(event));
        const i = Math.floor(indexAt(event));
        if (i >= 0 && i < n) showItem(i, event);
        else hideItem();
      }, { signal });
      const release = () => {
        if (!dragging) return;
        dragging = false;
        commit();
      };
      chart.addEventListener("pointerup", release, { signal });
      chart.addEventListener("pointercancel", release, { signal });
      chart.addEventListener("pointerleave", hideItem, { signal });

      handle.addEventListener("keydown", (event) => {
        const step = event.shiftKey ? 10 : 1;
        const moves = { ArrowLeft: -step, ArrowRight: step, Home: -n, End: n };
        if (!(event.key in moves)) return;
        event.preventDefault();
        update(cut + moves[event.key]);
        commit();
      }, { signal });

      model.on("change:cut", () => update(model.get("cut")));
      update(cut);

      return () => {
        controller.abort();
        stopFollowingTheme();
      };
    }

    export default { render };
    """

    return (TriageStrip,)


@app.cell(hide_code=True)
def _(VIZ_CSS):
    class VerdictShift(anywidget.AnyWidget):
        """Jev's probability for each verdict, for the original claim and an edited one."""

        verdicts = traitlets.List().tag(sync=True)
        original = traitlets.Dict().tag(sync=True)
        edited = traitlets.Dict().tag(sync=True)
        _css = VIZ_CSS
        _esm = r"""
    // Jev's probability for each verdict, for the original claim and an edited one.
    const SVG = "http://www.w3.org/2000/svg";

    const SERIES = [
      { key: "original", label: "original claim", color: "var(--muted)" },
      { key: "edited", label: "edited claim", color: "var(--series-1)" },
    ];

    function svg(tag, attrs, parent) {
      const node = document.createElementNS(SVG, tag);
      for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
      parent.append(node);
      return node;
    }

    // marimo records its theme on <body data-theme>. Copy it onto the widget, because the widget's CSS
    // lives in a shadow root and cannot see <body>. Outside marimo, the CSS falls back to the OS setting.
    function followTheme(root) {
      const apply = () => {
        const theme = document.body.dataset.theme;
        if (theme === "dark" || theme === "light") root.dataset.theme = theme;
        else delete root.dataset.theme;
      };
      apply();
      const observer = new MutationObserver(apply);
      observer.observe(document.body, { attributes: true, attributeFilter: ["data-theme"] });
      return () => observer.disconnect();
    }

    function render({ model, el }) {
      const root = document.createElement("div");
      root.className = "viz";
      el.append(root);
      const stopFollowingTheme = followTheme(root);

      function draw() {
        const verdicts = model.get("verdicts");
        const probabilities = { original: model.get("original"), edited: model.get("edited") };
        root.replaceChildren();

        const title = document.createElement("div");
        title.className = "title";
        title.textContent = "Jev's probability for each verdict";
        const legend = document.createElement("div");
        legend.className = "legend";
        for (const series of SERIES) {
          const swatch = document.createElement("i");
          swatch.className = "swatch dot";
          swatch.style.background = series.color;
          const item = document.createElement("span");
          item.append(swatch, document.createTextNode(series.label));
          legend.append(item);
        }
        root.append(title, legend);

        const tip = document.createElement("div");
        tip.className = "tip";
        tip.hidden = true;

        const width = 640;
        const left = 150;
        const right = width - 96;
        const top = 4;
        const rowHeight = 32;
        const plotBottom = top + verdicts.length * rowHeight;
        const height = plotBottom + 22;
        const x = (p) => left + p * (right - left);
        const chart = svg("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": "Jev's probability for each verdict" }, root);

        for (const tick of [0, 0.25, 0.5, 0.75, 1]) {
          svg("line", { class: "grid", x1: x(tick), x2: x(tick), y1: top, y2: plotBottom }, chart);
          const label = svg("text", { class: "muted", x: x(tick), y: plotBottom + 16, "text-anchor": "middle" }, chart);
          label.textContent = tick.toFixed(2);
        }

        function showTip(target, value, detail) {
          const strong = document.createElement("strong");
          strong.textContent = value;
          const sub = document.createElement("div");
          sub.className = "sub";
          sub.textContent = detail;
          tip.replaceChildren(strong, sub);
          tip.hidden = false;
          const box = target.getBoundingClientRect();
          const frame = root.getBoundingClientRect();
          const tipLeft = box.left - frame.left + box.width / 2 - tip.offsetWidth / 2;
          tip.style.left = `${Math.max(4, Math.min(tipLeft, frame.width - tip.offsetWidth - 4))}px`;
          const above = box.top - frame.top - tip.offsetHeight - 6;
          tip.style.top = `${above > 0 ? above : box.bottom - frame.top + 6}px`;
        }
        const hideTip = () => (tip.hidden = true);

        verdicts.forEach((verdict, r) => {
          const y = top + r * rowHeight + rowHeight / 2;
          svg("text", { x: 0, y: y + 4 }, chart).textContent = verdict;
          const before = probabilities.original[verdict];
          const after = probabilities.edited[verdict];
          svg("line", { x1: x(Math.min(before, after)), x2: x(Math.max(before, after)), y1: y, y2: y, stroke: "var(--axis)", "stroke-width": 2 }, chart);

          for (const series of SERIES) {
            const p = probabilities[series.key][verdict];
            svg("circle", { cx: x(p), cy: y, r: 5, fill: series.color, stroke: "var(--surface)", "stroke-width": 2 }, chart);
            const hit = svg("circle", { cx: x(p), cy: y, r: 12, fill: "transparent", tabindex: 0 }, chart);
            const detail = `${series.label}, ${verdict}`;
            hit.addEventListener("pointerenter", () => showTip(hit, p.toFixed(2), detail));
            hit.addEventListener("focus", () => showTip(hit, p.toFixed(2), detail));
            hit.addEventListener("pointerleave", hideTip);
            hit.addEventListener("blur", hideTip);
          }

          const change = svg("text", { class: "secondary", x: right + 14, y: y + 4 }, chart);
          change.textContent = `${before.toFixed(2)} to ${after.toFixed(2)}`;
        });

        root.append(tip);
      }

      draw();
      model.on("change:verdicts", draw);
      model.on("change:original", draw);
      model.on("change:edited", draw);
      return () => stopFollowingTheme();
    }

    export default { render };
    """

    return (VerdictShift,)


if __name__ == "__main__":
    app.run()
