# Checking claims against abstracts

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/shntnu/jev-claim-check/blob/main/claims.py)

A [marimo](https://marimo.io) notebook that checks biomedical claims against the abstracts cited for them.
It uses Jev through [DSPy](https://dspy.ai) to classify 340 claim-abstract pairs from the development split of [SciFact](https://github.com/allenai/scifact). The three labels are supported, contradicted, and not enough information.

The notebook compares the model's answers with the dataset labels.
Hover over the confidence chart to inspect individual answers, or edit a claim to compare the resulting probabilities against the same abstract.

Confidence rescales the highest probability relative to an equal split across the three labels: `(max_probability - 1/3) / (2/3)`.
For probabilities of 38%, 41%, and 21%, the model selects the 41% answer and the confidence is `(0.41 - 1/3) / (2/3) = 0.115`, displayed as 0.12.
This follows TypeSafe's published [Choice confidence formula](https://github.com/typesafe-ai/system-one-adapter-python/blob/main/src/system_one_adapter/_utils/confidence_metrics.py).
Confidence is not the probability that the answer is correct; the chart compares this score with agreement against the dataset labels.

## Run it

The notebook needs a TypeSafe API key as `TYPESAFE_API_KEY`, and each full pass costs about a cent on that key.
Without the key, it stops at a message explaining how to add one.

On molab, open the notebook from the badge above and add the key as a notebook secret.
Secrets are not copied into forks, so everyone who runs the notebook uses their own key.

Locally, put the key in a `.env` file next to the notebook and run:

```sh
uvx --env-file .env marimo run --sandbox claims.py
```

The notebook declares its dependencies inline, so `--sandbox` installs them in an isolated environment.
Use `marimo edit` instead of `marimo run` to work on the code.

Run `uv run check_demo.py` to check the confidence chart and claim-editing flow with synthetic predictions, without API calls.

## Data

SciFact (Wadden et al. 2020) is downloaded from AllenAI the first time the notebook runs, about 3 MB.
It is licensed under CC BY-NC 2.0.
