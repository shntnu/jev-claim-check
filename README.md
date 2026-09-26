# Checking claims against abstracts

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/shntnu/jev-claim-check/blob/main/claims.py)

A [marimo](https://marimo.io) notebook that checks biomedical claims against the abstracts cited for them.
It uses Jev through [DSPy](https://dspy.ai) to classify 340 claim-abstract pairs from the development split of [SciFact](https://github.com/allenai/scifact). The three labels are supported, contradicted, and not enough information.

The notebook compares the model's answers with the dataset labels.
You can select uncertain answers for review and edit claims to compare the resulting probabilities against the same abstract.

## Run it

The notebook needs a TypeSafe API key as `TYPESAFE_API_KEY`, and each full pass costs about a cent on that key.
Without the key, it stops at a message explaining how to add one.

On molab, open the notebook from the badge above and add the key as a notebook secret.
Secrets are not copied into forks, so everyone who runs the notebook uses their own key.

Locally, put the key in a `.env` file next to the notebook and run:

```sh
uvx --env-file .env marimo edit --sandbox claims.py
```

The notebook declares its dependencies inline, so `--sandbox` installs them in an isolated environment.

Run `uv run check_demo.py` to check the review slider and claim-editing flow with synthetic predictions, without API calls.

## Data

SciFact (Wadden et al. 2020) is downloaded from AllenAI the first time the notebook runs, about 3 MB.
It is licensed under CC BY-NC 2.0.
