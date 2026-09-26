# Does this abstract support this claim?

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/shntnu/jev-claim-check/blob/main/claims.py)

A [marimo](https://marimo.io) notebook that checks biomedical claims against the abstracts cited for them.
One [DSPy](https://dspy.ai) signature runs on Jev, [TypeSafe](https://typesafe.ai)'s System One model, over the 340 claim-abstract pairs in [SciFact](https://github.com/allenai/scifact)'s dev split.

- Jev answers every pair in seconds, for about a cent, and a recorded Claude Haiku 4.5 run on the same pairs gives a reference point.
- A draggable strip sorts Jev's verdicts by confidence, so you can choose how many of the least confident go to a person.
- A text box lets you edit any claim and see whether Jev's verdict flips against the same abstract.

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

## Data

SciFact (Wadden et al. 2020) is downloaded from AllenAI the first time the notebook runs, about 3 MB.
It is licensed under CC BY-NC 2.0.
