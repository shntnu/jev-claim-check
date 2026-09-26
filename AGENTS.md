# Project instructions

Use the TypeSafe skill at `.agents/skills/typesafe-ai/SKILL.md` when working on this project.
Read its live documentation references as needed for the task.
Use the installed marimo skills under `.agents/skills/` for applicable notebook work, starting with `marimo-notebook/SKILL.md` for authoring and editing.
Upstream references to the `anywidget` or `marimo-anywidget` skill refer to the installed `anywidget-generator` skill.

Install the project-local skill for Codex from a fresh clone with:

```sh
npx skills@1.5.20 add typesafe-ai/skills --skill typesafe-ai --agent codex -y
npx skills@1.5.20 add marimo-team/skills \
  -s add-molab-badge -s anywidget-generator -s auto-paper-demo \
  -s implement-paper -s implement-paper-auto -s jupyter-to-marimo \
  -s marimo-batch -s marimo-notebook -s streamlit-to-marimo \
  -s wasm-compatibility -a codex -y
```

Track `skills-lock.json` as the skill inventory and drift record.
The installer-owned skill directories are ignored; replay the commands above to restore them.
