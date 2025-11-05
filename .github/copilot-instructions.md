## Purpose
Provide concise, actionable guidance for AI coding agents working in this repository so they can be immediately productive.

## Quick summary (what this project is)
- A small FastAPI service that manages prompt templates stored as JSON in the `templates/` directory and exposes endpoints to list, fetch and render templates (`/templates`, `/templates/{id}`, `/render`). See `main.py` for implementation.

## Key files
- `main.py` — FastAPI app, template loading, rendering, and token estimation (uses `tiktoken`).
- `templates/*.json` — Template artifacts. Filenames must match the template `id` (e.g. `summary_v1.json`).

## Project-specific conventions and patterns
- Template JSON shape (discoverable in `templates/template.json`): must include `id`, `name`, `description`, `version`, `placeholders` and `prompt`.
- Filenames: template file must be saved at `templates/{id}.json`. The server loads templates by `template_id` → `{id}.json`.
- Placeholder syntax: prompts use Python `string.Template` placeholders (e.g. `$input`, `$persona`). Agents editing templates must preserve `$placeholder` tokens and not switch to other substitution syntaxes.
- Rendering behavior: `main.py` calls `string.Template.substitute(req.params)` and will raise a 400 HTTP error if a placeholder is missing. Do not remove explicit placeholder definitions from `placeholders` object.

## Integration points and external dependencies
- `tiktoken` is used to estimate token counts (fallback to `cl100k_base` encoding when model not known). Keep token-estimation calls in sync with models used in downstream systems.
- `LM_STUDIO_URL` and `MODEL_NAME` constants in `main.py` indicate an intended integration with a local LM Studio endpoint. The code does not currently call that URL — if adding calls, follow current config variables and make URL/credentials configurable.

## How to run locally (developer workflows)
1. Install dependencies from `requirements.txt` into your Python environment.
2. Start the API for development with uvicorn (example):

   powershell:
   ```powershell
   uvicorn main:app --reload --port 8000
   ```

3. Visit `http://localhost:8000/docs` for the auto-generated Swagger UI.

Example render request body (use the `summary_v1` template):
```json
{
  "template_id": "summary_v1",
  "params": { "input": "Long text to summarize" },
  "model_name": "gpt-4o-mini"
}
```

## Error modes agents should be aware of
- Missing template file -> 404 (raised by `load_template`).
- Missing placeholder -> 400 (raised in `render_prompt` when `substitute` fails).

## Editing and PR guidance for agents
- When modifying templates, update the `placeholders` map to reflect required keys and preserve prompt formatting and line breaks.
- Keep `id` stable for existing templates; if changing `id`, rename the file accordingly and update any external references.
- If you add model-specific behavior, ensure token estimation remains consistent and document any changes to `MODEL_NAME` defaults.

## Useful examples in repo
- `templates/summary_v1.json` — canonical summary template.
- `templates/recipe.json` — example of procedural prompt pattern.

## Notes for future automation
- There are currently no unit tests or CI config in the repo. If adding generation or update automation for templates, include a basic validation step that asserts:
  - JSON schema presence of `id`, `prompt`, `placeholders`.
  - Filename equals `id + .json`.

If any part of this instruction is unclear or you'd like more examples (cURL/PowerShell samples, validation script), say which area to expand and I'll iterate.
