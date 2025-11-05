# Prompt Template Library

A small FastAPI service to manage and render prompt templates stored as JSON files.

This repository was created as part of a university project to explore prompt engineering patterns, template management, and a minimal API for rendering prompts with well-defined placeholders and token estimation.

## Project scope & goals

- Provide a tiny HTTP API to list, fetch and render prompt templates stored in `templates/`.
- Keep templates as JSON artifacts that declare their `id`, `name`, `description`, `version`, `placeholders` and `prompt` text.
- Use Python's `string.Template` for deterministic placeholder substitution and fail fast when required parameters are missing.
- Estimate token usage for rendered prompts (uses `tiktoken` where available) to help clients plan prompt costs.
- Be easy to run locally and portable across machines (simple requirements and a single entrypoint: `main.py`).

## Project structure

- `main.py` — FastAPI application exposing endpoints to list templates, fetch single templates, and render prompts.
- `templates/` — JSON files for each template. Filenames must match the template `id` (for example `summary_v1.json`).
- `requirements.txt` — Python dependencies used by the project.

## Template JSON shape and conventions

Each template JSON should include at least:

- `id` (string): unique id, also used as filename (e.g. `summary_v1.json`).
- `name` (string): human friendly name.
- `description` (string): short description of intent.
- `version` (string): semantic or simple version.
- `placeholders` (object): a map of placeholder names to descriptions (required keys for rendering).
- `prompt` (string): the actual prompt body using Python `string.Template` placeholders, e.g. `$input`, `$persona`.

Important: keep `$placeholder` tokens and do not switch substitution syntaxes. The server calls `string.Template.substitute(req.params)` and will return 400 if a placeholder is missing.

## Running locally (Windows PowerShell)

1. Create and activate a virtual environment:

```powershell
Set-Location 'C:\Users\lucab\Documents\LLMod Code'
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

3. Run the FastAPI app (development mode):

```powershell
uvicorn main:app --reload --port 8000
```

4. Open the automatic API docs:

Visit http://localhost:8000/docs to see the Swagger UI and try endpoints.

## API quick reference

- `GET /templates` — list available templates (reads `templates/*.json`).
- `GET /templates/{id}` — fetch the raw template JSON for `id`.
- `POST /render` — body example:

```json
{
	"template_id": "summary_v1",
	"params": { "input": "Long text to summarize" },
	"model_name": "gpt-4o-mini"
}
```

The response contains the rendered prompt and a token estimate.

## Development notes

- Token estimation uses `tiktoken` when available. The service falls back to `cl100k_base` encoding for unknown models.
- Templates must live under `templates/` and filename must equal `{id}.json`.
- If a required placeholder is missing in `params`, the render call will return HTTP 400.

## University project context

This repository was developed as part of the lecture "Large Language Models" at Technische Hochschule Nürnberg Georg Simon Ohm.

Course / Topic (as given by the professor):

Thema 2: Prompt-Patterns & Template-Bibliothek

Sie erläutern den Stand der Forschung zum Thema Prompts, Prompt-Engineering und Prompt-Patterns.
Sie sammeln funktionierende Prompt-Patterns für ausgewählte Aufgaben (z. B. Korrektur, Extraktion, Planung) und bauen eine kleine, getestete Bibliothek.

Ziel / Artefakt:

- Katalog (10–20 Patterns) mit Demo-Snippets und Einsatzhinweisen.

Praxis-Level (Bronze / Silber / Gold):

- Bronze: 10 Patterns mit Minimal-Tests sammeln.
- Silber: Einheitliche Template-API und A/B-Tests.
- Gold: Stil- vs. Substanz-Kontrolle getrennt messen.

English summary:

This project documents the state of research on prompts and prompt-engineering, collects working prompt patterns for selected tasks (e.g., correction, extraction, planning), and builds a small, tested template library. The intended artifact is a catalog of 10–20 patterns with demo snippets and usage guidance. Optional practice levels (Bronze/Silver/Gold) define increasingly advanced deliverables such as minimal tests, a unified template API, A/B tests, and separate measurements for style vs. substance.