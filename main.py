from typing import Union, Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import json
import os
import string

import tiktoken

app = FastAPI(
    title="Prompt Pattern Templating API",
    version="1.0.0",
    description="API zur Verwaltung und Ausführung von Prompt Templates"
)

LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
MODEL_NAME = "lmstudio-model"

class ExecuteRequest(BaseModel):
    template_id: str
    params: dict

class Template(BaseModel):
    id: str
    name: str
    description: str
    version: str
    prompt: str
    placeholders: Dict[str, str]

class TemplateSummary(BaseModel):
    id: str
    name: str
    description: str
    version: str

class RenderRequest(BaseModel):
    template_id: str
    params: Dict[str, str]
    model_name: str = "gpt-4o-mini"

class RenderResponse(BaseModel): 
    template_id: str
    rendered_prompt: str
    estimated_tokens: int

TEMPLATE_DIR = "templates"

def load_template(template_id: str) -> Template:
    path = os.path.join(TEMPLATE_DIR, f"{template_id}.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Template {template_id} nicht gefunden.")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Template(**data)

def list_templates() -> List[TemplateSummary]:
    result = []
    for file in os.listdir(TEMPLATE_DIR):
        if file.endswith(".json"):
            with open(os.path.join(TEMPLATE_DIR, file), "r", encoding="utf-8") as f:
                data = json.load(f)
                result.append(TemplateSummary(
                    id=data["id"],
                    name=data["name"],
                    description=data["description"],
                    version=data["version"]
                ))
    return result

def estimate_tokens(text: str, model_name: str = "gpt-4o-mini") -> int:
    try: 
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

# --- API Endpoints ---

@app.get("/")
def root():
    return {"message": "Template API is running. Visit /docs for Swagger UI."}

@app.get("/templates", response_model=List[TemplateSummary])
def get_templates():
    return list_templates()

@app.get("/templates/{template_id}")
def get_template(template_id: str):
    return load_template(template_id)

@app.post("/render", response_model=RenderResponse)
def render_prompt(req: RenderRequest):
    template = load_template(req.template_id)

    prompt_template = template.prompt
    safe_template = string.Template(prompt_template)

    try:
        rendered = safe_template.substitute(req.params)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Fehlender Platzhalter: {e}")

    #est_tokens = len(rendered.split())
    est_tokens = estimate_tokens(rendered, req.model_name)

    return RenderResponse(
        template_id=template.id,
        rendered_prompt=rendered,
        estimated_tokens=est_tokens
    )

