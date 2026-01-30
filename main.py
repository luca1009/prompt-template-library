from typing import Union, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import json
import os
import string

import tiktoken # Für Token-Schätzung

import requests # Für LM Studio API-Aufrufe

app = FastAPI(
    title="Prompt Pattern Templating API",
    version="1.0.0",
    description="API zur Verwaltung und Ausführung von Prompt Templates"
)

LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
MODEL_NAME = "lmstudio-model"

class ExecuteRequest(BaseModel):
    template_id: str
    params: Dict[str, str]
    model_name: str = MODEL_NAME

class RenderRequest(BaseModel):
    template_id: str
    params: Dict[str, str]
    model_name: str = MODEL_NAME

class Template(BaseModel):
    id: str
    name: str
    description: str
    version: str
    prompt: str = ""
    placeholders: Dict[str, str] = Field(default_factory=dict)
    system_prompt: str = ""
    user_prompt: str = ""
    temperature: float = 0.7

class TemplateSummary(BaseModel):
    id: str
    name: str
    description: str
    version: str

class RenderResponse(BaseModel): 
    template_id: str
    rendered_prompt: str
    system_prompt: str
    user_prompt: str
    estimated_tokens: int

class ExecuteResponse(BaseModel):
    template_id: str
    rendered_prompt: str 
    system_prompt: Optional[str] = None
    user_prompt: Optional[str] = None
    estimated_tokens: int
    model_name: str
    response_text: str

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

# LM Studio Integration
def execute_prompt(
        model_name: str, 
        system_prompt: str, 
        user_prompt: str,
        temperature: float
) -> str:
    
    messages = []
    if system_prompt and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature
    }

    print(payload)

    try:
        response = requests.post(LM_STUDIO_URL, json=payload, timeout=60)
        response.raise_for_status()
        j = response.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Fehler bei LM Studio Anfrage: {e}")
    
    response_text = ""
    if "choices" in j and len(j["choices"]) > 0:
        response_text = j["choices"][0]["message"]["content"]

    return response_text

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

    system_prompt = template.system_prompt or ""
    user_prompt = template.user_prompt or template.prompt or ""

    safe_template = string.Template(prompt_template)

    try:
        rendered = safe_template.substitute(req.params)
        rendered_system_prompt = string.Template(system_prompt).substitute(req.params) if system_prompt else ""
        rendered_user_prompt = string.Template(user_prompt).substitute(req.params)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Fehlender Platzhalter: {e}")

    full_prompt = f"System: {rendered_system_prompt}\nUser: {rendered_user_prompt}".strip()

    #est_tokens = len(rendered.split())
    est_tokens = estimate_tokens(full_prompt, req.model_name)

    return RenderResponse(
        template_id=template.id,
        rendered_prompt=full_prompt,
        system_prompt=rendered_system_prompt,
        user_prompt=rendered_user_prompt,
        estimated_tokens=est_tokens
    )

@app.post("/execute", response_model=ExecuteResponse)
def render_and_execute(req: ExecuteRequest):
    template = load_template(req.template_id)

    prompt_template = template.prompt 
    safe_template = string.Template(prompt_template)

    system_prompt = template.system_prompt or ""
    user_prompt = template.user_prompt or template.prompt or ""

    try:
        rendered = safe_template.substitute(req.params)
        rendered_system_prompt = string.Template(system_prompt).substitute(req.params) if system_prompt else ""
        rendered_user_prompt = string.Template(user_prompt).substitute(req.params)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Fehlender Platzhalter: {e}")
    
    full_prompt = f"System: {rendered_system_prompt}\nUser: {rendered_user_prompt}".strip()
    est_tokens = estimate_tokens(full_prompt, req.model_name)

    response_text = execute_prompt(
        model_name=req.model_name, 
        system_prompt=rendered_system_prompt, 
        user_prompt=rendered_user_prompt, 
        temperature=template.temperature)

    return ExecuteResponse(
        template_id=template.id,
        rendered_prompt=full_prompt,
        system_prompt=rendered_system_prompt,
        user_prompt=rendered_user_prompt,
        estimated_tokens=est_tokens,
        model_name=req.model_name,
        response_text=response_text
    )

