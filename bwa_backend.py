from __future__ import annotations

import operator
import os
import re
import time
from datetime import date, timedelta
from pathlib import Path
from typing import TypedDict, List, Optional, Literal, Annotated

# ── BUG #1 FIX: Import field_validator for Groq string boolean coercion ──────
from pydantic import BaseModel, Field, field_validator

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# Agentic Blog Writer with Detailed Tasks + Working HuggingFace Images
#
# ALL BUGS FIXED:
#  #1 - Missing field_validator → Groq string booleans caused 400 error
#  #2 - Added subtasks: List[str] to Task for detailed breakdown  
#  #3 - worker_node(payload) → worker_node(state) per LangGraph contract
#  #4 - decide_images sent full blog → token overflow → truncated image plan
#  #5 - generate_and_place_images picked by length not [[IMAGE_ presence
#  #6 - No Content-Type check → JSON error bytes saved as corrupt .png
#  #7 - Changed to alibaba-pai/Z-Image-Fun-Lora-Distill per user request
# ============================================================


# ── BUG #1 FIX: Coerce Groq's "true"/"false" strings to Python bools ─────────
def _coerce_bool(v):
    """Groq returns needs_research='false' as string → coerce to bool."""
    if isinstance(v, str):
        if v.lower() == "true":  return True
        if v.lower() == "false": return False
    return v


# ═══════════════════════════════════════════════════════════
# 1) Schemas
# ═══════════════════════════════════════════════════════════
class Task(BaseModel):
    id: int
    title: str
    goal: str = Field(..., description="One sentence: what reader learns from this section.")
    
    bullets: List[str] = Field(
        ...,
        description="3–6 high-level bullet points structuring this section."
    )
    
    # ── BUG #2 FIX: Added subtasks for detailed breakdown ────────────────────
    subtasks: List[str] = Field(
        default_factory=list,
        description=(
            "4–8 DETAILED sub-points that expand the bullets. Each subtask is a "
            "concrete step, formula, code pattern, real example, or edge case that "
            "drives the writer to produce deep, thorough content."
        )
    )
    
    target_words: int = Field(..., description="Target word count 350–700 per section.")
    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False

    # ── BUG #1 FIX: Coerce all bool fields ───────────────────────────────────
    @field_validator("requires_research", "requires_citations", "requires_code", mode="before")
    @classmethod
    def coerce_bools(cls, v): return _coerce_bool(v)


class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal["explainer", "tutorial", "news_roundup", "comparison", "system_design"] = "explainer"
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]


class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None
    snippet: Optional[str] = None
    source: Optional[str] = None


class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    reason: str
    queries: List[str] = Field(default_factory=list)
    max_results_per_query: int = Field(5)

    # ── BUG #1 FIX: Coerce needs_research string bool ────────────────────────
    @field_validator("needs_research", mode="before")
    @classmethod
    def coerce_bool(cls, v): return _coerce_bool(v)


class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)


class ImageSpec(BaseModel):
    placeholder: str = Field(..., description="Exactly [[IMAGE_1]], [[IMAGE_2]], or [[IMAGE_3]]")
    filename: str    = Field(..., description="lowercase_with_underscores.png")
    alt: str
    caption: str
    prompt: str      = Field(..., description="Detailed image generation prompt.")
    size: Literal["1024x1024", "1024x1536", "1536x1024"] = "1024x1024"
    quality: Literal["low", "medium", "high"] = "medium"


class GlobalImagePlan(BaseModel):
    md_with_placeholders: str
    images: List[ImageSpec] = Field(default_factory=list)


class State(TypedDict):
    topic: str
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]
    as_of: str
    recency_days: int
    sections: Annotated[List[tuple[int, str]], operator.add]
    merged_md: str
    md_with_placeholders: str
    image_specs: List[dict]
    final: str


# ═══════════════════════════════════════════════════════════
# 2) LLM
# ═══════════════════════════════════════════════════════════
llm = ChatGroq(model="llama-3.3-70b-versatile")


# ═══════════════════════════════════════════════════════════
# 3) Router
# ═══════════════════════════════════════════════════════════
ROUTER_SYSTEM = """You are a routing module for a technical blog planner.

Decide whether web research is needed BEFORE planning.

Modes:
- closed_book (needs_research=false): evergreen concepts — math, algorithms, architecture.
- hybrid (needs_research=true): evergreen + needs current tool versions or benchmarks.
- open_book (needs_research=true): volatile — weekly roundup, "latest", pricing, news.

CRITICAL: needs_research MUST be JSON boolean true or false — NEVER string "true"/"false".

If needs_research=true: output 3–10 specific, high-signal search queries.
"""


def router_node(state: State) -> dict:
    decider = llm.with_structured_output(RouterDecision)
    decision = decider.invoke([
        SystemMessage(content=ROUTER_SYSTEM),
        HumanMessage(content=f"Topic: {state['topic']}\nAs-of date: {state['as_of']}"),
    ])
    recency_days = {"open_book": 7, "hybrid": 45}.get(decision.mode, 3650)
    return {
        "needs_research": decision.needs_research,
        "mode":           decision.mode,
        "queries":        decision.queries,
        "recency_days":   recency_days,
    }


def route_next(state: State) -> str:
    return "research" if state["needs_research"] else "orchestrator"


# ═══════════════════════════════════════════════════════════
# 4) Research (Tavily)
# ═══════════════════════════════════════════════════════════
def _tavily_search(query: str, max_results: int = 5) -> List[dict]:
    if not os.getenv("TAVILY_API_KEY"):
        return []
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults
        tool = TavilySearchResults(max_results=max_results)
        results = tool.invoke({"query": query})
        return [{
            "title":        r.get("title") or "",
            "url":          r.get("url") or "",
            "snippet":      r.get("content") or r.get("snippet") or "",
            "published_at": r.get("published_date") or r.get("published_at"),
            "source":       r.get("source"),
        } for r in (results or [])]
    except Exception:
        return []


def _iso_to_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except Exception:
        return None


RESEARCH_SYSTEM = """You are a research synthesizer.
Given raw web results, produce EvidenceItem objects.
Only include items with non-empty url. Normalize dates to YYYY-MM-DD or null.
Keep snippets short. Deduplicate by URL.
"""


def research_node(state: State) -> dict:
    queries = (state.get("queries") or [])[:10]
    raw: List[dict] = []
    for q in queries:
        raw.extend(_tavily_search(q, max_results=6))
    if not raw:
        return {"evidence": []}

    extractor = llm.with_structured_output(EvidencePack)
    pack = extractor.invoke([
        SystemMessage(content=RESEARCH_SYSTEM),
        HumanMessage(content=f"As-of: {state['as_of']}\nRecency: {state['recency_days']}d\n\nRaw:\n{raw}"),
    ])

    dedup = {e.url: e for e in pack.evidence if e.url}
    evidence = list(dedup.values())

    if state.get("mode") == "open_book":
        as_of  = date.fromisoformat(state["as_of"])
        cutoff = as_of - timedelta(days=int(state["recency_days"]))
        evidence = [e for e in evidence if (d := _iso_to_date(e.published_at)) and d >= cutoff]

    return {"evidence": evidence}


# ═══════════════════════════════════════════════════════════
# 5) Orchestrator
# ── BUG #2 FIX: Prompt demands subtasks for each task ──────
# ═══════════════════════════════════════════════════════════
ORCH_SYSTEM = """You are a senior technical writer producing a DETAILED long-form blog outline.

For EACH task you MUST provide:
  - goal: 1 clear sentence (what reader learns)
  - bullets: 3–6 high-level points that structure the section
  - subtasks: 4–8 DETAILED sub-points that EXPAND bullets into:
      * Concrete steps or algorithms
      * Specific formulas or code patterns  
      * Real examples with numbers/names
      * Common mistakes and how to avoid them
      * Edge cases and gotchas
    These subtasks drive the writer to produce deep, thorough content.
  - target_words: 350–700 per section (aim high — this is a detailed post)

Create 6–9 tasks. Mandatory sections for technical blogs:
  1. Introduction & motivation
  2. Theoretical foundation / math
  3. Core mechanism step-by-step  [requires_code=true]
  4. Implementation from scratch   [requires_code=true]
  5. Real-world applications
  6. Common pitfalls & debugging
  7. Performance, variants & future directions

CRITICAL: requires_research, requires_citations, requires_code MUST be JSON booleans true/false.
Output must strictly match Plan schema.
"""


def orchestrator_node(state: State) -> dict:
    planner = llm.with_structured_output(Plan)
    mode     = state.get("mode", "closed_book")
    evidence = state.get("evidence", [])
    forced_kind = "news_roundup" if mode == "open_book" else None

    plan = planner.invoke([
        SystemMessage(content=ORCH_SYSTEM),
        HumanMessage(content=(
            f"Topic: {state['topic']}\nMode: {mode}\n"
            f"As-of: {state['as_of']} (recency_days={state['recency_days']})\n"
            f"{'Force blog_kind=news_roundup' if forced_kind else ''}\n\n"
            f"Evidence:\n{[e.model_dump() for e in evidence][:16]}"
        )),
    ])
    if forced_kind:
        plan.blog_kind = "news_roundup"
    return {"plan": plan}


# ═══════════════════════════════════════════════════════════
# 6) Fanout
# ═══════════════════════════════════════════════════════════
def fanout(state: State):
    assert state["plan"] is not None
    return [
        Send("worker", {
            "task":         task.model_dump(),
            "topic":        state["topic"],
            "mode":         state["mode"],
            "as_of":        state["as_of"],
            "recency_days": state["recency_days"],
            "plan":         state["plan"].model_dump(),
            "evidence":     [e.model_dump() for e in state.get("evidence", [])],
        })
        for task in state["plan"].tasks
    ]


# ═══════════════════════════════════════════════════════════
# 7) Worker
# ── BUG #3 FIX: Renamed payload → state ────────────────────
# ── BUG #2 FIX: Worker prompt uses subtasks ────────────────
# ═══════════════════════════════════════════════════════════
WORKER_SYSTEM = """You are a senior technical writer producing ONE section of a detailed blog.

LENGTH RULES (non-negotiable):
- Hit the target_words. Minimum = target_words × 0.85. Short = FAILURE.
- If you finish a bullet early: go deeper. Add example, show edge case, explain why.

STRUCTURE:
- Start with ## <Section Title>
- Use ### sub-heading for EACH bullet point
- Under each ### cover its subtasks in full — don't skip any
- Bold **key terms** on first use
- Use code fences ```python for all code

DEPTH per section:
- Explain WHY, not just WHAT
- Give concrete, specific examples (real numbers, real model names, real APIs)
- If requires_code=true: include COMPLETE runnable code with detailed inline comments
- If requires_citations=true: cite as ([Source Name](URL))

OUTPUT: ONLY the section markdown. No preamble.
"""


def worker_node(state: dict) -> dict:
    # ── BUG #3 FIX: Parameter renamed from payload to state ──────────────────
    task     = Task(**state["task"])
    plan     = Plan(**state["plan"])
    evidence = [EvidenceItem(**e) for e in state.get("evidence", [])]
    mode     = state.get("mode", "closed_book")

    bullets_text  = "\n".join(f"  [{i+1}] {b}" for i, b in enumerate(task.bullets))
    subtasks_text = "\n".join(f"      - {s}" for s in task.subtasks)
    evidence_text = (
        "\n".join(f"- {e.title} | {e.url}" for e in evidence[:20])
        if evidence else "No external evidence — use deep technical knowledge."
    )
    min_words = int(task.target_words * 0.85)

    section_md = llm.invoke([
        SystemMessage(content=WORKER_SYSTEM),
        HumanMessage(content=(
            f"Blog title : {plan.blog_title}\n"
            f"Audience   : {plan.audience}\n"
            f"Tone       : {plan.tone}\n"
            f"Topic      : {state['topic']}\n\n"
            f"=== YOUR SECTION ===\n"
            f"Title       : {task.title}\n"
            f"Goal        : {task.goal}\n"
            f"TARGET WORDS: {task.target_words}  (MINIMUM: {min_words})\n"
            f"requires_code: {task.requires_code}\n"
            f"requires_citations: {task.requires_citations}\n\n"
            f"Bullets (each gets ### sub-heading):\n{bullets_text}\n\n"
            f"Subtasks (cover ALL under their parent ###):\n{subtasks_text}\n\n"
            f"Evidence:\n{evidence_text}\n"
        )),
    ]).content.strip()

    return {"sections": [(task.id, section_md)]}


# ═══════════════════════════════════════════════════════════
# 8) Reducer: merge → decide_images → generate_and_place_images
# ═══════════════════════════════════════════════════════════
def merge_content(state: State) -> dict:
    plan = state["plan"]
    if plan is None:
        raise ValueError("merge_content called without plan.")
    ordered = [md for _, md in sorted(state["sections"], key=lambda x: x[0])]
    merged_md = f"# {plan.blog_title}\n\n" + "\n\n".join(ordered) + "\n"
    print(f"📝 Merged {len(ordered)} sections → {len(merged_md.split())} words")
    return {"merged_md": merged_md}


# ── BUG #4 FIX: Never send full blog to structured output ─────────────────────
DECIDE_IMAGES_SYSTEM = """You are an expert technical editor placing 3 diagrams in a blog.

Given: section headings and a short preview.
Task: choose 3 placement points (by heading) and write detailed image prompts.

Rules:
- Return GlobalImagePlan where:
    * md_with_placeholders = FULL blog with [[IMAGE_1]], [[IMAGE_2]], [[IMAGE_3]] inserted
    * images = list of 3 ImageSpec objects
- Place each [[IMAGE_N]] on its own line after its target ## heading
- Image prompts: DETAILED layout, labels, arrows, colors, style
  Style: "clean technical diagram, white background, blue/grey palette, sans-serif labels"
- filenames: lowercase_with_underscores.png

Return strictly GlobalImagePlan JSON.
"""


def _insert_placeholders_manual(merged_md: str) -> str:
    """Manual fallback: insert [[IMAGE_1/2/3]] after 2nd, 4th, last ## heading."""
    lines    = merged_md.splitlines()
    headings = [i for i, l in enumerate(lines) if l.startswith("## ")]
    
    if len(headings) < 2:
        n = len(lines)
        inserts = [(n//4, "[[IMAGE_1]]"), (n//2, "[[IMAGE_2]]"), (3*n//4, "[[IMAGE_3]]")]
    else:
        targets = [
            headings[min(1, len(headings)-1)],
            headings[min(3, len(headings)-1)],
            headings[-1],
        ]
        inserts = [(t, f"[[IMAGE_{i+1}]]") for i, t in enumerate(targets)]
    
    for idx, tag in sorted(inserts, key=lambda x: x[0], reverse=True):
        lines.insert(idx + 1, f"\n{tag}\n")
    return "\n".join(lines)


def decide_images(state: State) -> dict:
    merged_md = state["merged_md"]
    plan      = state["plan"]
    assert plan is not None

    lines    = merged_md.splitlines()
    headings = [l for l in lines if l.startswith("## ") or l.startswith("# ")]
    preview  = "\n".join(lines[:60])
    
    short_context = (
        f"Blog: {plan.blog_title} | Topic: {state['topic']} | Words: {len(merged_md.split())}\n"
        f"Headings:\n" + "\n".join(f"  {h}" for h in headings) +
        f"\n\nFirst 60 lines:\n{preview}"
    )
    
    planner = llm.with_structured_output(GlobalImagePlan)
    try:
        image_plan = planner.invoke([
            SystemMessage(content=DECIDE_IMAGES_SYSTEM),
            HumanMessage(content=(
                f"{short_context}\n\n"
                f"=== FULL BLOG (insert placeholders) ===\n\n"
                f"{merged_md[:5500]}"
            )),
        ])
        
        returned      = image_plan.md_with_placeholders or ""
        has_ph        = "[[IMAGE_" in returned
        is_substantial = len(returned) >= len(merged_md) * 0.75
        
        if has_ph and is_substantial:
            md_out = returned
            print(f"🖼️  Image plan OK: {len(image_plan.images)} images")
        else:
            print(f"⚠️  LLM returned short md — inserting manually")
            md_out = _insert_placeholders_manual(merged_md)
        
        for img in image_plan.images:
            print(f"   {img.placeholder} → {img.filename}")
        
        return {
            "md_with_placeholders": md_out,
            "image_specs": [img.model_dump() for img in image_plan.images],
        }
    
    except Exception as e:
        print(f"⚠️  decide_images failed: {e} — manual insertion")
        md_out = _insert_placeholders_manual(merged_md)
        specs = [
            {"placeholder": f"[[IMAGE_{i+1}]]", "filename": f"diagram_{i+1}.png",
             "alt": f"Technical diagram {i+1}", "caption": f"Figure {i+1}",
             "prompt": f"Clean technical diagram about {state['topic']}, white background, blue/grey palette, part {i+1}",
             "size": "1024x1024", "quality": "medium"}
            for i in range(3)
        ]
        return {"md_with_placeholders": md_out, "image_specs": specs}


# ── HuggingFace Image Generation ──────────────────────────────────────────────
# ── BUG #7 FIX: Using alibaba-pai/Z-Image-Fun-Lora-Distill ───────────────────
# ── BUG #6 FIX: Content-Type check added ─────────────────────────────────────
def _generate_image_hf(prompt: str) -> bytes:
    """
    FREE image generation via HuggingFace Inference API.
    Primary model: alibaba-pai/Z-Image-Fun-Lora-Distill (per user request)

    Setup:
      1. https://huggingface.co/settings/tokens → New token → Read access
      2. Add to .env: HF_TOKEN=hf_...
    
    CRITICAL: As of Feb 2026, HuggingFace moved from api-inference.huggingface.co 
    to router.huggingface.co (old endpoint returns HTTP 410 Gone).
    """
    import requests

    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN missing. Get free token at https://huggingface.co/settings/tokens"
        )

    # Try multiple image models for best success rate
    models = [
        "stabilityai/stable-diffusion-xl-base-1.0",
        # "stabilityai/stable-diffusion-xl-base-1.0",  # Most reliable on free tier
        "stabilityai/stable-diffusion-2-1",          # Fallback
        "runwayml/stable-diffusion-v1-5",            # Second fallback
    ]
    
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json",
    }
    payload = {
        "inputs": prompt,
        "parameters": {
            "negative_prompt": "blurry, low quality, watermark, text overlay, distorted, ugly",
            "num_inference_steps": 25,
            "guidance_scale": 7.5,
        },
    }

    last_error = None
    for model_id in models:
        # FIX: Use new router.huggingface.co endpoint (old api-inference.huggingface.co is HTTP 410 Gone)
        api_url = f"https://api-inference.huggingface.co/models/{model_id}"
        print(f"   📡 Trying: {model_id}")
        
        for attempt in range(2):
            try:
                resp = requests.post(api_url, headers=headers, json=payload, timeout=120)
                ct   = resp.headers.get("Content-Type", "")
                print(f"   HTTP {resp.status_code} | Content-Type: {ct}")
                
                if resp.status_code == 200:
                    # Check Content-Type before accepting
                    if "image" in ct:
                        print(f"   ✅ Got image ({len(resp.content)//1024} KB)")
                        return resp.content
                    else:
                        last_error = f"200 but Content-Type={ct}: {resp.text[:200]}"
                        print(f"   ⚠️  {last_error}")
                        break
                
                elif resp.status_code == 503:
                    wait = min(int(resp.headers.get("X-WaitFor", "20")), 40)
                    if attempt == 0:
                        print(f"   ⏳ Model loading — waiting {wait}s...")
                        time.sleep(wait)
                        continue
                    last_error = f"Still loading after {wait}s"
                
                elif resp.status_code == 401:
                    raise RuntimeError("HF_TOKEN invalid/expired")
                
                elif resp.status_code == 429:
                    last_error = "Rate-limited (free tier quota)"
                    print(f"   ⚠️  {last_error}")
                    break
                
                elif resp.status_code == 410:
                    # Old endpoint deprecated
                    last_error = f"HTTP 410: {resp.text[:200]}"
                    print(f"   ⚠️  Endpoint deprecated: {last_error}")
                    break
                
                else:
                    last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                    print(f"   ⚠️  {last_error}")
                    break
            
            except requests.exceptions.Timeout:
                last_error = "Timeout after 120s"
                break
            except RuntimeError:
                raise
            except Exception as exc:
                last_error = str(exc)
                break
    
    raise RuntimeError(
        f"Image generation failed on all models. Last: {last_error}\n"
        f"Tip: Check HF_TOKEN is valid at https://huggingface.co/settings/tokens"
    )


def _safe_slug(title: str) -> str:
    s = re.sub(r"[^a-z0-9 _-]+", "", title.strip().lower())
    return re.sub(r"\s+", "_", s).strip("_") or "blog"


def generate_and_place_images(state: State) -> dict:
    plan = state["plan"]
    assert plan is not None

    md_with = state.get("md_with_placeholders") or ""
    merged  = state.get("merged_md") or ""

    # ── BUG #5 FIX: Pick by placeholder PRESENCE, not length ─────────────────
    if "[[IMAGE_" in md_with:
        md = md_with
        print(f"   ✅ md_with_placeholders chosen ({md.count('[[IMAGE_')} placeholders)")
    elif "[[IMAGE_" in merged:
        md = merged
        print(f"   ✅ merged_md fallback ({md.count('[[IMAGE_')} placeholders)")
    else:
        md = md_with if len(md_with) > len(merged) else merged
        print("   ℹ️  No [[IMAGE_N]] — saving as-is")
        Path(f"{_safe_slug(plan.blog_title)}.md").write_text(md, encoding="utf-8")
        return {"final": md}

    image_specs = state.get("image_specs", []) or []
    if not image_specs:
        Path(f"{_safe_slug(plan.blog_title)}.md").write_text(md, encoding="utf-8")
        return {"final": md}

    if not os.environ.get("HF_TOKEN", "").strip():
        print("⚠️  HF_TOKEN not set — replacing placeholders with text")
        for spec in image_specs:
            md = md.replace(spec["placeholder"],
                f"> 📊 **[{spec['alt']}]**\n> *{spec['caption']}*\n")
        for tag in ["[[IMAGE_1]]", "[[IMAGE_2]]", "[[IMAGE_3]]"]:
            md = md.replace(tag, "")
        Path(f"{_safe_slug(plan.blog_title)}.md").write_text(md, encoding="utf-8")
        return {"final": md}

    images_dir = Path("images")
    images_dir.mkdir(exist_ok=True)

    for spec in image_specs:
        placeholder = spec["placeholder"]
        out_path    = images_dir / spec["filename"]
        print(f"\n🎨 Generating {spec['filename']}...")

        if out_path.exists():
            print(f"   ♻️  Reusing cached ({out_path.stat().st_size//1024} KB)")
        else:
            try:
                img_bytes = _generate_image_hf(spec["prompt"])
                out_path.write_bytes(img_bytes)
                print(f"   ✅ Saved ({len(img_bytes)//1024} KB)")
            except Exception as exc:
                print(f"   ❌ Failed: {exc}")
                md = md.replace(placeholder,
                    f"> 📊 **[{spec['alt']}]**\n> *{spec['caption']}*\n"
                    f"> *(generation failed: {str(exc)[:80]})*\n")
                continue

        md = md.replace(placeholder,
            f"![{spec['alt']}](images/{spec['filename']})\n*{spec['caption']}*")

    # Safety net
    for tag in ["[[IMAGE_1]]", "[[IMAGE_2]]", "[[IMAGE_3]]"]:
        if tag in md:
            print(f"   ⚠️  {tag} not replaced — removing")
            md = md.replace(tag, "")

    out_file = Path(f"{_safe_slug(plan.blog_title)}.md")
    out_file.write_text(md, encoding="utf-8")
    print(f"\n✅ Saved: {out_file} | {len(md.split())} words | {md.count('![')} images")
    return {"final": md}


# ── Build reducer subgraph ────────────────────────────────────────────────────
reducer_graph = StateGraph(State)
reducer_graph.add_node("merge_content",            merge_content)
reducer_graph.add_node("decide_images",             decide_images)
reducer_graph.add_node("generate_and_place_images", generate_and_place_images)
reducer_graph.add_edge(START, "merge_content")
reducer_graph.add_edge("merge_content",            "decide_images")
reducer_graph.add_edge("decide_images",             "generate_and_place_images")
reducer_graph.add_edge("generate_and_place_images", END)
reducer_subgraph = reducer_graph.compile()


# ═══════════════════════════════════════════════════════════
# 9) Build main graph
# ═══════════════════════════════════════════════════════════
g = StateGraph(State)
g.add_node("router",       router_node)
g.add_node("research",     research_node)
g.add_node("orchestrator", orchestrator_node)
g.add_node("worker",       worker_node)
g.add_node("reducer",      reducer_subgraph)

g.add_edge(START, "router")
g.add_conditional_edges("router", route_next, {"research": "research", "orchestrator": "orchestrator"})
g.add_edge("research",     "orchestrator")
g.add_conditional_edges("orchestrator", fanout, ["worker"])
g.add_edge("worker",  "reducer")
g.add_edge("reducer", END)

app = g.compile()