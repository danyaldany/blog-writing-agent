from __future__ import annotations
import json
import os
import re
import zipfile
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional, List, Iterator, Tuple

import pandas as pd
import streamlit as st

from bwa_backend import app


# ═══════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════
def safe_slug(title: str) -> str:
    s = re.sub(r"[^a-z0-9 _-]+", "", title.strip().lower())
    return re.sub(r"\s+", "_", s).strip("_") or "blog"


def bundle_zip(md_text: str, md_filename: str, images_dir: Path) -> bytes:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(md_filename, md_text.encode("utf-8"))
        if images_dir.exists() and images_dir.is_dir():
            for p in images_dir.rglob("*"):
                if p.is_file():
                    z.write(p, arcname=str(p))
    return buf.getvalue()


def images_zip(images_dir: Path) -> Optional[bytes]:
    if not images_dir.exists() or not images_dir.is_dir():
        return None
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in images_dir.rglob("*"):
            if p.is_file():
                z.write(p, arcname=str(p))
    return buf.getvalue()


# ── BUG #8 FIX: Stream-only try_stream (no double invoke) ────────────────────
def try_stream(graph_app, inputs: Dict[str, Any]) -> Iterator[Tuple[str, Any]]:
    """
    Stream graph updates. Yields ("updates"|"values", payload) during execution,
    then ("final", full_state) once complete.
    Never calls invoke() twice — avoids running graph twice.
    """
    last_state: Dict[str, Any] = {}

    try:
        for step in graph_app.stream(inputs, stream_mode="updates"):
            last_state.update(_flatten_step(step))
            yield ("updates", step)
        final = {**inputs, **last_state}
        yield ("final", final)
        return
    except Exception:
        pass

    try:
        for step in graph_app.stream(inputs, stream_mode="values"):
            if isinstance(step, dict):
                last_state.update(step)
            yield ("values", step)
        final = {**inputs, **last_state}
        yield ("final", final)
        return
    except Exception:
        pass

    # Last resort: plain invoke (only if both stream modes fail)
    out = graph_app.invoke(inputs)
    yield ("final", out)


def _flatten_step(step: Any) -> Dict[str, Any]:
    """Extract inner dict from single-key node-update step."""
    if isinstance(step, dict) and len(step) == 1:
        inner = next(iter(step.values()))
        if isinstance(inner, dict):
            return inner
    return step if isinstance(step, dict) else {}


def extract_latest_state(current_state: Dict[str, Any], step_payload: Any) -> Dict[str, Any]:
    if isinstance(step_payload, dict):
        if len(step_payload) == 1 and isinstance(next(iter(step_payload.values())), dict):
            current_state.update(next(iter(step_payload.values())))
        else:
            current_state.update(step_payload)
    return current_state


# ═══════════════════════════════════════════════════════════
# Markdown renderer with local image support
# ═══════════════════════════════════════════════════════════
_MD_IMG_RE       = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)")
_CAPTION_LINE_RE = re.compile(r"^\*(?P<cap>.+)\*$")


def _resolve_image_path(src: str) -> Path:
    return Path(src.strip().lstrip("./")).resolve()


def render_markdown_with_local_images(md: str):
    matches = list(_MD_IMG_RE.finditer(md))
    if not matches:
        st.markdown(md, unsafe_allow_html=False)
        return

    parts: List[Tuple[str, str]] = []
    last = 0
    for m in matches:
        before = md[last: m.start()]
        if before:
            parts.append(("md", before))
        alt = (m.group("alt") or "").strip()
        src = (m.group("src") or "").strip()
        parts.append(("img", f"{alt}|||{src}"))
        last = m.end()
    if md[last:]:
        parts.append(("md", md[last:]))

    i = 0
    while i < len(parts):
        kind, payload = parts[i]
        if kind == "md":
            st.markdown(payload, unsafe_allow_html=False)
            i += 1
            continue

        alt, src = payload.split("|||", 1)
        caption = None
        if i + 1 < len(parts) and parts[i + 1][0] == "md":
            nxt = parts[i + 1][1].lstrip()
            if nxt.strip():
                first_line = nxt.splitlines()[0].strip()
                mcap = _CAPTION_LINE_RE.match(first_line)
                if mcap:
                    caption = mcap.group("cap").strip()
                    parts[i + 1] = ("md", "\n".join(nxt.splitlines()[1:]))

        if src.startswith("http://") or src.startswith("https://"):
            st.image(src, caption=caption or alt or None, use_container_width=True)
        else:
            img_path = _resolve_image_path(src)
            if img_path.exists():
                st.image(str(img_path), caption=caption or alt or None, use_container_width=True)
            else:
                st.warning(f"Image not found: `{src}` (looked at `{img_path}`)")
        i += 1


# ═══════════════════════════════════════════════════════════
# Past blogs helpers
# ═══════════════════════════════════════════════════════════
def list_past_blogs() -> List[Path]:
    files = [p for p in Path(".").glob("*.md") if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def read_md_file(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def extract_title_from_md(md: str, fallback: str) -> str:
    for line in md.splitlines():
        if line.startswith("# "):
            return line[2:].strip() or fallback
    return fallback


# ═══════════════════════════════════════════════════════════
# Streamlit UI — Layout UNCHANGED
# ═══════════════════════════════════════════════════════════
st.set_page_config(page_title="LangGraph Blog Writer", layout="wide")
st.title("Blog Writing Agent")

with st.sidebar:
    st.header("Generate New Blog")
    topic  = st.text_area("Topic", height=120)
    as_of  = st.date_input("As-of date", value=date.today())
    run_btn = st.button("🚀 Generate Blog", type="primary")

    st.divider()
    st.subheader("Past blogs")

    past_files = list_past_blogs()
    if not past_files:
        st.caption("No saved blogs found (*.md in current folder).")
        selected_md_file = None
    else:
        options: List[str] = []
        file_by_label: Dict[str, Path] = {}
        for p in past_files[:50]:
            try:
                md_text = read_md_file(p)
                title   = extract_title_from_md(md_text, p.stem)
            except Exception:
                title = p.stem
            label = f"{title}  ·  {p.name}"
            options.append(label)
            file_by_label[label] = p

        selected_label   = st.radio("Select a blog to load", options=options,
                                    index=0, label_visibility="collapsed")
        selected_md_file = file_by_label.get(selected_label)

        if st.button("📂 Load selected blog"):
            if selected_md_file:
                md_text = read_md_file(selected_md_file)
                st.session_state["last_out"] = {
                    "plan": None, "evidence": [], "image_specs": [], "final": md_text,
                }
                st.session_state["topic_prefill"] = extract_title_from_md(md_text, selected_md_file.stem)

if "topic_prefill" in st.session_state and isinstance(st.session_state["topic_prefill"], str):
    pass

if "last_out" not in st.session_state:
    st.session_state["last_out"] = None

tab_plan, tab_evidence, tab_preview, tab_images, tab_logs = st.tabs(
    ["🧩 Plan", "🔎 Evidence", "📝 Markdown Preview", "🖼️ Images", "🧾 Logs"]
)

logs: List[str] = []
def log(msg: str): logs.append(msg)


if run_btn:
    if not topic.strip():
        st.warning("Please enter a topic.")
        st.stop()

    inputs: Dict[str, Any] = {
        "topic":                topic.strip(),
        "mode":                 "",
        "needs_research":       False,
        "queries":              [],
        "evidence":             [],
        "plan":                 None,
        "as_of":                as_of.isoformat(),
        "recency_days":         7,
        "sections":             [],
        "merged_md":            "",
        "md_with_placeholders": "",
        "image_specs":          [],
        "final":                "",
    }

    status        = st.status("Running graph…", expanded=True)
    progress_area = st.empty()
    current_state: Dict[str, Any] = {}
    last_node = None

    for kind, payload in try_stream(app, inputs):
        if kind in ("updates", "values"):
            node_name = None
            if isinstance(payload, dict) and len(payload) == 1 and isinstance(next(iter(payload.values())), dict):
                node_name = next(iter(payload.keys()))
            if node_name and node_name != last_node:
                status.write(f"➡️ Node: `{node_name}`")
                last_node = node_name

            current_state = extract_latest_state(current_state, payload)

            summary = {
                "mode":           current_state.get("mode"),
                "needs_research": current_state.get("needs_research"),
                "queries":        (current_state.get("queries") or [])[:5],
                "evidence_count": len(current_state.get("evidence") or []),
                "tasks":          len((current_state.get("plan") or {}).get("tasks", [])) if isinstance(current_state.get("plan"), dict) else None,
                "images":         len(current_state.get("image_specs") or []),
                "sections_done":  len(current_state.get("sections") or []),
            }
            progress_area.json(summary)
            log(f"[{kind}] {json.dumps(payload, default=str)[:1200]}")

        elif kind == "final":
            out = payload
            st.session_state["last_out"] = out
            status.update(label="✅ Done", state="complete", expanded=False)
            log("[final] received")


# ═══════════════════════════════════════════════════════════
# Render last result
# ═══════════════════════════════════════════════════════════
out = st.session_state.get("last_out")
if out:

    # ── Plan tab ────────────────────────────────────────────
    # ── BUG #9 FIX: Added subtasks display ─────────────────
    with tab_plan:
        st.subheader("Plan")
        plan_obj = out.get("plan")
        if not plan_obj:
            st.info("No plan found in output.")
        else:
            if hasattr(plan_obj, "model_dump"):
                plan_dict = plan_obj.model_dump()
            elif isinstance(plan_obj, dict):
                plan_dict = plan_obj
            else:
                plan_dict = json.loads(json.dumps(plan_obj, default=str))

            st.write("**Title:**", plan_dict.get("blog_title"))
            cols = st.columns(3)
            cols[0].write("**Audience:** " + str(plan_dict.get("audience")))
            cols[1].write("**Tone:** "     + str(plan_dict.get("tone")))
            cols[2].write("**Blog kind:** " + str(plan_dict.get("blog_kind", "")))

            tasks = plan_dict.get("tasks", [])
            if tasks:
                # Summary table (same as before)
                df = pd.DataFrame([{
                    "id":                t.get("id"),
                    "title":             t.get("title"),
                    "target_words":      t.get("target_words"),
                    "requires_research": t.get("requires_research"),
                    "requires_citations":t.get("requires_citations"),
                    "requires_code":     t.get("requires_code"),
                    "tags":              ", ".join(t.get("tags") or []),
                } for t in tasks]).sort_values("id")
                st.dataframe(df, use_container_width=True, hide_index=True)

                # ── BUG #9 FIX: Detailed task+subtask breakdown ──────────
                st.markdown("---")
                st.markdown("#### Task & Subtask Detail")
                for t in sorted(tasks, key=lambda x: x.get("id", 0)):
                    with st.expander(f"[{t.get('id')}] {t.get('title')} — {t.get('target_words')}w"):
                        st.markdown(f"**Goal:** {t.get('goal', '')}")

                        bullets  = t.get("bullets", [])
                        subtasks = t.get("subtasks", [])

                        if bullets:
                            st.markdown("**Bullets:**")
                            for b in bullets:
                                st.markdown(f"- {b}")

                        if subtasks:
                            st.markdown("**Subtasks (detailed):**")
                            for s in subtasks:
                                st.markdown(f"  &nbsp;&nbsp;↳ {s}")
                        else:
                            st.caption("No subtasks — run new blog to see subtask breakdown.")

                        flags = []
                        if t.get("requires_code"):     flags.append("💻 code")
                        if t.get("requires_research"): flags.append("🔍 research")
                        if t.get("requires_citations"):flags.append("📎 citations")
                        if flags:
                            st.markdown("**Flags:** " + " · ".join(flags))

    # ── Evidence tab (unchanged) ──────────────────────────────
    with tab_evidence:
        st.subheader("Evidence")
        evidence = out.get("evidence") or []
        if not evidence:
            st.info("No evidence returned (closed_book mode or no Tavily key).")
        else:
            rows = []
            for e in evidence:
                if hasattr(e, "model_dump"): e = e.model_dump()
                rows.append({"title": e.get("title"), "published_at": e.get("published_at"),
                             "source": e.get("source"), "url": e.get("url")})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Preview tab (unchanged) ───────────────────────────────
    with tab_preview:
        st.subheader("Markdown Preview")
        final_md = out.get("final") or ""
        if not final_md:
            st.warning("No final markdown found.")
        else:
            render_markdown_with_local_images(final_md)

            plan_obj   = out.get("plan")
            blog_title = (
                plan_obj.blog_title if hasattr(plan_obj, "blog_title")
                else plan_obj.get("blog_title", "blog") if isinstance(plan_obj, dict)
                else extract_title_from_md(final_md, "blog")
            )
            md_filename = f"{safe_slug(blog_title)}.md"
            st.download_button("⬇️ Download Markdown", data=final_md.encode("utf-8"),
                               file_name=md_filename, mime="text/markdown")
            bundle = bundle_zip(final_md, md_filename, Path("images"))
            st.download_button("📦 Download Bundle (MD + images)", data=bundle,
                               file_name=f"{safe_slug(blog_title)}_bundle.zip",
                               mime="application/zip")

    # ── Images tab (unchanged) ────────────────────────────────
    with tab_images:
        st.subheader("Images")
        specs      = out.get("image_specs") or []
        images_dir = Path("images")

        if not specs and not images_dir.exists():
            st.info("No images generated for this blog.")
        else:
            if specs:
                st.write("**Image plan:**")
                st.json(specs)
            if images_dir.exists():
                files = [p for p in images_dir.iterdir() if p.is_file()]
                if not files:
                    st.warning("images/ exists but is empty.")
                else:
                    for p in sorted(files):
                        st.image(str(p), caption=p.name, use_container_width=True)
                z = images_zip(images_dir)
                if z:
                    st.download_button("⬇️ Download Images (zip)", data=z,
                                       file_name="images.zip", mime="application/zip")

    # ── Logs tab (unchanged) ──────────────────────────────────
    with tab_logs:
        st.subheader("Logs")
        if "logs" not in st.session_state:
            st.session_state["logs"] = []
        if logs:
            st.session_state["logs"].extend(logs)
        st.text_area("Event log",
                     value="\n\n".join(st.session_state["logs"][-80:]),
                     height=520)

else:
    st.info("Enter a topic and click **Generate Blog**.")
