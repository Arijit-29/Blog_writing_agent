from __future__ import annotations
import base64,mimetypes
import json
import re
import zipfile
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional, List, Iterator, Tuple
import markdown as md_lib
import pandas as pd
import streamlit as st

# -----------------------------
# Import your compiled LangGraph app
# -----------------------------
from server import run as run_blog_workflow


# =========================================================
# Helpers (kept functionally identical to the backend I/O
# contract used previously)
# =========================================================
def safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"


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


def try_stream(graph_app, inputs: Dict[str, Any]) -> Iterator[Tuple[str, Any]]:
    """Stream graph progress if available; else invoke."""
    try:
        for step in graph_app.stream(inputs, stream_mode="updates"):
            yield ("updates", step)
        out = graph_app.invoke(inputs)
        yield ("final", out)
        return
    except Exception:
        pass

    try:
        for step in graph_app.stream(inputs, stream_mode="values"):
            yield ("values", step)
        out = graph_app.invoke(inputs)
        yield ("final", out)
        return
    except Exception:
        pass

    out = graph_app.invoke(inputs)
    yield ("final", out)


def extract_latest_state(current_state: Dict[str, Any], step_payload: Any) -> Dict[str, Any]:
    if isinstance(step_payload, dict):
        if len(step_payload) == 1 and isinstance(next(iter(step_payload.values())), dict):
            inner = next(iter(step_payload.values()))
            current_state.update(inner)
        else:
            current_state.update(step_payload)
    return current_state

# -----------------------------
# Markdown renderer that supports local images
# -----------------------------
_MD_IMG_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)")
_CAPTION_LINE_RE = re.compile(r"^\*(?P<cap>.+)\*$")


def _resolve_image_path(src: str) -> Path:
    src = src.strip().lstrip("./")
    return Path(src).resolve()


def render_markdown_with_local_images(md: str):
    matches = list(_MD_IMG_RE.finditer(md))
    if not matches:
        st.markdown(md, unsafe_allow_html=False)
        return

    parts: List[Tuple[str, str]] = []
    last = 0
    for m in matches:
        before = md[last : m.start()]
        if before:
            parts.append(("md", before))
        alt = (m.group("alt") or "").strip()
        src = (m.group("src") or "").strip()
        parts.append(("img", f"{alt}|||{src}"))
        last = m.end()

    tail = md[last:]
    if tail:
        parts.append(("md", tail))

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
                    rest = "\n".join(nxt.splitlines()[1:])
                    parts[i + 1] = ("md", rest)

        if src.startswith("http://") or src.startswith("https://"):
            st.image(src, caption=caption or (alt or None), use_container_width=True)
        else:
            img_path = _resolve_image_path(src)
            if img_path.exists():
                st.image(str(img_path), caption=caption or (alt or None), use_container_width=True)
            else:
                st.warning(f"Image not found: `{src}` (looked for `{img_path}`)")

        i += 1

def _image_to_data_uri(src: str) -> Optional[str]:
    """Read a local image file and return it as a base64 data: URI, or None if missing."""
    img_path = _resolve_image_path(src)
    if not img_path.exists() or not img_path.is_file():
        return None
    mime, _ = mimetypes.guess_type(str(img_path))
    mime = mime or "image/png"
    data = base64.b64encode(img_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"

def markdown_to_standalone_html(md_text: str, title: str) -> str:
    """
    Convert the blog markdown into a single, self-contained HTML file:
    - local images are embedded as base64 data URIs (so the file works
      even without an accompanying images/ folder)
    - remote images and all links (citations, sources, etc.) stay live
    - lightly styled for pleasant reading in any browser
    """
 
    def _embed(m: "re.Match") -> str:
        alt = m.group("alt") or ""
        src = (m.group("src") or "").strip()
        if src.startswith("http://") or src.startswith("https://"):
            return m.group(0)
        data_uri = _image_to_data_uri(src)
        if data_uri:
            return f"![{alt}]({data_uri})"
        return m.group(0)
 
    md_with_embedded_images = _MD_IMG_RE.sub(_embed, md_text)
 
    body_html = md_lib.markdown(
        md_with_embedded_images,
        extensions=["extra", "sane_lists", "toc", "nl2br"],
    )
 
    safe_title = title or "Blog"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{safe_title}</title>
<style>
  :root {{
    color-scheme: light;
  }}
  body {{
    margin: 0;
    padding: 0;
    background: #f6f5fb;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    color: #1f2430;
  }}
  .wrap {{
    max-width: 780px;
    margin: 0 auto;
    padding: 3rem 1.5rem 5rem 1.5rem;
  }}
  article {{
    background: #ffffff;
    border-radius: 18px;
    padding: 2.4rem 2.6rem;
    box-shadow: 0 10px 30px rgba(30, 20, 60, 0.06);
    line-height: 1.7;
    font-size: 1.05rem;
  }}
  h1, h2, h3, h4 {{
    line-height: 1.3;
    font-weight: 800;
  }}
  h1 {{
    font-size: 2rem;
    background: linear-gradient(90deg, #6C63FF, #FF6584);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
  }}
  h2 {{ font-size: 1.4rem; margin-top: 2.2rem; }}
  h3 {{ font-size: 1.15rem; margin-top: 1.6rem; }}
  a {{ color: #6C63FF; text-decoration: none; border-bottom: 1px solid rgba(108,99,255,0.35); }}
  a:hover {{ border-bottom-color: #6C63FF; }}
  img {{
    max-width: 100%;
    height: auto;
    border-radius: 12px;
    display: block;
    margin: 1.4rem auto;
    box-shadow: 0 6px 18px rgba(0,0,0,0.08);
  }}
  em {{ color: #6b7280; font-size: 0.92rem; display: block; text-align: center; margin-top: -0.9rem; }}
  blockquote {{
    border-left: 4px solid #6C63FF;
    margin: 1.2rem 0;
    padding: 0.4rem 1rem;
    color: #444;
    background: #f9f8ff;
    border-radius: 0 8px 8px 0;
  }}
  code {{
    background: #f1f0f8;
    padding: 0.15rem 0.4rem;
    border-radius: 6px;
    font-size: 0.92em;
  }}
  pre {{
    background: #1f2430;
    color: #e6e6f0;
    padding: 1rem 1.2rem;
    border-radius: 10px;
    overflow-x: auto;
  }}
  pre code {{ background: none; padding: 0; color: inherit; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1.2rem 0; }}
  th, td {{ border: 1px solid #e5e7eb; padding: 0.5rem 0.75rem; text-align: left; }}
  th {{ background: #f6f5fb; }}
  hr {{ border: none; border-top: 1px solid #e5e7eb; margin: 2rem 0; }}
  .footer-note {{
    text-align: center;
    color: #9ca3af;
    font-size: 0.82rem;
    margin-top: 2rem;
  }}
</style>
</head>
<body>
  <div class="wrap">
    <article>
      {body_html}
    </article>
    <p class="footer-note">Generated by the Blog Writing Agent</p>
  </div>
</body>
</html>
"""

def html_to_pdf_bytes(html_string: str) -> Optional[bytes]:
    """
    Render a self-contained HTML string to PDF bytes using WeasyPrint.
    Links stay clickable (real PDF link annotations) and embedded
    base64 images render as images, not code — it's a genuine
    "print this page" style export, not a screenshot.
    Returns None if WeasyPrint (or its system libs) aren't available,
    so the app can degrade gracefully instead of crashing.
    """
    try:
        from weasyprint import HTML  # imported lazily: heavy + needs system libs
    except Exception:
        return None
    try:
        return HTML(string=html_string).write_pdf()
    except Exception:
        return None

def extract_used_local_images(md_text: str) -> List[Tuple[Path, str]]:
    """
    Return the local images actually referenced by ![...](...) in this
    blog's markdown, in the order they appear, deduplicated, skipping
    remote (http/https) images and any path that isn't a real file on
    disk. Each entry is (resolved_path, alt_text).
    """
    seen: set = set()
    used: List[Tuple[Path, str]] = []
    for m in _MD_IMG_RE.finditer(md_text or ""):
        src = (m.group("src") or "").strip()
        alt = (m.group("alt") or "").strip()
        if not src or src.startswith("http://") or src.startswith("https://"):
            continue
        img_path = _resolve_image_path(src)
        if img_path in seen:
            continue
        if img_path.exists() and img_path.is_file():
            seen.add(img_path)
            used.append((img_path, alt))
    return used
 
def images_zip_subset(paths: List[Path]) -> Optional[bytes]:
    """Zip only the given image files (not the whole images/ folder)."""
    files = [p for p in paths if p.exists() and p.is_file()]
    if not files:
        return None
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, arcname=p.name)
    return buf.getvalue()
# =========================================================
# Page setup + visual theme
# =========================================================
st.set_page_config(
    page_title="Blog Writing Agent",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      #MainMenu, header, footer {visibility: hidden;}
      .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 980px;}

      .bwa-hero {
        text-align: center;
        padding: 1.6rem 1rem 1.2rem 1rem;
      }
      .bwa-hero h1 {
        font-size: 2.1rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        background: linear-gradient(90deg, #6C63FF, #FF6584);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
      }
      .bwa-hero p {
        color: #8a8a99;
        font-size: 1.02rem;
        margin-top: 0;
      }

      .bwa-badge {
        display:inline-block;
        padding: 0.15rem 0.6rem;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 600;
        margin-right: 0.35rem;
        margin-bottom: 0.25rem;
      }
      .badge-blue   {background:#EEF2FF; color:#4338CA;}
      .badge-green  {background:#ECFDF5; color:#047857;}
      .badge-amber  {background:#FFFBEB; color:#B45309;}
      .badge-pink   {background:#FDF2F8; color:#BE185D;}
      .badge-gray   {background:#F3F4F6; color:#374151;}

      .bwa-card {
        border: 1px solid rgba(120,120,140,0.18);
        border-radius: 14px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
        background: rgba(120,120,140,0.03);
      }

      .bwa-lock {
        text-align:center;
        padding: 0.9rem 1rem;
        border-radius: 12px;
        background: #ECFDF5;
        color: #047857;
        font-weight: 600;
        border: 1px solid #A7F3D0;
      }

      div[data-testid="stChatInput"] textarea {
        border-radius: 14px !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="bwa-hero">
      <h1>✍️ Blog Writing Agent</h1>
      <p>Tell me what you'd like a blog written about — I'll research, plan, draft and illustrate it.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# Session state
# =========================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "generated" not in st.session_state:
    st.session_state.generated = False
if "last_out" not in st.session_state:
    st.session_state.last_out = None
if "logs" not in st.session_state:
    st.session_state.logs = []
if "as_of_date" not in st.session_state:
    st.session_state.as_of_date = date.today()
# =========================================================
# Chat history
# =========================================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=("🧑" if msg["role"] == "user" else "✍️")):
        st.markdown(msg["content"])


# =========================================================
# Input area — only available before the first blog is made
# =========================================================
topic: Optional[str] = None

if not st.session_state.generated:
    with st.expander("⚙️ Advanced settings", expanded=False):
        st.session_state.as_of_date = st.date_input(
            "As-of date (for freshness of research)",
            value=st.session_state.as_of_date,
        )
    topic = st.chat_input("Describe the blog you want, e.g. 'The rise of agentic AI in 2026'…")
else:
    st.markdown(
        """
        <div class="bwa-lock">
          ✅ Your blog is ready below. This chat is now closed — refresh the page to start a new one.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# Run the graph when a topic is submitted
# =========================================================
if topic and topic.strip():
    st.session_state.messages.append({"role": "user", "content": topic})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(topic)

    inputs: Dict[str, Any] = {
        "topic": topic.strip(),
        "mode": "",
        "needs_research": False,
        "queries": [],
        "evidence": [],
        "plan": None,
        "as_of": st.session_state.as_of_date.isoformat(),
        "recency_days": 7,
        "sections": [],
        "merged_md": "",
        "md_with_placeholders": "",
        "image_specs": [],
        "final": "",
    }

    with st.chat_message("assistant", avatar="✍️"):
        status = st.status("Working on your blog…", expanded=True)
        progress_area = st.empty()

        run_logs: List[str] = []
        try:
            out = run_blog_workflow(topic=topic.strip(), batch_size=2, delay=15.0)
            st.session_state.last_out = out
            status.update(label="✅ Blog complete", state="complete", expanded=False)
            run_logs.append("[workflow] completed")
            progress_area.write("Generating the final markdown and images…")
        except Exception as exc:
            status.update(label="❌ Blog generation failed", state="error", expanded=False)
            st.error(f"Blog generation failed: {exc}")
            run_logs.append(f"[error] {exc}")
            out = None

        st.session_state.logs.extend(run_logs)

        plan_obj = (st.session_state.last_out or {}).get("plan")
        if hasattr(plan_obj, "blog_title"):
            done_title = plan_obj.blog_title
        elif isinstance(plan_obj, dict):
            done_title = plan_obj.get("blog_title", topic.strip())
        else:
            done_title = topic.strip()

        done_msg = f"Done! 🎉 I've written **{done_title}** — scroll down for the plan, evidence, preview and images."
        st.markdown(done_msg)
        st.session_state.messages.append({"role": "assistant", "content": done_msg})

    st.session_state.generated = True
    st.rerun()


# =========================================================
# Results — rendered below the chat once available
# =========================================================
out = st.session_state.last_out
if out:
    st.divider()

    tab_preview, tab_plan, tab_evidence, tab_images, tab_logs = st.tabs(
        ["📝 Preview", "🧩 Plan", "🔎 Evidence", "🖼️ Images", "🧾 Logs"]
    )

    # --- Preview tab ---
    with tab_preview:
        final_md = out.get("final") or ""
        if not final_md:
            st.warning("No final markdown found.")
        else:
            with st.container(border=True):
                render_markdown_with_local_images(final_md)

            plan_obj = out.get("plan")
            if hasattr(plan_obj, "blog_title"):
                blog_title = plan_obj.blog_title
            elif isinstance(plan_obj, dict):
                blog_title = plan_obj.get("blog_title", "blog")
            else:
                blog_title = "blog"

            md_filename = f"{safe_slug(blog_title)}.md"
            html_export = markdown_to_standalone_html(final_md, blog_title)
            pdf_bytes = html_to_pdf_bytes(html_export)
            if pdf_bytes:
                    st.download_button(
                        "📄 Download PDF",
                        data=pdf_bytes,
                        file_name=f"{safe_slug(blog_title)}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                        help="Same rendered page as a PDF — images embedded, links stay clickable.",
                    )
            else:
                    st.button(
                        "📄 PDF unavailable",
                        disabled=True,
                        use_container_width=True,
                        help="Install WeasyPrint (pip install weasyprint) plus its system libraries to enable PDF export.",
                    )


    # --- Plan tab ---
    with tab_plan:
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

            st.markdown(f"### {plan_dict.get('blog_title', '')}")
            st.markdown(
                f"""
                <span class="bwa-badge badge-blue">Audience: {plan_dict.get('audience', '—')}</span>
                <span class="bwa-badge badge-pink">Tone: {plan_dict.get('tone', '—')}</span>
                <span class="bwa-badge badge-amber">Kind: {plan_dict.get('blog_kind', '—')}</span>
                """,
                unsafe_allow_html=True,
            )

            tasks = plan_dict.get("tasks", [])
            if tasks:
                df = pd.DataFrame(
                    [
                        {
                            "id": t.get("id"),
                            "title": t.get("title"),
                            "target_words": t.get("target_words"),
                            "requires_research": t.get("requires_research"),
                            "requires_citations": t.get("requires_citations"),
                            "requires_code": t.get("requires_code"),
                            "tags": ", ".join(t.get("tags") or []),
                        }
                        for t in tasks
                    ]
                ).sort_values("id")
                st.dataframe(df, use_container_width=True, hide_index=True)
                with st.expander("Task details (raw)"):
                    st.json(tasks)

    # --- Evidence tab ---
    with tab_evidence:
        evidence = out.get("evidence") or []
        if not evidence:
            st.info("No evidence returned (maybe closed-book mode or no search results).")
        else:
            rows = []
            for e in evidence:
                if hasattr(e, "model_dump"):
                    e = e.model_dump()
                rows.append(
                    {
                        "title": e.get("title"),
                        "published_at": e.get("published_at"),
                        "source": e.get("source"),
                        "url": e.get("url"),
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # --- Images tab ---
    with tab_images:
        specs = out.get("image_specs") or []
        final_md = out.get("final") or ""
        used_images = extract_used_local_images(final_md)
 
        if not specs and not used_images:
            st.info("No images used in this blog.")
        else:
            if specs:
                with st.expander("Image plan"):
                    st.json(specs)
 
            if not used_images:
                st.warning("No images from images/ are referenced in this blog's markdown.")
            else:
                cols = st.columns(2)
                for idx, (p, alt) in enumerate(used_images):
                    with cols[idx % 2]:
                        st.image(str(p), caption=alt or p.name, use_container_width=True)
 
                z = images_zip_subset([p for p, _ in used_images])  # zips only the used ones
                if z:
                    st.download_button("⬇️ Download Images (zip)", data=z, file_name="images.zip", mime="application/zip")

    # --- Logs tab ---
    with tab_logs:
        st.text_area(
            "Event log",
            value="\n\n".join(st.session_state.logs[-80:]),
            height=420,
        )
else:
    if not st.session_state.generated:
        st.caption("💬 Type your topic in the chat box below to get started.")