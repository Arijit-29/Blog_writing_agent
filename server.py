from __future__ import annotations
import operator
import re
import time
import os
import io
import requests
import base64
from typing import List, Annotated, TypedDict, Literal, Optional
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from pathlib import Path
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain_core.rate_limiters import InMemoryRateLimiter

load_dotenv()
os.environ["LANGSMITH_PROJECT"] = "Blog_Writing_Agent"
rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.5,  # tune to your Groq tier
    check_every_n_seconds=0.1,
    max_bucket_size=2,
)


# 1)Schemas
class Task(BaseModel):
    id: int
    title: str
    goal: str = Field(
        ...,
        description="One sentence describing what the reader should be able to do/understand after this section.",
    )
    bullets: List[str] = Field(
        ...,
        min_length=3,
        max_length=5,
        description="3–5 concrete, non-overlapping subpoints to cover in this section.",
    )
    target_words: int = Field(
        ...,
        description="Target word count for this section (120–450).",
    )
    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False


class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal[
        "explainer", "tutorial", "news_roundup", "comparison", "system_design"
    ] = "explainer"
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]


class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None  # keep if Tavily provides; DO NOT rely on it
    snippet: Optional[str] = None
    source: Optional[str] = None


class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    queries: List[str] = Field(default_factory=list)


class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)


class ImageSpec(BaseModel):
    after_section_id: int = Field(
        ..., description="The section id (Task.id) this image should appear after."
    )
    placeholder: str = Field(..., description="e.g. [[IMAGE_1]]")
    filename: str = Field(..., description="Save under images/, e.g. qkv_flow.png")
    alt: str
    caption: str
    prompt: str = Field(..., description="Prompt to send to the image model.")
    size: Literal["1024x1024", "1024x1536", "1536x1024"] = "1024x1024"
    quality: Literal["low", "medium", "high"] = "medium"


class GlobalImagePlan(BaseModel):
    images: List[ImageSpec] = Field(default_factory=list)


Task.model_rebuild()
Plan.model_rebuild()


class State(TypedDict):
    topic: str
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]
    sections: Annotated[List[tuple[int, str]], operator.add]
    merged_md: str
    md_with_placeholders: str
    image_specs: List[dict]
    final: str


# 2)LLM
llm = ChatGroq(model="openai/gpt-oss-120b", rate_limiter=rate_limiter, max_retries=4)
worker_llm = ChatGroq(
    model="llama-3.3-70b-versatile", rate_limiter=rate_limiter, max_retries=4
)

# 3)Decide Router
ROUTER_SYSTEM = """You are a routing module for a technical blog planner.

Decide whether web research is needed BEFORE planning.

Modes:
- closed_book (needs_research=false):
  Evergreen topics where correctness does not depend on recent facts (concepts, fundamentals).
- hybrid (needs_research=true):
  Mostly evergreen but needs up-to-date examples/tools/models to be useful.
- open_book (needs_research=true):
  Mostly volatile: weekly roundups, "this week", "latest", rankings, pricing, policy/regulation.

If needs_research=true:
- Output 3–10 high-signal queries.
- Queries should be scoped and specific (avoid generic queries like just "AI" or "LLM").
- If user asked for "last week/this week/latest", reflect that constraint IN THE QUERIES.
"""


def router_node(state: State) -> dict:
    topic = state["topic"]
    decider = llm.with_structured_output(RouterDecision)
    decision = decider.invoke(
        [
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(content=f"Topic: {topic}"),
        ]
    )

    return {
        "needs_research": decision.needs_research,  # type: ignore
        "mode": decision.mode,  # type: ignore
        "queries": decision.queries,  # type: ignore
    }


def route_next(state: State) -> str:
    return "research" if state["needs_research"] else "orchestrator"


# 4)Reasearch Tavily
def _tavily_search(query: str, max_results: int = 5) -> List[dict]:
    tool = TavilySearch(max_results=max_results)
    try:
        results = tool.invoke({"query": query})
    except Exception as exc:  # pragma: no cover - defensive guard
        print(f"Tavily search failed for query {query!r}: {exc}")
        return []

    if isinstance(results, dict):
        if isinstance(results.get("results"), list):
            items = results["results"]
        elif isinstance(results.get("data"), list):
            items = results["data"]
        else:
            items = [results]
    elif isinstance(results, list):
        items = results
    else:
        items = [results] if results else []

    normalized: List[dict] = []
    for item in items:
        if isinstance(item, dict):
            normalized.append(
                {
                    "title": item.get("title") or item.get("name") or "",
                    "url": item.get("url") or item.get("link") or "",
                    "snippet": item.get("content")
                    or item.get("snippet")
                    or item.get("description")
                    or "",
                    "published_at": item.get("published_date")
                    or item.get("published_at")
                    or item.get("date"),
                    "source": item.get("source")
                    or item.get("provider")
                    or item.get("site"),
                }
            )
        elif isinstance(item, str) and item.strip():
            normalized.append(
                {
                    "title": item.strip(),
                    "url": "",
                    "snippet": "",
                    "published_at": None,
                    "source": None,
                }
            )

    return normalized


RESEARCH_SYSTEM = """You are a research synthesizer for technical writing.

Given compact web search results, produce a deduplicated list of EvidenceItem objects.

Rules:
- Only include items with a non-empty url.
- Prefer relevant + authoritative sources (company blogs, docs, reputable outlets).
- If a published date is explicitly present in the result payload, keep it as YYYY-MM-DD.
  If missing or unclear, set published_at=null. Do NOT guess.
- Keep snippets short.
- Deduplicate by URL.
"""


def _compact_results(raw_results: List[dict], limit: int = 12) -> List[dict]:
    compacted: List[dict] = []
    for item in raw_results[:limit]:
        if not isinstance(item, dict):
            continue

        title = str(item.get("title") or item.get("name") or "").strip()
        url = str(item.get("url") or item.get("link") or "").strip()
        snippet = str(
            item.get("snippet") or item.get("content") or item.get("description") or ""
        ).strip()
        if not title and not url and not snippet:
            continue

        compacted.append(
            {
                "title": title[:120],
                "url": url,
                "snippet": " ".join(snippet.split())[:220],
                "published_at": item.get("published_date")
                or item.get("published_at")
                or item.get("date"),
                "source": item.get("source")
                or item.get("provider")
                or item.get("site"),
            }
        )

    return compacted


def research_node(state: State) -> dict:
    queries = [q.strip() for q in (state.get("queries", []) or []) if q and q.strip()]
    queries = queries[:4]
    max_results = 2

    raw_results: List[dict] = []

    for q in queries:
        raw_results.extend(_tavily_search(q, max_results=max_results))

    if not raw_results:
        return {"evidence": []}

    compact_results = _compact_results(raw_results, limit=12)
    if not compact_results:
        return {"evidence": []}

    evidence_items = [
        EvidenceItem(
            title=item.get("title") or "Untitled result",
            url=item.get("url") or "",
            published_at=item.get("published_at"),
            snippet=item.get("snippet"),
            source=item.get("source"),
        )
        for item in compact_results
        if item.get("url")
    ]

    # Deduplicate by URL before returning.
    dedup = {}
    for evidence in evidence_items:
        if getattr(evidence, "url", ""):
            dedup[evidence.url] = evidence

    return {"evidence": list(dedup.values())}


# 5)Orchestrator
ORCH_SYSTEM = """You are a senior technical writer and developer advocate.
Your job is to produce a highly actionable outline for a technical blog post.

Hard requirements:
- Create 5–9 sections (tasks) suitable for the topic and audience.
- Each task must include:
  1) goal (1 sentence)
  2) 3–6 bullets that are concrete, specific, and non-overlapping
  3) target word count (120–550)

Quality bar:
- Assume the reader is a developer; use correct terminology.
- Bullets must be actionable: build/compare/measure/verify/debug.
- Ensure the overall plan includes at least 2 of these somewhere:
  * minimal code sketch / MWE (set requires_code=True for that section)
  * edge cases / failure modes
  * performance/cost considerations
  * security/privacy considerations (if relevant)
  * debugging/observability tips

Grounding rules:
- Mode closed_book: keep it evergreen; do not depend on evidence.
- Mode hybrid:
  - Use evidence for up-to-date examples (models/tools/releases) in bullets.
  - Mark sections using fresh info as requires_research=True and requires_citations=True.
- Mode open_book:
  - Set blog_kind = "news_roundup".
  - Every section is about summarizing events + implications.
  - DO NOT include tutorial/how-to sections unless user explicitly asked for that.
  - If evidence is empty or insufficient, create a plan that transparently says "insufficient sources"
    and includes only what can be supported.

Output must strictly match the Plan schema.
"""


def orchestrator_node(state: State) -> dict:
    planner = llm.with_structured_output(Plan)

    evidence = state.get("evidence", [])
    mode = state.get("mode", "closed_book")

    plan = planner.invoke(
        [
            SystemMessage(content=ORCH_SYSTEM),
            HumanMessage(
                content=(
                    f"Topic: {state['topic']}\n"
                    f"Mode: {mode}\n\n"
                    f"Evidence (ONLY use for fresh claims; may be empty):\n"
                    f"{[e.model_dump() for e in evidence][:16]}"
                )
            ),
        ]
    )
    return {"plan": plan}


# 6)Workers
WORKER_SYSTEM = """You are a senior technical writer and developer advocate.
Write ONE section of a technical blog post in Markdown.

Hard constraints:
- Follow the provided Goal and cover ALL Bullets in order (do not skip or merge bullets).
- Stay close to Target words (±15%).
- Output ONLY the section content in Markdown (no blog title H1, no extra commentary).
- Start with a '## <Section Title>' heading.

Scope guard:
- If blog_kind == "news_roundup": do NOT turn this into a tutorial/how-to guide.
  Do NOT teach web scraping, RSS, automation, or "how to fetch news" unless bullets explicitly ask for it.
  Focus on summarizing events and implications.

Grounding policy:
- If mode == open_book:
  - Do NOT introduce any specific event/company/model/funding/policy claim unless it is supported by provided Evidence URLs.
  - For each event claim, attach a source as a Markdown link: ([Source](URL)).
  - Only use URLs provided in Evidence. If not supported, write: "Not found in provided sources."
- If requires_citations == true:
  - For outside-world claims, cite Evidence URLs the same way.
- Evergreen reasoning is OK without citations unless requires_citations is true.

Code:
- If requires_code == true, include at least one minimal, correct code snippet relevant to the bullets.

Style:
- Short paragraphs, bullets where helpful, code fences for code.
- Avoid fluff/marketing. Be precise and implementation-oriented.
"""


def worker_node(payload: dict) -> dict:
    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])
    evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]
    topic = payload["topic"]
    mode = payload.get("mode", "closed_book")
    bullets_text = "\n- " + "\n- ".join(task.bullets)
    needs_evidence = (
        task.requires_research or task.requires_citations or mode == "open_book"
    )
    evidence_text = ""
    if evidence and needs_evidence:
        evidence_text = "\n".join(
            f"- {e.title} | {e.url} | {e.published_at or 'date:unknown'}".strip()
            for e in evidence[:8]
        )
    section_md = worker_llm.invoke(
        [
            SystemMessage(content=WORKER_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog title: {plan.blog_title}\n"
                    f"Audience: {plan.audience}\n"
                    f"Tone: {plan.tone}\n"
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Constraints: {plan.constraints}\n"
                    f"Topic: {topic}\n"
                    f"Mode: {mode}\n\n"
                    f"Section title: {task.title}\n"
                    f"Goal: {task.goal}\n"
                    f"Target words: {task.target_words}\n"
                    f"Tags: {task.tags}\n"
                    f"requires_research: {task.requires_research}\n"
                    f"requires_citations: {task.requires_citations}\n"
                    f"requires_code: {task.requires_code}\n"
                    f"Bullets:{bullets_text}\n\n"
                    f"Evidence (ONLY use these URLs when citing):\n{evidence_text}\n"
                )
            ),
        ]
    ).content.strip()  # type: ignore

    return {"sections": [(task.id, section_md)]}


# 7)Reducers
def merge_content(plan: Plan, sections: List[tuple[int, str]]) -> dict:
    if plan is None:
        raise ValueError("merge_content called without plan.")
    ordered_sections = [md for _, md in sorted(sections, key=lambda x: x[0])]
    body = "\n\n".join(ordered_sections).strip()
    merged_md = f"# {plan.blog_title}\n\n{body}\n"
    return {"merged_md": merged_md}


def _safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"


DECIDE_IMAGES_SYSTEM = """You are an expert technical editor and AI image prompt engineer.
Decide if diagrams are needed for THIS blog to materially improve understanding.

Rules:
- Max 3 images total.
- Insert placeholders exactly: [[IMAGE_1]], [[IMAGE_2]], [[IMAGE_3]].
- If no images needed: return images=[].

FLUX.1-schnell Prompting Rules for Diagrams:
- Enforce a technical vector aesthetic. Always start diagram prompts with: "Minimalist flat vector architecture diagram, clean lines, dark mode technical aesthetic, schematic layout, high contrast..."
- DO NOT use Stable Diffusion syntax, prompt weights, or parentheses for emphasis (e.g., no `(word)++`).
- Be incredibly precise, detailed, and direct. Avoid vague terms.
- Explicitly define the tone, style, and color palette (e.g., deep blues and vibrant neon accents).
- Organize the layout in a hierarchical manner, clearly stating what is in the foreground, middle ground, and background.

Return strictly GlobalImagePlan.
"""


def decide_images(topic: str, plan: Plan) -> GlobalImagePlan:
    planner = llm.with_structured_output(GlobalImagePlan)
    section_list = "\n".join(
        f"- id={t.id}: {t.title} (goal: {t.goal})" for t in plan.tasks
    )
    image_plan = planner.invoke(
        [
            SystemMessage(content=DECIDE_IMAGES_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Topic: {topic}\n\n"
                    f"Sections:\n{section_list}\n\n"
                    "Propose up to 3 images, each attached to one section id above."
                )
            ),
        ]
    )
    return image_plan  # type: ignore


CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")
CLOUDFLARE_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN")

MODEL = "@cf/black-forest-labs/flux-1-schnell"

API_URL = (
    f"https://api.cloudflare.com/client/v4/accounts/"
    f"{CLOUDFLARE_ACCOUNT_ID}/ai/run/{MODEL}"
)

HEADERS = {
    "Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}",
    "Content-Type": "application/json",
}


def _cf_generate_image_bytes(prompt: str) -> bytes:
    payload = {
        "prompt": prompt,
        "width": 1024,
        "height": 1024,
        "steps": 4,
    }

    response = requests.post(
        API_URL,
        headers=HEADERS,
        json=payload,
        timeout=300,
    )

    response.raise_for_status()

    data = response.json()

    if not data.get("success", False):
        raise RuntimeError(data.get("errors", data))

    image_b64 = data["result"]["image"]

    return base64.b64decode(image_b64)


def insert_image_placeholders(
    sections: List[tuple[int, str]], images: List[ImageSpec]
) -> List[tuple[int, str]]:
    """Deterministically append each image's placeholder after its target section's
    markdown, instead of relying on the LLM to insert them into a giant echoed doc."""
    by_section: dict[int, List[str]] = {}
    for img in images:
        by_section.setdefault(img.after_section_id, []).append(img.placeholder)
    updated: List[tuple[int, str]] = []
    for sid, md in sections:
        placeholders = by_section.get(sid, [])
        if placeholders:
            md = md.rstrip() + "\n\n" + "\n\n".join(placeholders)
        updated.append((sid, md))
    return updated


def generate_and_place_images(plan: Plan, md: str, image_specs: List[dict]) -> dict:
    # If no images requested, just write merged markdown as-is.
    if not image_specs:
        filename = f"{_safe_slug(plan.blog_title)}.md"
        output_path = Path.cwd() / filename
        output_path.write_text(md, encoding="utf-8")
        print(f"Saved to: {output_path.resolve()}")
        print(f"Characters written: {len(md)}")
        return {"final": md}

    images_dir = Path("images")
    images_dir.mkdir(exist_ok=True)

    for spec in image_specs:
        placeholder = spec["placeholder"]
        filename = spec["filename"]
        out_path = images_dir / filename

        if not out_path.exists():
            try:
                img_bytes = _cf_generate_image_bytes(spec["prompt"])
                out_path.write_bytes(img_bytes)
            except Exception as e:
                # graceful fallback: keep doc usable even if image gen fails
                prompt_block = (
                    f"> **[IMAGE GENERATION FAILED]** {spec.get('caption','')}\n>\n"
                    f"> **Alt:** {spec.get('alt','')}\n>\n"
                    f"> **Prompt:** {spec.get('prompt','')}\n>\n"
                    f"> **Error:** {e}\n"
                )
                md = md.replace(placeholder, prompt_block)
                continue

        img_md = f"![{spec['alt']}](images/{filename})\n*{spec['caption']}*"
        md = md.replace(placeholder, img_md)

    filename = f"{_safe_slug(plan.blog_title)}.md"
    output_path = Path.cwd() / filename
    output_path.write_text(md, encoding="utf-8")
    print(f"Saved to: {output_path.resolve()}")
    print(f"Characters written: {len(md)}")
    return {"final": md}


# 8) Final batched workers
def run_workers_batched(plan, topic, mode, evidence, batch_size=2, delay=15.0):
    sections = []
    tasks = plan.tasks
    evidence_dump = [e.model_dump() for e in evidence]
    plan_dump = plan.model_dump()

    for i in range(0, len(tasks), batch_size):
        batch = tasks[i : i + batch_size]
        for task in batch:
            payload = {
                "task": task.model_dump(),
                "topic": topic,
                "mode": mode,
                "plan": plan_dump,
                "evidence": evidence_dump,
            }
            result = worker_node(payload)
            sections.extend(result["sections"])
        if (i + batch_size) < len(tasks):
            time.sleep(delay)

    return sections


# 9) Build Graph
graph = StateGraph(State)
graph.add_node("router", router_node)
graph.add_node("research", research_node)
graph.add_node("orchestrator", orchestrator_node)
graph.add_edge(START, "router")
graph.add_conditional_edges(
    "router", route_next, {"research": "research", "orchestrator": "orchestrator"}
)
graph.add_edge("research", "orchestrator")
graph.add_edge("orchestrator", END)
app = graph.compile()


# 10)Run
def run(topic: str, batch_size: int = 2, delay: float = 15.0):
    plan_state = app.invoke(
        {
            "topic": topic,
            "mode": "",
            "needs_research": False,
            "queries": [],
            "evidence": [],
            "plan": None,
            "sections": [],
            "final": "",
        }  # type: ignore
    )

    plan: Plan = plan_state["plan"]
    mode = plan_state["mode"]
    evidence = plan_state.get("evidence", [])

    sections = run_workers_batched(
        plan=plan,
        topic=topic,
        mode=mode,
        evidence=evidence,
        batch_size=batch_size,
        delay=delay,
    )
    image_plan = decide_images(topic, plan)
    sections_with_placeholders = insert_image_placeholders(sections, image_plan.images)
    merged = merge_content(plan, sections_with_placeholders)
    result = generate_and_place_images(
        plan, merged["merged_md"], [img.model_dump() for img in image_plan.images]
    )
    return {
        **plan_state,
        "sections": sections,
        "merged_md": merged["merged_md"],
        "md_with_placeholders": merged["merged_md"],
        "image_specs": [img.model_dump() for img in image_plan.images],
        "final": result["final"],
    }
