# 🚀Agentic AI Blog Writing Agent with Research + Image Generation

An end-to-end **Agentic AI Blog Writing System** built using **LangGraph, LangChain, OpenAI, Tavily, and Gemini Image Generation**.

This project automatically:

* Routes a topic (research or not)
* Creates a structured blog plan
* Splits into parallel writing tasks
* Merges sections
* Decides if diagrams are needed
* Generates images using Gemini
* Produces a final Markdown blog file with embedded images

---

# 🧠 System Architecture

This system is built as a **multi-agent workflow using LangGraph**, where each node has a specialized responsibility.

![Blog UI Screenshot](https://github.com/user-attachments/assets/01955b10-3129-4c80-8eb8-09efde18d53d)

---

# 🏗️ Core Technologies

## 🔹 Backend

* **LangGraph** → Workflow orchestration (multi-node graph execution)
* **LangChain** → LLM abstraction & structured outputs
* **OpenAI (gpt-4.1-mini)** → Planning + Section writing
* **Tavily API** → Web search for research mode
* **Google Gemini (Image Model)** → Diagram generation
* **Pydantic** → Strict schema validation
* **Python 3.10+**

## 🔹 Frontend (UI)

* Topic input field
* Displays final generated Markdown
* Automatically embeds generated diagrams
* Image files saved under `/images`
* Final blog exported as `.md`

---

# ⚙️ How the System Works (Step-by-Step)

## 1️⃣ Router Node

**Purpose:** Decide if research is needed.

Input:

```python
Topic: "Self Attention in Transformer Architecture"
```

Output:

```python
{
  needs_research: False,
  mode: "closed_book",
  queries: []
}
```

Modes:

* `closed_book` → evergreen topics
* `hybrid` → mix of evergreen + fresh info
* `open_book` → latest news / volatile topics

---

## 2️⃣ Research Node (Conditional)

If `needs_research=True`:

* Uses **TavilySearchResults**
* Fetches up to 6 results per query
* Normalizes and deduplicates URLs
* Outputs structured `EvidenceItem` objects

This ensures:

* No hallucinated links
* Controlled citations
* Clean evidence pack

---

## 3️⃣ Orchestrator Node (Planner Agent)

This is the brain of the system.

It generates a structured `Plan` object:

```python
class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal["explainer", "tutorial", "news_roundup", ...]
    tasks: List[Task]
```

Each `Task` contains:

```python
class Task(BaseModel):
    id: int
    title: str
    goal: str
    bullets: List[str] (3–6 required)
    target_words: int
    requires_research: bool
    requires_citations: bool
    requires_code: bool
```

This ensures:

* Structured planning
* No vague outlines
* Every section has a measurable goal
* Code / citations flags are explicit

---

## 4️⃣ Fanout Pattern (Parallel Workers)

LangGraph uses:

```python
Send("worker", payload)
```

Each task is sent to a **separate Worker Agent**.

This enables:

* Parallel section writing
* Scalable architecture
* Clean separation of responsibilities

---

## 5️⃣ Worker Node (Section Writer)

Each worker:

* Receives ONE task
* Writes one section only
* Follows strict markdown formatting
* Adds citations if required
* Adds code snippet if required
* Respects word count ±15%

Output:

```markdown
## Section Title
<content>
```

All sections are stored as:

```python
sections: List[(task_id, markdown)]
```

---

## 6️⃣ Reducer Subgraph (Advanced Design)

This is a nested LangGraph workflow:

```
merge_content
    ↓
decide_images
    ↓
generate_and_place_images
```

### 🔹 merge_content

* Orders sections by task ID
* Creates full blog markdown

### 🔹 decide_images

* LLM analyzes blog
* Decides if diagrams are needed
* Inserts placeholders:

  ```
  [[IMAGE_1]]
  [[IMAGE_2]]
  ```

Returns:

```python
GlobalImagePlan:
    md_with_placeholders
    images: List[ImageSpec]
```

---

### 🔹 generate_and_place_images

For each image spec:

* Calls Gemini image model
* Saves image in `/images`
* Replaces placeholder with:

```markdown
![alt](images/file.png)
*caption*
```

Graceful fallback:

* If image generation fails → Inserts diagnostic block instead of crashing.

---

# 📦 Final Output

* `<Blog_Title>.md`
* `/images/*.png`
* Fully formatted blog
* Technical diagrams embedded

---

# 🎯 Key Engineering Highlights

✅ Multi-agent architecture
✅ Structured LLM outputs with Pydantic validation
✅ Tool-calling enforcement
✅ Conditional routing
✅ Research grounding with citation control
✅ Parallel execution with LangGraph fanout
✅ Subgraph composition (Reducer Graph inside Main Graph)
✅ AI image generation pipeline
✅ Production-ready error handling

---

# 🔥 Why This Project

This project demonstrates:

* Agentic AI system design
* Workflow orchestration
* Structured output enforcement
* Parallel LLM task execution
* Tool integration
* Research-grounded generation
* Image synthesis pipeline
* End-to-end automation

This is not just prompting —
This is **AI system engineering**.

---

# 🚀 How to Run

```bash
pip install -r requirements.txt
```

Set environment variables:

```bash
OPENAI_API_KEY=...
TAVILY_API_KEY=...
GOOGLE_API_KEY=...
```

Run:

```python
run("Self Attention in Transformer Architecture")
```

Output:

* Markdown file generated
* Images stored in `/images`

---

# 🧩 Future Improvements

* Add citation auto-formatting (APA/MLA)
* Add cost tracking
* Deploy as SaaS

---

# 👨‍💻 Author

**Danyal Arshad BS Computer Science Focus Areas: Generative AI, NLP, Agentic Systems, LLM Engineering**

---

