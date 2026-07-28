# ✍️ Blog Writing Agent
 
A chat-based Streamlit app that researches, plans, drafts, and illustrates a full blog post from a single topic prompt — powered by a LangGraph agent.
 
## Features
 
- 💬 Chat-style interface — just type a topic, no forms
- 🔎 Auto web research via Tavily when the topic needs current info
- 🧩 Structured planning (title, audience, tone, sections) before writing
- 🖼️ Auto-generated diagrams (Cloudflare Flux) inserted into the post
- 📄 One-click PDF export (images embedded, links clickable)
- 🔒 One blog per session — chat locks after generation
## Project structure
 
```
client.py         # Streamlit frontend
server.py         # LangGraph backend (router → research → plan → write → images)
requirements.txt  # Python dependencies
packages.txt      # System packages (for PDF export via WeasyPrint)
```
 
## Getting started (local)
 
1. **Clone the repo**
```bash
   git clone https://github.com/<your-username>/<your-repo>.git
   cd <your-repo>
```
 
2. **Create and activate a virtual environment**
   macOS/Linux:
```bash
   python3 -m venv venv
   source venv/bin/activate
```
   Windows (PowerShell):
```powershell
   python -m venv venv
   venv\Scripts\activate
```
 
3. **Install Python dependencies**
```bash
   pip install -r requirements.txt
```
 
4. **Install system packages for PDF export** (WeasyPrint needs these; see `packages.txt`)
   Debian/Ubuntu/WSL:
```bash
   sudo apt-get update && sudo apt-get install -y $(cat packages.txt)
```
   macOS:
```bash
   brew install pango cairo gdk-pixbuf libffi
```
   Windows: see the [WeasyPrint install docs](https://weasyprint.readthedocs.io/en/stable/first_steps.html#windows), or just run this project inside WSL using the Ubuntu command above.
 
5. **Add your API keys**
   Create a `.env` file in the project root:
```
   GROQ_API_KEY=your_key
   TAVILY_API_KEY=your_key
   CLOUDFLARE_ACCOUNT_ID=your_id
   CLOUDFLARE_API_TOKEN=your_token
```
 
6. **Run the app**
```bash
   streamlit run client.py
```
   Opens automatically at `http://localhost:8501`.
  
## Tech stack
Streamlit · LangGraph · LangChain (Groq) · Tavily Search · Cloudflare Workers AI (image gen) · WeasyPrint (PDF export)
