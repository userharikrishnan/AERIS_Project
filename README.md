# AERIS — Autonomous Execution and Reasoning Intelligence System

> *"What do you need?" — Your Jarvis-like personal AI assistant, built completely from scratch.*

AERIS is a distributed intelligence system with a custom-built SLM at its core. It understands natural language, executes real system actions, learns from every interaction, and gets smarter with use.

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Ensure models are trained (skip if checkpoints exist)
python train.py

# 3. Start AERIS
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Then interact via HTTP API at `http://localhost:8000`

---

## 🧠 Architecture Overview

```
User Input (text)
      │
      ▼
  AttentionEngine  ── filters noise, priority management
      │
      ▼
  NLPProcessor ──────── intent classification (BiGRU + Multi-Head Attention)
      │                  entity extraction, context enrichment
      ▼
  ReasoningEngine ───── memory retrieval + plan generation + verification
      │
      ▼
  CommandEngine ─────── action routing with confidence gating
      │
      ▼
  SmartConfirmation ─── first time: ask | repeat: auto-approve
      │
      ├── [auto-approve] → PlanExecutor → ToolDispatcher → ACTUAL TOOL EXECUTION
      └── [first time]   → User Confirmation → ToolDispatcher → ACTUAL TOOL EXECUTION
                                                      │
                                                 LearningEngine
                                                 (records outcome)
```

---

## 🛠️ Available Tools

| Tool | Capabilities | Description |
|------|-------------|-------------|
| **AppTool** | `open_app`, `close_app`, `launch` | Launch any Windows app (dynamic discovery via registry + filesystem scan) |
| **BrowserTool** | `web_search`, `web_navigate` | Open URLs and search queries in your default browser |
| **WebScraperTool** | `web_scrape`, `scrape` | Extract structured content from any web page |
| **ReportGeneratorTool** | `generate_report` | Create .md, .html, .txt, or .pdf reports and save to disk |
| **FileSystemTool** | `file_read`, `file_write`, `file_list`, `file_delete` | Full filesystem operations |
| **SystemTool** | `screenshot`, `clipboard_read`, `clipboard_write`, `system_info`, `lock_screen` | OS-level actions |
| **ProcessTool** | `list_processes`, `process_info`, `kill_process` | Process management (psutil-powered) |

---

## 🔐 Smart Dual-Mode Confirmation

AERIS uses an intelligent confirmation system:

- **First time**: AERIS asks for your approval before doing anything
- **Next time**: Same action auto-executes without prompting
- **Error recovery**: If something goes wrong, trust is degraded and AERIS asks again
- **Always-confirm**: High-risk actions (delete, format, kill) always ask regardless

```json
// Example: First time
{
  "confirmation_required": true,
  "first_time": true,
  "smart_confirmation_note": "First time executing 'open_app' — confirmation required.",
  "action": "open_app",
  "params": {"app": "notepad"}
}

// Example: Second time (same action)
{
  "mode": "auto_executed",
  "auto_approved": true,
  "reason": "Auto-approved: 'open_app' confirmed 1 time(s) previously.",
  "response": "Opening notepad for you now."
}
```

---

## 🔗 API Reference

### Core Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Health status + active tools |
| `/status` | GET | Full system status |
| `/core/input` | POST | Send text to AERIS → intent classification + response |
| `/core/confirm` | POST | Confirm or reject a planned action |
| `/core/memory` | GET | View stored memories |
| `/core/goals` | GET | View active goals |

### Teaching Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/teach/app` | POST | Teach AERIS a new app path |
| `/teach/browser` | POST | Set your preferred browser |
| `/correct` | POST | Correct AERIS's wrong interpretation |

### Update Management

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/updates/check` | GET | Check for pending model updates in `updates/` folder |
| `/updates/apply` | POST | Apply a pending model update package |

---

## 💡 Example Interactions

### Say "hi"
```bash
curl -X POST http://localhost:8000/core/input \
  -H "Content-Type: application/json" \
  -d '{"text": "hello"}'
```

### Open an app
```bash
curl -X POST http://localhost:8000/core/input \
  -H "Content-Type: application/json" \
  -d '{"text": "open chrome"}'
```

### Scrape a website + save report
```bash
curl -X POST http://localhost:8000/core/input \
  -H "Content-Type: application/json" \
  -d '{"text": "go to https://news.ycombinator.com and scrape the headlines, then save a report to my desktop"}'
```

### Check system info
```bash
curl -X POST http://localhost:8000/core/input \
  -H "Content-Type: application/json" \
  -d '{"text": "what is my system info"}'
```

### Take a screenshot
```bash
curl -X POST http://localhost:8000/core/input \
  -H "Content-Type: application/json" \
  -d '{"text": "take a screenshot"}'
```

### Teach AERIS your browser preference
```bash
curl -X POST http://localhost:8000/teach/browser \
  -H "Content-Type: application/json" \
  -d '{"browser_name": "firefox"}'
```

### Teach AERIS an unknown app
```bash
curl -X POST http://localhost:8000/teach/app \
  -H "Content-Type: application/json" \
  -d '{"app_name": "postman", "path": "C:/Users/User/AppData/Local/Postman/Postman.exe"}'
```

### End of session
```bash
curl -X POST http://localhost:8000/core/input \
  -H "Content-Type: application/json" \
  -d '{"text": "thanks, that will be all"}'
```

---

## 📦 Model Update System

AERIS uses a local file-based update system. No cloud required.

To update your models:
1. Download an AERIS model update package (`.zip`)
2. Drop it in the `updates/` folder
3. Call `POST /updates/apply`
4. Restart AERIS to load new models

The update system preserves all your personal data (preferences, app mappings, corrections).

---

## 🧩 New Packages — What & Why

| Package | What it does | Install |
|---------|-------------|---------|
| `requests` | HTTP fetching for web scraper | `pip install requests` |
| `beautifulsoup4` | HTML parsing for web scraper | `pip install beautifulsoup4` |
| `lxml` | Fast HTML parser (bs4 backend) | `pip install lxml` |
| `psutil` | CPU, RAM, disk, process info | `pip install psutil` |
| `pyperclip` | Clipboard read/write | `pip install pyperclip` |
| `reportlab` | PDF report generation (optional) | `pip install reportlab` |

---

## 🗂️ Project Structure

```
AERIS_Project/
├── main.py                      # FastAPI app, AerisSystem container
├── train.py                     # ML training pipeline
├── requirements.txt             # All dependencies
├── config/
│   └── update_manifest.json     # Version tracking
├── checkpoints/                 # Trained model files
│   ├── aeris_slm_best.pt        # Language model
│   ├── intent_classifier.pt     # Intent classifier
│   ├── plan_scorer.pt           # Plan confidence scorer
│   └── tokenizer.json           # Vocabulary
├── updates/                     # Drop model update .zip here
├── db/
│   ├── memory.db                # Episodic + long-term memory
│   ├── preferences.db           # Preferences + auto-approvals
│   └── apps.db                  # Dynamic app registry
├── models/                      # Neural network architectures
│   ├── slm.py                   # Transformer SLM (NEW: Upgraded)
│   ├── intent_classifier.py     # BiGRU + Multi-Head Attention
│   ├── tokenizer.py             # Word-level tokenizer
│   └── embeddings.py            # Embedding layer
├── services/
│   ├── tool_dispatcher.py       # NEW: Central tool execution dispatcher
│   ├── smart_confirmation.py    # NEW: Dual-mode auto-approve system
│   ├── session_engine.py        # NEW: Session lifecycle management
│   ├── plan_executor.py         # NEW: Multi-step plan execution
│   ├── app_registry.py          # NEW: Dynamic app discovery
│   ├── learning_engine.py       # NEW: Passive/active learning
│   ├── update_manager.py        # NEW: Local model update system
│   ├── nlp_processor.py         # Intent + entity extraction
│   ├── reasoning_engine.py      # Planning + verification
│   ├── memory_engine.py         # Persistent memory
│   ├── language_engine.py       # SLM response generation
│   ├── command_engine.py        # Action routing
│   ├── planner.py               # Step-plan generation
│   └── tools/
│       ├── app_tool.py          # UPGRADED: Dynamic app launcher
│       ├── browser_tool.py      # Web navigation
│       ├── filesystem_tool.py   # File operations
│       ├── web_scraper_tool.py  # NEW: Web content extraction
│       ├── report_generator_tool.py  # NEW: Report creation
│       ├── system_tool.py       # NEW: Screenshot, clipboard, sysinfo
│       └── process_tool.py      # NEW: Process management
└── routes/
    ├── core.py                  # Main API endpoints (UPGRADED)
    └── agent.py                 # Tool execution boundary
```

---

## ⚡ What Changed in v0.2

| Component | Before | After |
|-----------|--------|-------|
| Tool execution | Planned but never called | `ToolDispatcher` calls actual tools ✅ |
| App launch | 7 hardcoded apps | Dynamic registry (100s of apps) ✅ |
| Confirmation | Always ask | First time ask, then auto-approve ✅ |
| Session | None | Full session start/end lifecycle ✅ |
| SLM | GRU, 12 tokens | Transformer, 50 tokens, GELU ✅ |
| Responses | "Ready." | Jarvis-style intent-aware responses ✅ |
| Web scraping | Not available | Full HTML extraction ✅ |
| Reports | Not available | .md / .html / .pdf generation ✅ |
| System info | Not available | CPU/RAM/disk/battery via psutil ✅ |
| Screenshots | Not available | Saves to Desktop/AERIS_Screenshots ✅ |
| Clipboard | Not available | Read/write via pyperclip ✅ |
| Process control | Not available | List/inspect/terminate (protected) ✅ |
| Learning | Passive signals only | Full learning engine with corrections ✅ |
| Updates | Manual file swap | Local zip update manager ✅ |

---

## 🔜 Next Steps (Recommended)

1. **Retrain intent classifier** with new intents: `WEB_SCRAPE`, `GENERATE_REPORT`, `SYSTEM_INFO`
2. **Expand training data** in `data/intent_training_data.py` and `data/aeris_training_data.py`
3. **Retrain SLM** with the new Transformer architecture for 2x quality improvement
4. **Add frontend UI** — Qt6 desktop window or web dashboard to display AERIS status

---

*AERIS is built entirely with custom ML — no third-party LLMs, no external AI APIs.*
