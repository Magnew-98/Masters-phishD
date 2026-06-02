# Agentic AI for Phishing Detection

Masters research project comparing agentic AI architectures for phishing email detection using the Enron dataset and a locally-hosted LLM (llama3.1 via Ollama).

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10+ | See Windows note below |
| [Ollama](https://ollama.com) | Latest | Installed automatically by setup script |
| llama3.1 model | — | Pulled automatically by setup script |
| Enron dataset | — | Must be placed manually (see below) |

### Dataset

Place `Enron.csv` at:
```
src/datasets/Enron.csv
```

The file must have columns: `subject`, `body`, `label` (0 = legitimate, 1 = phishing).

---

## Installation

### Linux / macOS

```bash
git clone <repo-url>
cd Masters-phishD
chmod +x install_prerequisites.sh
./install_prerequisites.sh
```

The script will:
- Check for Python 3.10+
- Create a `.venv` virtual environment
- Install all Python dependencies (including RAG dependencies: `sentence-transformers`, `chromadb`)
- Install Ollama if not present
- Start the Ollama daemon
- Pull the `llama3.1` model
- Verify the dataset exists
- Create the `run_experiment.sh` wrapper

### Windows

Open **PowerShell** (not Command Prompt) and run:

```powershell
git clone <repo-url>
cd Masters-phishD
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\install_prerequisites.ps1
```

The script handles everything including Ollama installation. If you have **Python from the Microsoft Store**, the script detects this automatically and uses a workaround for a known venv creation issue.

> **Note:** After installation, use `run_experiment.ps1` rather than calling Python directly — it sets the correct working directory and uses the project's virtual environment.

---

## Running Experiments

All experiments are run through a single command. The `--components` flag specifies which agent(s) to use (space-separated), and the `--rag` flag optionally prepends a RAG retrieval step.

### Linux / macOS

```bash
./run_experiment.sh --components binary --batch-size 20
```

### Windows

```powershell
.\run_experiment.ps1 --components binary --batch-size 20
```

The examples below use the Windows syntax. Replace `.\run_experiment.ps1` with `./run_experiment.sh` on Linux/macOS.

---

## Available Agents

| Name | Description |
|---|---|
| `binary` | Baseline — general phishing indicator analysis followed by classification |
| `technical` | Specialist — URLs, domain spoofing, file attachments, header anomalies |
| `sentiment` | Specialist — psychological manipulation (unsolicited reward, implausible authority, extreme urgency) |
| `linguistic` | Specialist — homoglyph substitutions, brand misspellings, grammar and register anomalies |

When more than one agent is specified, a **coordinator** node synthesises all specialist analyses and their directional leanings into a final classification.

---

## Usage

### Test a single agent (dry run — nothing saved to disk)

```powershell
.\run_experiment.ps1 --components binary --batch-size 100 --dry-run
.\run_experiment.ps1 --components technical --batch-size 100 --dry-run
.\run_experiment.ps1 --components sentiment --batch-size 100 --dry-run
.\run_experiment.ps1 --components linguistic --batch-size 100 --dry-run
```

### Combine specialists with a coordinator

```powershell
.\run_experiment.ps1 --components technical sentiment linguistic --batch-size 20 --dry-run
```

### Add RAG

The `--rag` flag prepends a retrieval step that finds the 3 most similar emails from the knowledge base before analysis. Works with any combination:

```powershell
.\run_experiment.ps1 --components technical sentiment linguistic --rag --batch-size 20 --dry-run
```

> On first use of `--rag`, the knowledge base index is built from the RAG training split (~2,977 emails). This runs once and is saved to `results/rag_index/`. All subsequent runs load the existing index.

### Run parallel specialist nodes

On hardware with sufficient VRAM, specialist nodes can run concurrently rather than sequentially. Requires Ollama to be configured for concurrent requests:

```powershell
# Set before starting Ollama (Windows)
$env:OLLAMA_NUM_PARALLEL = 3
ollama serve

# Then run with --parallel
.\run_experiment.ps1 --components technical sentiment linguistic --parallel --batch-size 20
```

### Commit real results (drop `--dry-run`)

```powershell
.\run_experiment.ps1 --components binary --batch-size 100
.\run_experiment.ps1 --components technical sentiment linguistic --rag --batch-size 50
```

Results are appended to `results/results.csv`. Batches can be run across multiple sessions — the harness tracks which emails have already been processed and resumes automatically.

### Check accumulated metrics without running inference

```powershell
.\run_experiment.ps1 --components binary --metrics-only
.\run_experiment.ps1 --components technical sentiment linguistic --rag --metrics-only
```

---

## Architectures Under Evaluation

| Agent name (auto-derived) | Command | Description |
|---|---|---|
| `binary` | `--components binary` | Baseline single-agent |
| `technical_sentiment_linguistic` | `--components technical sentiment linguistic` | Multi-specialist, no RAG |
| `technical_sentiment_linguistic_rag` | `--components technical sentiment linguistic --rag` | Multi-specialist with RAG |

---

## Dataset Split

On first run, the dataset is split once and saved to `results/split.json`:

| Split | Size | Purpose |
|---|---|---|
| RAG knowledge base | ~2,977 emails (10%, stratified) | Source for RAG retrieval index |
| Test set | ~26,791 emails (90%, stratified) | Evaluation — used by all agents |

All agents, including those without RAG, are evaluated on the same test set for a fair comparison.

---

## Results

Results are stored in `results/results.csv`:

| Column | Description |
|---|---|
| `email_id` | Stable row ID from the original dataset |
| `agent_name` | Auto-derived from components, e.g. `technical_sentiment_linguistic_rag` |
| `true_label` | Ground truth: `phishing` or `legitimate` |
| `prediction` | Agent prediction: `phishing` or `legitimate` |
| `confidence` | Model confidence between 0.0 and 1.0 |

All architectures share one file and are independently tracked by `agent_name`, enabling direct comparison.

---

## Project Structure

```
Masters-phishD/
├── src/
│   ├── agents/
│   │   ├── binary/           # Baseline agent (analyse + classify)
│   │   ├── technical/        # Technical specialist (analyse + classify)
│   │   ├── sentiment/        # Sentiment specialist (analyse + classify)
│   │   ├── linguistic/       # Linguistic specialist (analyse + classify)
│   │   ├── coordinator/      # Multi-specialist coordinator (classify)
│   │   ├── rag/              # RAG retrieval node
│   │   └── shared_llm.py     # Shared Ollama LLM instance (llama3.1)
│   ├── graph/
│   │   └── factory.py        # Dynamically builds any agent combination
│   ├── schemas/
│   │   ├── state.py          # Shared LangGraph state (EmailState)
│   │   └── outputs.py        # Pydantic output schemas
│   ├── evaluation/
│   │   └── run_experiment.py # Evaluation harness (batching, metrics, split)
│   └── datasets/
│       └── Enron.csv         # Dataset (not tracked in git)
├── results/
│   ├── results.csv           # Accumulated experiment results
│   ├── split.json            # Stable RAG/test split (generated once)
│   └── rag_index/            # ChromaDB vector store (built on first RAG run)
├── install_prerequisites.sh  # Linux/macOS setup script
├── install_prerequisites.ps1 # Windows setup script
├── run_experiment.sh         # Linux/macOS experiment runner
└── run_experiment.ps1        # Windows experiment runner
```

---

## How It Works

### Single agent

```
email → [analyse] → [classify] → prediction + confidence
```

### Multi-specialist (with optional RAG)

```
email → [rag_retrieve]          ← optional, finds 3 similar known emails
              ↓
       [analyse_technical]  ─┐
       [analyse_sentiment]   ├─→ [coordinate] → prediction + confidence
       [analyse_linguistic] ─┘
```

Each specialist produces a prose analysis and a directional leaning (`phishing` / `legitimate` / `uncertain`). The coordinator receives all analyses and leanings, applies an evidence hierarchy (technical > linguistic > sentiment), and makes the final classification.

The graph for any combination is built dynamically by `factory.py` — no separate graph files are maintained per architecture.

### LLM

All nodes use `llama3.1` via Ollama locally at `temperature=0.1`, `seed=42`. Inference is fully local with no external API calls.

### Reproducibility note

Exact numerical results may vary between runs even with a fixed seed, due to hardware-level floating-point non-determinism in local LLM inference. Results should be interpreted in terms of relative architecture comparisons rather than absolute metric values.
