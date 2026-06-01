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
- Install all Python dependencies
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

All experiments are run through a single command. The `--components` flag specifies which agent(s) to use, and the `--rag` flag optionally prepends a RAG retrieval step.

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
| `binary` | Baseline — general phishing indicator analysis + classification |
| `technical` | Specialist — URLs, domain spoofing, file attachments, header anomalies |
| `sentiment` | Specialist — psychological manipulation tactics (reward, authority, extreme urgency) |
| `linguistic` | Specialist — homoglyphs, spelling patterns, grammar anomalies *(coming soon)* |

When more than one agent is specified, a **coordinator** node synthesises all specialist analyses into a final classification.

---

## Usage Examples

### Test a single agent (dry run — nothing saved)

```powershell
.\run_experiment.ps1 --components binary --batch-size 20 --dry-run
.\run_experiment.ps1 --components technical --batch-size 20 --dry-run
.\run_experiment.ps1 --components sentiment --batch-size 20 --dry-run
```

### Combine agents with a coordinator

```powershell
.\run_experiment.ps1 --components binary technical --batch-size 20 --dry-run
.\run_experiment.ps1 --components binary sentiment --batch-size 20 --dry-run
.\run_experiment.ps1 --components binary technical sentiment --batch-size 20 --dry-run
```

### Add RAG (retrieval-augmented generation)

The `--rag` flag prepends a retrieval step that finds similar known emails before analysis. Works with any combination:

```powershell
.\run_experiment.ps1 --components binary --rag --batch-size 20 --dry-run
.\run_experiment.ps1 --components binary technical --rag --batch-size 20 --dry-run
.\run_experiment.ps1 --components binary technical sentiment --rag --batch-size 20 --dry-run
```

> On first use of `--rag`, the knowledge base index is built from the RAG training split (~2,977 emails). This takes a few minutes and runs once — subsequent runs reuse the saved index at `results/rag_index/`.

### Commit real results (drop `--dry-run`)

```powershell
.\run_experiment.ps1 --components binary --batch-size 50
.\run_experiment.ps1 --components binary technical --rag --batch-size 50
```

Results are appended to `results/results.csv`. You can run batches across multiple sessions — the harness tracks which emails have already been processed and resumes automatically.

### Check accumulated metrics without running inference

```powershell
.\run_experiment.ps1 --components binary --metrics-only
.\run_experiment.ps1 --components binary technical --rag --metrics-only
```

---

## Dataset Split

On first run, the dataset is split once and saved to `results/split.json`:

| Split | Size | Purpose |
|---|---|---|
| RAG training set | ~2,977 emails (10%) | Knowledge base for RAG retrieval |
| Test set | ~26,791 emails (90%) | Evaluation for all agents |

All agents — including those without RAG — are evaluated on the same test set for a fair comparison.

---

## Results

Results are stored in `results/results.csv` with columns:

| Column | Description |
|---|---|
| `email_id` | Stable row ID from the original dataset |
| `agent_name` | Auto-derived from components, e.g. `binary_technical_rag` |
| `true_label` | Ground truth: `phishing` or `legitimate` |
| `prediction` | Agent prediction: `phishing` or `legitimate` |
| `confidence` | Confidence score between 0.0 and 1.0 |

Each agent combination is independently tracked, so all architectures share one file and can be compared directly.

---

## Project Structure

```
Masters-phishD/
├── src/
│   ├── agents/
│   │   ├── binary/          # Baseline agent (analyse + classify)
│   │   ├── technical/       # Technical specialist (analyse + classify)
│   │   ├── sentiment/       # Sentiment specialist (analyse + classify)
│   │   ├── coordinator/     # Multi-specialist coordinator
│   │   ├── rag/             # RAG retrieval node
│   │   └── shared_llm.py    # Shared Ollama LLM instance
│   ├── graph/
│   │   ├── factory.py       # Builds any agent combination dynamically
│   │   └── binary_graph.py  # (legacy standalone graph)
│   ├── schemas/
│   │   ├── state.py         # Shared LangGraph state (EmailState)
│   │   └── outputs.py       # Pydantic output schemas
│   ├── evaluation/
│   │   └── run_experiment.py  # Evaluation harness
│   └── datasets/
│       └── Enron.csv        # Dataset (not tracked in git)
├── results/
│   ├── results.csv          # Accumulated experiment results
│   ├── split.json           # Stable train/test split
│   └── rag_index/           # ChromaDB vector store (built on first RAG run)
├── install_prerequisites.sh   # Linux/macOS setup
├── install_prerequisites.ps1  # Windows setup
├── run_experiment.sh          # Linux/macOS experiment runner
└── run_experiment.ps1         # Windows experiment runner
```

---

## How It Works

Each experiment run invokes a LangGraph pipeline where nodes pass state between them:

```
[rag_retrieve]  ← optional, prepended when --rag is used
      ↓
[analyse_*]     ← one node per component, runs in sequence
      ↓
[coordinate]    ← only when multiple components are specified
      ↓
  prediction + confidence
```

Single-component runs use a dedicated classify node instead of the coordinator. The LLM for all nodes is `llama3.1` running locally via Ollama at `temperature=0.2`.
