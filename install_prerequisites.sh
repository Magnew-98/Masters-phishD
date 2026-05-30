#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
PYTHON_MIN_MAJOR=3
PYTHON_MIN_MINOR=10

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ── 1. Python version check ──────────────────────────────────────────────────
info "Checking Python version..."
PYTHON_BIN=""
for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" &>/dev/null; then
        ver=$("$candidate" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        major="${ver%%.*}"; minor="${ver##*.}"
        if [[ "$major" -gt "$PYTHON_MIN_MAJOR" ]] || \
           { [[ "$major" -eq "$PYTHON_MIN_MAJOR" ]] && [[ "$minor" -ge "$PYTHON_MIN_MINOR" ]]; }; then
            PYTHON_BIN="$candidate"
            info "Using Python $ver ($PYTHON_BIN)"
            break
        fi
    fi
done

if [[ -z "$PYTHON_BIN" ]]; then
    error "Python $PYTHON_MIN_MAJOR.$PYTHON_MIN_MINOR+ not found."
    error "Install it with: sudo apt install python3.10  (or 3.11/3.12)"
    exit 1
fi

# ── 2. pip / venv availability ───────────────────────────────────────────────
info "Checking pip and venv..."
if ! "$PYTHON_BIN" -m pip --version &>/dev/null; then
    warn "pip not found — attempting to install via ensurepip..."
    "$PYTHON_BIN" -m ensurepip --upgrade || {
        error "Could not bootstrap pip. Try: sudo apt install python3-pip"
        exit 1
    }
fi

if ! "$PYTHON_BIN" -m venv --help &>/dev/null; then
    PYVER=$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    warn "venv module missing — installing python${PYVER}-venv..."
    sudo apt-get install -y "python${PYVER}-venv" || {
        error "Could not install python${PYVER}-venv. Run: sudo apt install python${PYVER}-venv"
        exit 1
    }
fi

# ── 3. Create / reuse virtual environment ────────────────────────────────────
if [[ -d "$VENV_DIR" ]] && [[ ! -x "$VENV_DIR/bin/pip" ]]; then
    warn "Virtual environment at $VENV_DIR is incomplete — removing and recreating..."
    rm -rf "$VENV_DIR"
fi

if [[ -d "$VENV_DIR" ]]; then
    info "Virtual environment already exists at $VENV_DIR"
else
    info "Creating virtual environment at $VENV_DIR ..."
    if ! "$PYTHON_BIN" -m venv "$VENV_DIR" 2>&1; then
        PYVER=$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        warn "venv creation failed — installing python${PYVER}-venv..."
        sudo apt-get install -y "python${PYVER}-venv" || {
            error "Could not install python${PYVER}-venv. Run: sudo apt install python${PYVER}-venv"
            exit 1
        }
        "$PYTHON_BIN" -m venv "$VENV_DIR"
    fi
fi

PIP="$VENV_DIR/bin/pip"
PYTHON="$VENV_DIR/bin/python3"

info "Upgrading pip..."
"$PIP" install --quiet --upgrade pip

# ── 4. Install Python packages ───────────────────────────────────────────────
info "Installing Python dependencies..."
"$PIP" install --quiet \
    "langgraph>=1.2" \
    "langchain>=1.3" \
    "langchain-core>=1.4" \
    "langchain-community>=0.4" \
    "langchain-ollama>=0.1" \
    "ollama>=0.4" \
    "pydantic>=2.7" \
    "pandas>=2.0" \
    "scikit-learn>=1.5" \
    "numpy>=1.26" \
    "scipy>=1.12" \
    "tqdm>=4.66" \
    "requests>=2.31" \
    "python-dotenv>=1.0" \
    "PyYAML>=6.0"

info "Python dependencies installed."

# ── 5. Ollama installation ────────────────────────────────────────────────────
info "Checking for Ollama..."
if ! command -v ollama &>/dev/null; then
    warn "Ollama not found. Installing via official install script..."
    if command -v curl &>/dev/null; then
        curl -fsSL https://ollama.com/install.sh | sh
    elif command -v wget &>/dev/null; then
        wget -qO- https://ollama.com/install.sh | sh
    else
        error "Neither curl nor wget found. Install Ollama manually from https://ollama.com"
        exit 1
    fi
else
    info "Ollama is already installed: $(ollama --version 2>/dev/null || echo '(version unknown)')"
fi

# ── 6. Ensure Ollama daemon is running ───────────────────────────────────────
info "Checking Ollama service..."
if ! curl -sf http://localhost:11434/api/tags &>/dev/null; then
    info "Starting Ollama daemon in background..."
    nohup ollama serve &>/tmp/ollama.log &
    OLLAMA_PID=$!
    echo "$OLLAMA_PID" > /tmp/ollama.pid
    # Wait up to 15s for the server to be ready
    for i in {1..15}; do
        if curl -sf http://localhost:11434/api/tags &>/dev/null; then
            info "Ollama daemon started (PID $OLLAMA_PID)."
            break
        fi
        sleep 1
    done
    if ! curl -sf http://localhost:11434/api/tags &>/dev/null; then
        error "Ollama did not start within 15 seconds. Check /tmp/ollama.log"
        exit 1
    fi
else
    info "Ollama daemon is already running."
fi

# ── 7. Pull llama3.1 model ───────────────────────────────────────────────────
MODEL="llama3.1"
info "Checking for model '$MODEL'..."
if ollama list 2>/dev/null | grep -q "^${MODEL}"; then
    info "Model '$MODEL' is already available."
else
    info "Pulling model '$MODEL' — this may take several minutes..."
    ollama pull "$MODEL"
    info "Model '$MODEL' pulled successfully."
fi

# ── 8. Verify dataset ────────────────────────────────────────────────────────
DATASET="$PROJECT_DIR/src/datasets/Enron.csv"
if [[ -f "$DATASET" ]]; then
    rows=$(("$(wc -l < "$DATASET")" - 1))
    info "Dataset found: $DATASET ($rows rows)"
else
    error "Dataset not found at $DATASET"
    error "Place the Enron.csv file there before running the experiment."
    exit 1
fi

# ── 9. Smoke-test imports ─────────────────────────────────────────────────────
info "Verifying Python imports..."
"$PYTHON" - <<'EOF'
import langgraph, langchain, langchain_ollama, ollama
import pandas, sklearn, numpy, tqdm
print("All imports OK")
EOF

# ── 10. Write run_experiment wrapper ─────────────────────────────────────────
RUNNER="$PROJECT_DIR/run_experiment.sh"
cat > "$RUNNER" <<'RUNNER_EOF'
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/.venv/bin/python3" -m src.evaluation.run_experiment "$@"
RUNNER_EOF
chmod +x "$RUNNER"
info "Created runner: $RUNNER"

# ── 11. Done ──────────────────────────────────────────────────────────────────
echo ""
info "Setup complete. Run the experiment with:"
echo ""
echo "    $RUNNER"
echo ""
