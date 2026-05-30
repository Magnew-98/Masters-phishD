#Requires -Version 5.1
$ErrorActionPreference = 'Stop'

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir    = Join-Path $ProjectDir '.venv'

function Write-Info { Write-Host "[INFO]  $args" -ForegroundColor Green }
function Write-Warn { Write-Host "[WARN]  $args" -ForegroundColor Yellow }
function Write-Err  { Write-Host "[ERROR] $args" -ForegroundColor Red }

# --- 1. Python version check -------------------------------------------------
Write-Info "Checking Python version..."
$PythonBin = $null
foreach ($candidate in @('python3.12', 'python3.11', 'python3.10', 'python3', 'python')) {
    try {
        $ver = & $candidate -c "import sys; print(str(sys.version_info.major) + '.' + str(sys.version_info.minor))" 2>$null
        if ($ver) {
            $parts = $ver.Trim().Split('.')
            $major = [int]$parts[0]
            $minor = [int]$parts[1]
            if ($major -gt 3 -or ($major -eq 3 -and $minor -ge 10)) {
                $PythonBin = $candidate
                Write-Info "Using Python $ver ($PythonBin)"
                break
            }
        }
    } catch {}
}

if (-not $PythonBin) {
    Write-Err "Python 3.10+ not found."
    Write-Err "Download from https://www.python.org/downloads/ and check 'Add to PATH'."
    exit 1
}

# --- 2. Create / reuse virtual environment -----------------------------------
$VenvPip = Join-Path $VenvDir 'Scripts\pip.exe'

if ((Test-Path $VenvDir) -and (-not (Test-Path $VenvPip))) {
    Write-Warn "Incomplete venv at $VenvDir - removing and recreating..."
    Remove-Item -Recurse -Force $VenvDir
}

if (-not (Test-Path $VenvDir)) {
    Write-Info "Creating virtual environment at $VenvDir ..."

    # Windows Store Python sandboxes ensurepip subprocess calls and hangs.
    # Detect it by checking if the resolved executable path is under WindowsApps.
    $resolvedPython = & $PythonBin -c "import sys; print(sys.executable)" 2>$null
    $isStorePython  = $resolvedPython -match 'WindowsApps'

    if ($isStorePython) {
        Write-Warn "Windows Store Python detected - using --without-pip workaround..."
        & $PythonBin -m venv --without-pip $VenvDir
        if ($LASTEXITCODE -ne 0) { Write-Err "Failed to create venv"; exit 1 }
        Write-Info "Bootstrapping pip via get-pip.py..."
        $getPipPath = Join-Path $env:TEMP "get-pip.py"
        Invoke-WebRequest -Uri "https://bootstrap.pypa.io/get-pip.py" -OutFile $getPipPath -UseBasicParsing
        & (Join-Path $VenvDir 'Scripts\python.exe') $getPipPath --quiet
        if ($LASTEXITCODE -ne 0) { Write-Err "Failed to install pip into venv"; exit 1 }
    } else {
        & $PythonBin -m venv $VenvDir
        if ($LASTEXITCODE -ne 0) { Write-Err "Failed to create venv"; exit 1 }
    }
} else {
    Write-Info "Virtual environment already exists at $VenvDir"
}

$Pip    = Join-Path $VenvDir 'Scripts\pip.exe'
$Python = Join-Path $VenvDir 'Scripts\python.exe'

Write-Info "Upgrading pip..."
& $Pip install --quiet --upgrade pip

# --- 3. Install Python packages ----------------------------------------------
Write-Info "Installing Python dependencies..."
$packages = @(
    'langgraph>=1.2',
    'langchain>=1.3',
    'langchain-core>=1.4',
    'langchain-community>=0.4',
    'langchain-ollama>=0.1',
    'ollama>=0.4',
    'pydantic>=2.7',
    'pandas>=2.0',
    'scikit-learn>=1.5',
    'numpy>=1.26',
    'scipy>=1.12',
    'tqdm>=4.66',
    'requests>=2.31',
    'python-dotenv>=1.0',
    'PyYAML>=6.0'
)
& $Pip install --quiet @packages
if ($LASTEXITCODE -ne 0) { Write-Err "pip install failed"; exit 1 }
Write-Info "Python dependencies installed."

# --- 4. Ollama installation --------------------------------------------------
Write-Info "Checking for Ollama..."
$ollamaCmd = Get-Command ollama -ErrorAction SilentlyContinue

if (-not $ollamaCmd) {
    Write-Warn "Ollama not found. Downloading installer..."
    $installerPath = Join-Path $env:TEMP "OllamaSetup.exe"
    Invoke-WebRequest -Uri "https://ollama.com/download/OllamaSetup.exe" -OutFile $installerPath
    Write-Info "Running Ollama installer - follow the prompts..."
    Start-Process -FilePath $installerPath -Wait
    $env:PATH = [System.Environment]::GetEnvironmentVariable('PATH', 'Machine') + ';' +
                [System.Environment]::GetEnvironmentVariable('PATH', 'User')
    $ollamaCmd = Get-Command ollama -ErrorAction SilentlyContinue
    if (-not $ollamaCmd) {
        Write-Err "Ollama still not on PATH after install. Open a new terminal and re-run this script."
        exit 1
    }
} else {
    $ollamaVer = & ollama --version 2>$null
    Write-Info "Ollama already installed: $ollamaVer"
}

# --- 5. Ensure Ollama daemon is running --------------------------------------
Write-Info "Checking Ollama service..."
$ollamaRunning = $false
try {
    Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -UseBasicParsing -ErrorAction Stop | Out-Null
    $ollamaRunning = $true
    Write-Info "Ollama daemon is already running."
} catch {}

if (-not $ollamaRunning) {
    Write-Info "Starting Ollama daemon in background..."
    Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
    for ($i = 0; $i -lt 15; $i++) {
        Start-Sleep 1
        try {
            Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -UseBasicParsing -ErrorAction Stop | Out-Null
            $ollamaRunning = $true
            break
        } catch {}
    }
    if (-not $ollamaRunning) {
        Write-Err "Ollama did not start within 15 seconds."
        exit 1
    }
    Write-Info "Ollama daemon started."
}

# --- 6. Pull llama3.1 model --------------------------------------------------
$Model = "llama3.1"
Write-Info "Checking for model '$Model'..."
$modelList = & ollama list 2>$null
if ($modelList -match [regex]::Escape($Model)) {
    Write-Info "Model '$Model' is already available."
} else {
    Write-Info "Pulling model '$Model' - this may take several minutes..."
    & ollama pull $Model
    if ($LASTEXITCODE -ne 0) { Write-Err "Failed to pull model '$Model'"; exit 1 }
    Write-Info "Model '$Model' pulled successfully."
}

# --- 7. Verify dataset -------------------------------------------------------
$Dataset = Join-Path $ProjectDir 'src\datasets\Enron.csv'
if (Test-Path $Dataset) {
    $reader = [System.IO.File]::OpenText($Dataset)
    $lineCount = 0
    while ($reader.ReadLine() -ne $null) { $lineCount++ }
    $reader.Close()
    $rows = $lineCount - 1
    Write-Info "Dataset found: $Dataset ($rows rows)"
} else {
    Write-Err "Dataset not found at $Dataset"
    Write-Err "Place Enron.csv at src\datasets\Enron.csv before running the experiment."
    exit 1
}

# --- 8. Smoke-test imports ---------------------------------------------------
Write-Info "Verifying Python imports..."
& $Python -c "import langgraph, langchain, langchain_ollama, ollama; import pandas, sklearn, numpy, tqdm; print('All imports OK')"
if ($LASTEXITCODE -ne 0) { Write-Err "Import check failed"; exit 1 }

# --- 9. Write run_experiment.ps1 wrapper -------------------------------------
$RunnerPath = Join-Path $ProjectDir 'run_experiment.ps1'
$runnerContent = @'
#Requires -Version 5.1
$ErrorActionPreference = 'Stop'
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectDir
& (Join-Path $ProjectDir '.venv\Scripts\python.exe') -m src.evaluation.run_experiment @args
'@
Set-Content -Path $RunnerPath -Value $runnerContent -Encoding UTF8
Write-Info "Created runner: $RunnerPath"

# --- Done --------------------------------------------------------------------
Write-Host ""
Write-Info "Setup complete. Run the experiment with:"
Write-Host ""
Write-Host "    .\run_experiment.ps1 --agent binary --batch-size 20"
Write-Host ""
