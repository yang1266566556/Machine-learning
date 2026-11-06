$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$projectPython = Join-Path $projectRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) {
    throw 'Create .venv with standard Windows Python 3.12 first; see README.md.'
}
function Invoke-CheckedPython {
    & $projectPython @args
    if ($LASTEXITCODE -ne 0) { throw "Python command failed: $args" }
}
Invoke-CheckedPython -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
$lockFile = Join-Path $projectRoot 'requirements-lock.txt'
if (Test-Path -LiteralPath $lockFile) {
    Invoke-CheckedPython -m pip install -r $lockFile --extra-index-url https://download.pytorch.org/whl/cu128
} else {
    Invoke-CheckedPython -m pip install -r (Join-Path $projectRoot 'requirements.txt')
}
Invoke-CheckedPython -m pip install --no-deps --no-build-isolation -e $projectRoot
Invoke-CheckedPython -m pip check
Invoke-CheckedPython (Join-Path $PSScriptRoot 'prepare_models.py')
Invoke-CheckedPython (Join-Path $PSScriptRoot 'verify_environment.py')
