import subprocess
import sys
import os
from pathlib import Path
import pytest

# Lista de subcomandos e argumentos mínimos para exercitar cada funcionalidade
test_cases = [
    ("analyze", ["--crypto", "BTC", "--save"]),
    ("simulate", ["--crypto", "BTC", "--model", "linear", "--save"]),
    ("train",    ["--crypto", "BTC", "--kfolds", "2"]),
    ("compare",  ["--crypto", "BTC"]),
    ("hypothesis", ["--threshold", "1.5", "--save"]),
    ("anova",      ["--alpha", "0.05", "--save"]),
    ("anova-groups", ["--metric", "volatility", "--alpha", "0.05", "--save"]),
]

@pytest.mark.parametrize("subcommand,args", test_cases)
def test_main_subcommands(subcommand, args):
    # Define o root do projeto para localizar main.py
    project_root = Path(__file__).parent.parent
    # Monta o comando de chamada com flag UTF-8
    cmd = [sys.executable, "-X", "utf8", str(project_root / "main.py"), subcommand] + args
    # Prepara ambiente com UTF-8
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    # Executa no cwd do projeto
    result = subprocess.run(
        cmd,
        cwd=str(project_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env
    )
    assert result.returncode == 0, (
        f"Subcomando '{subcommand}' falhou\n"
        f"STDOUT:\n{result.stdout.decode('utf-8', errors='ignore')}\n"
        f"STDERR:\n{result.stderr.decode('utf-8', errors='ignore')}"
    )

if __name__ == "__main__":
    pytest.main()