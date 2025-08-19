import subprocess
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    evaluate_py = repo_root / "evaluate.py"

    # Choose your model (task) and judge model for medical hallucinations
    model = "meta-llama/Llama-3.2-3B-Instruct"

    cmd = [
        sys.executable,
        str(evaluate_py),
        "--model",
        model,
        "--behavior",
        "hallu-med",
    ]

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
