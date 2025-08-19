import subprocess
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    evaluate_py = repo_root / "evaluate.py"

    # Choose your model and BBQ bias type
    model = "meta-llama/Llama-3.2-3B-Instruct"
    bias_type = "gender"  # one of: gender, race, nationality, physical, age, religion

    cmd = [
        sys.executable,
        str(evaluate_py),
        "--model",
        model,
        "--behavior",
        f"bias:{bias_type}",
    ]

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
