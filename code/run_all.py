"""
run_all.py - Runs the complete credit-card-default pipeline end to end.

Executes, in order:
  1. task1_eda.py            - EDA (pandas/numpy version)
  2. task1_eda_pytorch.py    - EDA (PyTorch tensor version)
  3. task2_prepare.py        - feature preparation
  4. task3_train.py          - trains all 6 classifiers
  5. task3b_leakage_demo.py  - target-leakage demonstration
  6. task4_evaluate.py       - evaluation, tuning, SHAP

Just run:  python run_all.py
...from anywhere - it works no matter which machine or folder the project
has been copied to. It doesn't rely on the current working directory: every
task script resolves its own paths from its own file location, and this
runner only needs to know where it itself lives.

If a task fails, its error is printed and the run stops - re-run once the
issue is fixed; earlier tasks' outputs are already saved and don't need to
be redone.
"""
import runpy
import sys
import time
from pathlib import Path

# Directory this script lives in - the same "code/" folder as every task
# script, wherever the project has been placed on disk.
CODE_DIR = Path(__file__).resolve().parent

TASKS = [
    "data_prep",            # sanity-checks the raw CSV / cleaning report
    "task1_eda",
    "task1_eda_pytorch",
    "task2_prepare",
    "task3_train",
    "task3b_leakage_demo",
    "task4_evaluate",
]


def main():
    # Make sure Python can import the task modules (data_prep, task2_prepare,
    # etc.) regardless of where run_all.py was launched from.
    if str(CODE_DIR) not in sys.path:
        sys.path.insert(0, str(CODE_DIR))

    print(f"Project code directory: {CODE_DIR}")
    print(f"Running {len(TASKS)} steps...\n")

    for i, task in enumerate(TASKS, start=1):
        script_path = CODE_DIR / f"{task}.py"
        if not script_path.exists():
            print(f"[{i}/{len(TASKS)}] SKIPPED - {script_path.name} not found")
            continue

        print(f"[{i}/{len(TASKS)}] Running {script_path.name} ...")
        t0 = time.time()
        try:
            # run_name="__main__" makes each script's `if __name__ ==
            # "__main__":` block execute, exactly as if it had been run
            # directly with `python <script>.py`.
            runpy.run_path(str(script_path), run_name="__main__")
        except Exception:
            print(f"\n--- Pipeline stopped: {script_path.name} raised an error ---")
            raise
        print(f"[{i}/{len(TASKS)}] Done in {time.time() - t0:.1f}s\n")

    print("Pipeline complete. Figures in ../figures and ../figures_pytorch, "
          "results in ../outputs and ../outputs_pytorch.")


if __name__ == "__main__":
    main()
