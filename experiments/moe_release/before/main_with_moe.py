"""
main_with_moe.py
-----------------
Runs WITH trained MoE re-scoring. Same CLI surface as main.py
(--ask, --questions_file, --dataset, --enable_images, interactive mode) --
this file adds nothing except pinning force_moe_rescore=True.

If no trained checkpoint loads successfully, stop with an error.
Train first with train_moe.py. Random weights are never used for this mode.

Usage: identical to main.py, e.g.
  python main_with_moe.py
  python main_with_moe.py --ask "what is rangoli"
  python main_with_moe.py --questions_file questions.txt
  python main_with_moe.py --image_query "Find images of Kathakali"
"""

from main import run_cli

if __name__ == "__main__":
    run_cli(force_moe_rescore=True)
