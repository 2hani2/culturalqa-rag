"""
main_with_moe.py
-----------------
Runs WITH trained conservative MoE re-scoring. Same CLI surface as main.py
(--ask, --questions_file, --dataset, --enable_images, interactive mode) --
this file pins force_moe_rescore=True. The default protects baseline evidence
at top 3 and 6 and allows MoE to reorder near ties. --legacy_moe reproduces
the historical cosine-only reranker. Image-only requests use an explicit
CLIP fallback unless --legacy_moe is specified.

If no trained checkpoint loads successfully, stop with an error.
The selected checkpoint and development policy are verified before use.
Random weights are never used for this mode.

Usage: identical to main.py, e.g.
  python main_with_moe.py
  python main_with_moe.py --ask "what is rangoli"
  python main_with_moe.py --questions_file questions.txt
  python main_with_moe.py --image_query "Find images of Kathakali"
"""

from main import run_cli

if __name__ == "__main__":
    run_cli(force_moe_rescore=True)
