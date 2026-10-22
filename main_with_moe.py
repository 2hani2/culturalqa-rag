"""Run unrestricted cross-encoder + MoE ranking: Z(CE) + 0.25 * Z(MoE).

Uses the temperature-0.1 checkpoint selected on development data. There are no
protected evidence bands. Standalone image retrieval is shared CLIP retrieval.
"""
from main import run_cli

if __name__ == '__main__':
    run_cli(force_moe_rescore=True)
