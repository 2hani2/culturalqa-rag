CULTURALQA GRAPH OUTPUTS - verified 17 October 2026

Eight figures: 300-DPI PNG, editable vector SVG, and one combined vector PDF.

The figures compare fresh baseline and conservative MoE against historical original MoE.
Conservative MoE preserves baseline top-3/top-6 evidence. Image mode uses CLIP fallback.
Matching baseline quality is not evidence of expert-mixing superiority.

FIGURE CAPTIONS

1. 01_text_quality
Text top-3 quality. Fresh conservative MoE matches the baseline by preserving its evidence. Original MoE values are historical on the same 100 questions.

2. 02_text_retrieval
Text Hit, MRR and nDCG at top 3 and top 6. Relevance labels contain verified positives but are not exhaustive. Metrics shown on the 0-1 scale.

3. 03_dataset_breakdown
Text results grouped by labelled source dataset. Every question searches all five datasets. There are 20 questions per dataset, with only 15 independent Indian Cultural Dataset source groups.

4. 04_modality_breakdown
80 text-only and 20 image-associated text questions. These categories describe the labelled source record. They are distinct from the 20 image-only retrieval tasks.

5. 05_image_retrieval
Image metrics on 20 queries. Repaired mode uses unchanged CLIP order, not a successful image-MoE reranker. Its equality with baseline follows from fallback.

6. 06_processing_time
Fresh-run mean processing components at top 3. Shared retrieval and per-context reader costs are reused across conditions. These sums exclude startup and are not independent end-to-end latency measurements.

7. 07_reranking_effects
Changes relative to the baseline on 100 text questions. Conservative MoE protects evidence membership while allowing bounded near-tie reordering. A changed answer is not necessarily a wrong answer.

8. 08_example_output
Actual saved answer outputs for the beach question, rendered as a comparison card. Original MoE retrieved DOSA#2 but a distractor produced a more confident wrong answer. The repair retains baseline evidence and returns the baseline answer.

LIMITATIONS
- Reused, previously inspected benchmark; not a blind independent test.
- Labels source-checked by the assistant, not independently reviewed by humans.
- Conservative answer parity follows from evidence protection; no superiority claim.
- Image mode is CLIP fallback. Simple-fusion alpha-zero fallback omitted from main plots because it duplicates baseline.
- Original MoE is historical; baseline quality matched between historical and fresh runs.
- Figures are descriptive point estimates; no statistical significance claim.

SOURCES
{
  "fresh": {
    "path": "validation_conservative_live_120/validation_report.json",
    "sha256": "5b17a1b59d57a4093353fb108db63094de6c834b101d3e4a97e3de286044be72",
    "created_at": "2026-10-16T07:40:15.160461+00:00"
  },
  "historical": {
    "path": "validation_large_120/validation_report.json",
    "sha256": "cb97f47afa222fc2d232bacf5cc0d110f779f3bc0b409813be91fa7df34b8389",
    "created_at": "2026-10-14T14:04:33.569004+00:00"
  }
}

REPRODUCE
Python 3.12 with matplotlib==3.11.2 and numpy==2.5.4:
python make_figures.py
The two source reports must be present in the parent project folder.
No models, API calls, training or report changes are performed by this script.
