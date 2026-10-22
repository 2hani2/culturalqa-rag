CulturalQA revised research paper

Open main.tex as the main document. Upload the entire folder to Overleaf,
including figures/, then compile main.tex. A compiled main.pdf is included.

This revises the user-supplied 10pt research-paper draft while preserving its
authors and article structure. Affiliation is omitted at the user's request.
The manuscript compares only Without MoE and unrestricted Cross-encoder + MoE.
The standalone CLIP image pathway is reported separately as a shared component.

Corrections include verified MRR/nDCG values and units, direct expert inputs and
ungated shared-expert flow, training/split provenance, query/candidate fusion and
score standardization, reader selection, RetVQA scope, and three results graphs.

Measurements are reused from the saved-candidate replay recorded on 15 October
2026. No model training, data relabelling, or fresh end-to-end evaluation was
performed for this document. data/ retains the verified underlying results and
provenance. The original project and previous paper are unchanged.

plot_figures.py regenerates PDF/PNG/SVG figures with matplotlib and numpy.
Compile with a normal LaTeX installation or Overleaf; no API key is needed.
