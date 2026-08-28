# Event Extraction Pipeline

Scripts for reproducing the event extraction experiments (see the repository root `README.md` for setup and usage).

- `Extraction.py` — calls the Dify workflow API and extracts activity labels for every message in `data/labeled/`; the run configuration (backbone and output folder) is set in the `__main__` block
- `eval_result.py` — computes per-activity MCC, balanced accuracy, F1, precision, and recall from the `pred_<activity>` columns of the prediction files
- `compare_eval.py` — the same evaluation for cross-configuration comparison directories
- `ablation_diagram.py` — generates the ablation figure
- `compare_diagram.py` — generates the configuration-comparison figure
