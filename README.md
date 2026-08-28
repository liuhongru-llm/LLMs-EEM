# LLMs-EEM: Event Log Extraction from Unstructured Enterprise Messaging Using LLMs

This repository contains the code, workflow definitions, annotated datasets, and raw predictions for reproducing the experiments reported in:

> **Event Log Extraction from Unstructured Enterprise Messaging Using LLMs**

LLMs-EEM extracts business-level events (activity labels) from unstructured customer-support conversations and turns them into event logs. Each inbound message is classified against a company-specific activity set by a Dify-orchestrated LLM workflow that combines four components:

1. **Trigger-word constraint** — a curated trigger-word base associated with each activity;
2. **RAG case retrieval** — four annotated exemplar cases retrieved per message as in-context demonstrations;
3. **Multi-path self-consistency** — ten parallel extraction passes, aggregated by a high-confidence voting rule (an activity is accepted when it receives at least 80% of the votes);
4. **Verification pass** — a final LLM call that re-examines borderline labels before they enter the event log.

Two LLM backbones are evaluated: **DeepSeek-Reasoner** (`deepseek-reasoner`) and **Gemini 3 Pro**.

## Repository structure

```
LLMs-EEM/
├── README.md
├── dify_workflows/                              # Dify workflow DSL files (import via Dify)
│   ├── llms-eem-workflow-deepseek-reasoner.yml  # full LLMs-EEM workflow (DeepSeek-Reasoner backbone)
│   ├── llms-eem-workflow-gemini-3-pro.yml       # full LLMs-EEM workflow (Gemini 3 Pro backbone)
│   └── coarse-grained-time-computation.yml      # coarse-grained temporal annotation workflow
└── event_extraction/
    ├── Extraction.py          # event extraction client (calls the Dify workflow API)
    ├── eval_result.py         # per-activity metrics: MCC / balanced accuracy / F1 / precision / recall
    ├── compare_eval.py        # evaluation for cross-configuration comparison directories
    ├── ablation_diagram.py    # ablation figure
    ├── compare_diagram.py     # configuration-comparison figure
    ├── environment.yml        # conda environment (Python 3.10)
    ├── data/
    │   ├── labeled/           # annotated datasets (300 inbound messages x 3 companies)
    │   └── keywords-template/ # trigger-word base (one sheet per dataset)
    ├── WorkFlow/              # LLMs-EEM (full) predictions, per backbone
    ├── WorkFlow_no_RAG/       # ablation: without RAG case retrieval
    ├── WorkFlow_no_self/      # ablation: without self-consistency (single extraction pass)
    ├── WorkFlow_no_val/       # ablation: without the verification pass
    ├── Single_LLM/            # zero-shot single-LLM baseline predictions
    └── *_Evaluation_Results/  # evaluation outputs (color-scaled Excel files)
```

The `WorkFlow*/<backbone>/` folders contain both the shipped predictions (`twcs-<company>-300-predicted-<model>.xlsx`, one `pred_<activity>` column per activity) and the per-message response caches (`nlp_cache_*.json`). The shipped predictions reproduce the results reported in the paper without re-invoking any API.

## Requirements

Python 3.10 with the pinned conda environment:

```bash
conda env create -f event_extraction/environment.yml
conda activate langAgents
```

Main dependencies: `pandas`, `numpy`, `scikit-learn`, `openpyxl`, `xlsxwriter`, `matplotlib`, `seaborn`, `requests`.

## Deploying the extraction workflow (Dify)

Event extraction is orchestrated by [Dify](https://dify.ai). Set it up once:

1. **Deploy Dify.** Either self-host it via Docker Compose ([installation guide](https://docs.dify.ai/)) or use Dify Cloud. The default endpoint in `Extraction.py` assumes a self-hosted instance at `http://localhost` — change `ENDPOINT` if yours differs.
2. **Import the workflow.** In Dify, create a new app from DSL and upload `dify_workflows/llms-eem-workflow-deepseek-reasoner.yml` (or the Gemini 3 Pro variant).
3. **Configure model providers.** In Dify settings, add credentials for the model plugins used by the workflow (DeepSeek and SiliconFlow; the Gemini workflow uses the corresponding Gemini provider).
4. **Publish the workflow**, then open the app's *API Access* page and create an API key.

The additional DSL `coarse-grained-time-computation.yml` is the workflow used for the coarse-grained temporal annotations of the event log.

## Running the experiments

### 0. Configure the API key

The Dify application key is read from the environment variable `DIFY_API_KEY` (never hard-code it):

```powershell
# Windows PowerShell
$env:DIFY_API_KEY = "app-xxxxxxxx"
```

```bash
# Linux / macOS
export DIFY_API_KEY="app-xxxxxxxx"
```

### 1. Event extraction

```bash
cd event_extraction
python Extraction.py
```

Select the run configuration in the `__main__` block at the bottom of `Extraction.py`:

```python
model = 'DeepSeekReasoner'                        # or 'Gemini3Pro'
output_base_dir = os.path.join('WorkFlow', model) # output folder
```

The script reads every `data/labeled/twcs-<company>-300.xlsx`, calls the workflow for each message, and writes `twcs-<company>-300-predicted-<model>.xlsx` into the configured output folder. An `nlp_cache_<company>_300.json` file is maintained per run so interrupted executions resume instead of re-billing completed messages.

**Ablations.** The ablation configurations are obtained by importing the full workflow, removing the corresponding nodes (the knowledge-retrieval branch for `w/o RAG`, nine of the ten parallel extraction paths for `w/o SC`, the verification call for `w/o Val`), and publishing each variant as a separate Dify app. Point `DIFY_API_KEY` at the variant you want to run and set `output_base_dir` accordingly (`WorkFlow_no_RAG`, `WorkFlow_no_self`, `WorkFlow_no_val`, or `Single_LLM` for the zero-shot single-LLM baseline).

### 2. Evaluation

```bash
python eval_result.py
```

Set `input_root` / `output_root` in the `__main__` block (e.g. `'WorkFlow'` / `'WorkFlow_Evaluation_Results'`). For every prediction file, the script computes per-activity MCC, balanced accuracy, F1, precision, and recall from the `pred_<activity>` columns and writes color-scaled Excel reports. `compare_eval.py` provides the same evaluation for cross-configuration comparison directories.

### 3. Figures

```bash
python ablation_diagram.py
python compare_diagram.py
```

## Data

- `data/labeled/` — 300 annotated inbound customer-support messages per company (AmazonHelp, AppleSupport, SpotifyCares) with binary ground-truth labels for each activity of the company's activity set.
- `data/keywords-template/` — the trigger-word base: one sheet per dataset, pairing each activity with its trigger words.

## Citation

If you use this code or data, please cite the paper:

```bibtex
@article{llmseem2026,
  title   = {Event Log Extraction from Unstructured Enterprise Messaging Using LLMs},
  author  = {Zhang, Rui and others},
  journal = {(to appear)},
  year    = {2026}
}
```

## License

This project is released under the MIT License.
