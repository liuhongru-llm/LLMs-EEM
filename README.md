# LLMs-EEM: Event Log Extraction from Unstructured Enterprise Messaging Using LLMs

This repository contains the code, workflow definitions, annotated datasets, and raw predictions for reproducing the experiments reported in:

> **Event Log Extraction from Unstructured Enterprise Messaging Using LLMs**

LLMs-EEM extracts business-level events (activity labels) from unstructured customer-support conversations and turns them into event logs. Each inbound message is classified against a company-specific activity set by a Dify-orchestrated LLM workflow that combines four components:

1. **Trigger-word constraint** — a curated trigger-word base associated with each activity;
2. **RAG case retrieval** — three annotated exemplar cases retrieved per message as in-context demonstrations;
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
    │   ├── keywords-template/ # trigger-word base (one sheet per dataset)
    │   └── case_base/         # retrieval case base (180 annotated cases with rationales,
    │                           # 6 files: 3 companies x inbound/outbound; load into Dify
    │                           # as the workflow's knowledge dataset)
    ├── WorkFlow/              # LLMs-EEM (full) predictions, per backbone
    ├── WorkFlow_10round/      # Sept 2026 re-verification run with voting caches
    ├── WorkFlow_no_RAG/       # ablation: without RAG case retrieval
    ├── WorkFlow_no_self/      # ablation: without self-consistency (single extraction pass)
    ├── WorkFlow_no_val/       # ablation: without the verification pass
    ├── Single_LLM/            # zero-shot single-LLM baseline predictions
    └── *_Evaluation_Results/  # evaluation outputs (color-scaled Excel files)
```

The `WorkFlow*/<backbone>/` folders contain both the shipped predictions (`twcs-<company>-300-predicted-<model>.xlsx`, one `pred_<activity>` column per activity) and the per-message response caches (`nlp_cache_*.json`). The shipped predictions reproduce the results reported in the paper without re-invoking any API. `WorkFlow_10round/DeepSeekReasoner/` additionally ships per-message voting caches (`vote_cache_<company>_300.json`): the ten per-round 0/1 votes and the verification verdict for every message–activity pair (see [Reproduction verification](#reproduction-verification-september-2026)).

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
3. **Bind the case base.** Create a Dify knowledge dataset from `event_extraction/data/case_base/*.txt` (one dataset per file, or one dataset per company) and re-bind the workflow's knowledge-retrieval node to it, since imported DSLs reference the original dataset IDs.
4. **Configure model providers.** In Dify settings, add credentials for the model plugins used by the workflow (DeepSeek and SiliconFlow; the Gemini workflow uses the corresponding Gemini provider).
5. **Publish the workflow**, then open the app's *API Access* page and create an API key.

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

## Reproduction verification (September 2026)

`event_extraction/WorkFlow_10round/DeepSeekReasoner/` contains an end-to-end re-run of the main configuration (10–13 September 2026) verifying the released artifact: the workflow, prompts, ten-round voting rule (an activity is accepted at ≥ 8 of 10 votes), and verification stage were executed unchanged. The run additionally records, for every message–activity pair, the ten per-round votes and the verification verdict (`vote_cache_<company>_300.json`), enabling offline sensitivity analysis of the voting threshold and the number of sampling rounds without re-invoking any API.

DeepSeek decommissioned the `deepseek-reasoner` API identifier on 24 July 2026 (it maps to `deepseek-v4-flash` in thinking mode, which the shipped DSL therefore uses), so the re-run used the successor model:

| Dataset | Paper (20 Mar 2026, `deepseek-reasoner`) | Re-run (Sept 2026, `deepseek-v4-flash`) |
|---|---|---|
| AmazonHelp | 0.911 | 0.914 |
| AppleSupport | 0.949 | 0.905 |
| SpotifyCares | 0.935 | 0.763 |

(macro-averaged per-activity F1, same metric as the paper). The predominantly action-oriented datasets reproduce the reported results. The gap on SpotifyCares is driven by precision (macro precision 0.934 → 0.681): the successor model takes a more lenient stance on the topic-type activities that dominate this activity set (e.g., Song/Artist, App, Premium). **All results reported in the paper are those of the original runs of 20 March 2026 with `deepseek-reasoner`.**

### Offline sensitivity analysis (from the voting caches)

All figures below are recomputed offline from `vote_cache_*.json` (no API calls) and therefore reflect the **September 2026 re-run model (`deepseek-v4-flash`)**, not the paper's original `deepseek-reasoner` runs. They are provided as a consistency check on the design choices, not as a re-statement of the paper's results.

**Voting threshold.** With ten sampling rounds fixed, the acceptance threshold t is swept from 5 to 10 (pure voting, verification stage disabled). The deployed setting t = 8 sits at or near the macro-F1 peak for all three datasets (0.921 / 0.911 / 0.851); t = 10 (unanimity) degrades all datasets sharply, and t ≤ 7 lowers SpotifyCares by 0.3–3.1 points.

| threshold | AmazonHelp | AppleSupport | SpotifyCares |
|---|---|---|---|
| 5 | 0.922 | 0.906 | 0.820 |
| 6 | 0.924 | 0.906 | 0.844 |
| 7 | 0.924 | 0.909 | 0.848 |
| **8 (deployed)** | **0.921** | **0.911** | **0.851** |
| 9 | 0.922 | 0.907 | 0.816 |
| 10 | 0.860 | 0.883 | 0.755 |

**Number of sampling rounds.** With the threshold set to ⌈0.8k⌉ for k rounds, macro F1 averaged over all C(10, k) subsets of the ten recorded votes: k = 5 already reaches 0.922 / 0.908 / 0.838, within 0.013 of the full k = 10 (0.921 / 0.911 / 0.851). The additional rounds mainly buy stability against near-threshold flips rather than average accuracy.

**Fraction of activities below the threshold.** Of the 16,897 message–activity pairs, 89.2% receive fewer than 8 votes and enter the verification stage; however, 86.1% of all pairs are zero-vote clear negatives, so the genuinely ambiguous zone (1–7 votes) is only 3.1% of pairs. The verification acceptance rate rises monotonically with the vote count — 0.2% at 0 votes, 28.0% at 1–4 votes, 63.6% at 5–7 votes — consistent with the confidence-partition design.

**SpotifyCares gap attribution.** Pure voting at t = 8 yields 0.851 on SpotifyCares, but the deployed pipeline (voting + verification) yields 0.763: under the successor model, the verification stage admits 184 additional positives on this dataset (24 zero-vote overrides and 160 borderline passes), which manifest as false positives on topic-type activities. The reproduction gap therefore stems from the successor model's verification behavior, not from the voting rule itself.

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
