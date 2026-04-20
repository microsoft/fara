<div align="center">

# Fara-7B: An Efficient Agentic Model for Computer Use

<img src="figures/model_accuracy_vs_cost_v2_glm_cost_updated.png" alt="Fara-7B Performance" width="600"/>

[![Microsoft](https://img.shields.io/badge/Microsoft-Project-0078D4?logo=microsoft)](https://aka.ms/msaif/fara)
[![Hugging Face Model](https://img.shields.io/badge/🤗-Model-yellow)](https://huggingface.co/microsoft/Fara-7b)
[![Foundry](https://img.shields.io/badge/Azure-Foundry-0089D6)](https://aka.ms/foundry-fara-7b)
[![Dataset](https://img.shields.io/badge/🤗-WebTailBench%20Dataset-orange)](https://huggingface.co/datasets/microsoft/WebTailBench)
[![Dataset](https://img.shields.io/badge/🤗-CUAVerifierBench-orange)](https://huggingface.co/datasets/microsoft/CUAVerifierBench)
[![Paper](https://img.shields.io/badge/Paper-2511.19663-red)](https://arxiv.org/abs/2511.19663)

</div>

---

## Updates

* **2026-04-19** — Released **[CUAVerifierBench](https://huggingface.co/datasets/microsoft/CUAVerifierBench)**,
  a human-annotated benchmark for evaluating CUA verifiers (i.e. judges that
  score agent trajectories). Two splits — `fara7b_om2w_browserbase` (106
  Fara-7B Online-Mind2Web/Browserbase trajectories, ~2 reviewers each) and
  `internal` (154 trajectories from a heldout aurora-v2 task suite) —
  with per-judge UV-blind / UV-informed labels, Universal Verifier
  outputs, and legacy verifier outputs side-by-side. See the new
  [`cuaverifierbench/`](cuaverifierbench/) directory in this repo for
  the build script.
* **2026-04-18** — Removed the `autogen-core` / `autogen-ext` dependency
  from `webeval`; chat completion clients are now self-contained under
  `webeval/src/webeval/oai_clients/`. No more autogen submodule install
  step; just `pip install -e .[vllm]` then `cd webeval; pip install -e .`.
* **2026-04-18** — Incorporated WebTailBench (initial / now-stale
  version) directly into the repo as a first-class benchmark. The
  loader auto-downloads `WebTailBench-v1-rubrics.tsv` from
  [`microsoft/WebTailBench`](https://huggingface.co/datasets/microsoft/WebTailBench)
  and threads each task's published `precomputed_rubric` through to
  the verifier. Reproducibility CLI lives in `webeval/scripts/webtailbench.py`.
* **2026-04-18** — Released the **Universal Verifier** (`MMRubricAgent`)
  as the official verifier for WebTailBench. Multimodal,
  rubric-grounded, two-model ensemble (`gpt-5.2` + `o4-mini`) with
  per-criterion scoring, outcome verification, and first-point-of-failure
  analysis. A stand-alone parallel runner is at
  `webeval/scripts/verify_trajectories.py` for re-scoring any directory
  of webeval trajectories without touching the solver.

---

## Overview

**Fara-7B** is Microsoft's first **agentic small language model (SLM)** designed specifically for computer use. With only 7 billion parameters, Fara-7B is an ultra-compact Computer Use Agent (CUA) that achieves state-of-the-art performance within its size class and is competitive with larger, more resource-intensive agentic systems.

Try Fara-7B locally as follows (see [Installation](#Installation) for detailed instructions on Windows ) or via Magentic-UI:

```bash
# 1. Clone repository
git clone https://github.com/microsoft/fara.git
cd fara

# 2. Setup environment
python3 -m venv .venv 
source .venv/bin/activate
pip install -e .
playwright install
```

Then in one process, host the model:
```bash
vllm serve "microsoft/Fara-7B" --port 5000 --dtype auto 
```
Then you can iteratively query it with:
```bash
fara-cli --task "whats the weather in new york now"
```

To try Fara-7B inside Magentic-UI, please follow the instructions here [Magentic-UI + Fara-7B](https://github.com/microsoft/magentic-ui/blob/main/README.md#fara-7b). You will need to serve the model as before, but instead of fara-cli you can use Magentic-UI which has a nice UI (see video demos below).


Notes:
- If you're using Windows, we highly recommend using WSL2 (Windows Subsystem for Linux). Please see the Windows instructions in the [Installation](#Installation) section.
- You might need to do `--tensor-parallel-size 2` with vllm command if you run out of memory

<table>
<tr>
<td width="33%" align="center">

**Shopping**  

<video src="https://github.com/user-attachments/assets/d2109eba-a91f-4a0b-8217-38c1dcc17e9a" width="100%" style="max-height: 300px;">
</video>

</td>
<td width="33%" align="center">

**GitHub Issues**  

<video src="https://github.com/user-attachments/assets/bb177a09-8fcb-41be-8639-32044c1ec0e8" width="100%" style="max-height: 300px;">
</video>

</td>
<td width="33%" align="center">

**Directions with Cheese**  

<video src="https://github.com/user-attachments/assets/b83d341e-25f6-4236-a946-4b8eaca987d5" width="100%" style="max-height: 300px;">
</video>

</td>
</tr>
</table>

### What Makes Fara-7B Unique

Unlike traditional chat models that generate text-based responses, Fara-7B leverages computer interfaces—mouse and keyboard—to perform multi-step tasks on behalf of users. The model:

- **Operates visually** by perceiving webpages and taking actions like scrolling, typing, and clicking on directly predicted coordinates without accessibility trees or separate parsing models
- **Enables on-device deployment** due to its compact 7B parameter size, resulting in reduced latency and improved privacy as user data remains local
- **Completes tasks efficiently**, averaging only ~16 steps per task compared to ~41 for comparable models

Fara-7B is trained using a novel synthetic data generation pipeline built on the [Magentic-One](https://www.microsoft.com/en-us/research/articles/magentic-one-a-generalist-multi-agent-system-for-solving-complex-tasks/) multi-agent framework, with 145K trajectories covering diverse websites, task types, and difficulty levels. The model is based on [Qwen2.5-VL-7B](https://arxiv.org/abs/2502.13923) and trained with supervised fine-tuning.

### Key Capabilities

Fara-7B can automate everyday web tasks including:
- Searching for information and summarizing results
- Filling out forms and managing accounts
- Booking travel, movie tickets, and restaurant reservations
- Shopping and comparing prices across retailers
- Finding job postings and real estate listings

### Performance Highlights

Fara-7B achieves state-of-the-art results across multiple web agent benchmarks, outperforming both comparable-sized models and larger systems:

| Model | Params | WebVoyager | Online-M2W | DeepShop | WebTailBench |
|-------|--------|------------|------------|----------|--------------|
| **SoM Agents** | | | | | |
| SoM Agent (GPT-4o-0513) | - | 90.6 | 57.7 | 49.1 | 60.4 |
| SoM Agent (o3-mini) | - | 79.3 | 55.4 | 49.7 | 52.7 |
| SoM Agent (GPT-4o) | - | 65.1 | 34.6 | 16.0 | 30.8 |
| GLM-4.1V-9B-Thinking | 9B | 66.8 | 33.9 | 32.0 | 22.4 |
| **Computer Use Models** | | | | | |
| OpenAI computer-use-preview | - | 70.9 | 42.9 | 24.7 | 25.7 |
| UI-TARS-1.5-7B | 7B | 66.4 | 31.3 | 11.6 | 19.5 |
| **Fara-7B** | **7B** | **73.5** | **34.1** | **26.2** | **38.4** |

*Table: Online agent evaluation results showing success rates (%) across four web benchmarks. Results are averaged over 3 runs.*

### WebTailBench: A New Benchmark for Real-World Web Tasks

We are releasing **[WebTailBench](https://huggingface.co/datasets/microsoft/WebTailBench)**, a new evaluation benchmark focusing on 11 real-world task types that are underrepresented or missing in existing benchmarks. The benchmark includes 609 tasks across diverse categories, with the first 8 segments testing single skills or objectives (usually on a single website), and the remaining 3 evaluating more difficult multi-step or cross-site tasks.

#### WebTailBench Results (Process / Outcome)

UPDATE: we release revised WebTailBench numbers from the new Universal Verifier below, reporting both process- and outcome-based success rate. Process sucess is defined by whether the assessed rubric score is at least 80%. 

| Task Segment | Tasks | SoM GPT-5 | SoM o3 | SoM 4o | GLM-4.1V-9B | OAI Comp-Use | UI-TARS-1.5 | **Fara-7B** |
|----------------|-------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|
| **Single-Site Tasks** |
| Shopping | 56 | 64.8 / 38.9 | 60.0 / 36.4 | 37.5 / 17.9 | 27.8 / 9.3 | 44.6 / 16.1 | 53.6 / 30.4 | **58.9 / 39.3** |
| Flights | 51 | 51.1 / 27.7 | 23.4 / 8.5 | 3.9 / 2.0 | 6.1 / 0.0 | 17.6 / 7.8 | 5.9 / 2.0 | **44.0 / 30.0** |
| Hotels | 52 | 84.6 / 23.1 | 54.9 / 15.7 | 32.7 / 9.6 | 21.2 / 5.8 | 32.7 / 13.5 | 28.8 / 5.8 | **51.9 / 32.7** |
| Restaurants | 52 | 68.6 / 41.2 | 47.1 / 27.5 | 32.7 / 13.5 | 34.6 / 17.3 | 34.6 / 17.3 | 11.5 / 3.8 | **35.3 / 13.7** |
| Activities | 80 | 71.2 / 43.8 | 55.1 / 37.2 | 27.5 / 15.0 | 32.9 / 13.9 | 40.0 / 22.5 | 8.8 / 5.0 | **36.0 / 16.0** |
| Ticketing | 57 | 56.4 / 21.8 | 37.5 / 16.1 | 24.6 / 0.0 | 24.6 / 8.8 | 40.4 / 14.0 | 14.0 / 1.8 | **33.9 / 7.1** |
| Real-Estate | 48 | 58.7 / 43.5 | 19.1 / 17.0 | 8.3 / 4.2 | 21.3 / 12.8 | 13.3 / 6.7 | 6.2 / 0.0 | **32.6 / 15.2** |
| Jobs/Careers | 50 | 61.7 / 46.8 | 42.9 / 30.6 | 26.0 / 18.0 | 28.0 / 14.0 | 36.0 / 20.0 | 20.0 / 12.0 | **30.6 / 20.4** |
| **Multi-Step Tasks** |
| Shopping List (2 items) | 51 | 72.5 / 52.9 | 48.0 / 28.0 | 9.8 / 5.9 | 16.3 / 8.2 | 35.3 / 25.5 | 27.5 / 11.8 | **55.1 / 32.7** |
| Comparison Shopping | 57 | 73.7 / 59.6 | 75.4 / 57.9 | 35.1 / 17.5 | 28.3 / 9.4 | 3.5 / 1.8 | 12.3 / 1.8 | **54.4 / 28.1** |
| Compositional Tasks | 55 | 72.0 / 52.0 | 63.0 / 42.6 | 25.5 / 14.5 | 21.8 / 10.9 | 20.0 / 10.9 | 14.5 / 9.1 | **27.8 / 11.1** |
| **Overall** |
| Macro Average | 609 | 66.9 / 41.0 | 47.9 / 28.9 | 24.0 / 10.7 | 23.9 / 10.0 | 28.9 / 14.2 | 18.5 / 7.6 | **41.9 / 22.4** |
| Micro Average | 609 | 67.3 / 41.2 | 48.9 / 29.7 | 24.5 / 11.0 | 24.5 / 10.2 | 29.5 / 14.5 | 18.2 / 7.6 | **41.8 / 22.2** |

*Each cell shows **Process / Outcome**. Process = Rubric Score ≥ 0.8 threshold; Outcome = task completion success.*

### CUAVerifierBench: Evaluating the Verifiers Themselves

While WebTailBench measures *agents*, **[CUAVerifierBench](https://huggingface.co/datasets/microsoft/CUAVerifierBench)** measures the *judges that score those agents*. Each row pairs a Fara-7B agent trajectory (instruction, screenshots, web_surfer log, final answer) with one human reviewer's verdict, plus the verdicts produced by the **Universal Verifier (`MMRubricAgent`)** and several legacy verifiers — so researchers can compute verifier–human agreement (Cohen's κ, accuracy, F1) on a fixed corpus and iterate on new judge prompts / architectures against a frozen ground-truth set.

The dataset is exposed as two HuggingFace configs joinable on `task_id`:

| Config | Granularity | Contents |
|---|---|---|
| `trajectories` | one row per task | instruction, screenshots, web_surfer log, verifier outputs, task-level human aggregates |
| `annotations` | one row per (task, judge) | per-reviewer outcome / process labels and free-text justifications |

Two splits ship today:

| Split | Source | Trajectories | Annotation rows |
|---|---|---|---|
| `fara7b_om2w_browserbase` | Fara-7B trajectories on Online-Mind2Web tasks executed via Browserbase | 106 | 215 (≈2 reviewers/task; UV-blind **and** UV-informed stages) |
| `internal` | Heldout aurora-v2 task suite scored with the same WebSurfer + verifier stack | 154 | 154 (1 reviewer/task; UV-blind only) |

Reviewer identities are anonymized as `Judge1` … `JudgeN` using a single shared map across both splits. The build script that produced the dataset (with full schema + provenance) lives at [`cuaverifierbench/build_dataset.py`](cuaverifierbench/build_dataset.py); see the [dataset README](https://huggingface.co/datasets/microsoft/CUAVerifierBench/blob/main/README.md) for the full column list.

```python
from datasets import load_dataset

trajs = load_dataset("microsoft/CUAVerifierBench", "trajectories",
                     split="fara7b_om2w_browserbase")
anns  = load_dataset("microsoft/CUAVerifierBench", "annotations",
                     split="fara7b_om2w_browserbase")
```

### Evaluation Infrastructure

Our evaluation setup leverages:

1. **Playwright** - A cross-browser automation framework that replicates browser environments
2. **Abstract Web Agent Interface** - Allows integration of any model from any source into the evaluation environment
3. **Fara-Agent Class** - Reference implementation for running the Fara model

> **Note:** Fara-7B is an experimental release designed to invite hands-on exploration and feedback from the community. We recommend running it in a sandboxed environment, monitoring its execution, and avoiding sensitive data or high-risk domains.

---

# Installation


##  Linux 

The following instructions are for Linux systems, see the Windows section below for Windows instructions. 

Install the package using pip and set up the environment with Playwright:

```bash
# 1. Clone repository
git clone https://github.com/microsoft/fara.git
cd fara

# 2. Setup environment
python3 -m venv .venv 
source .venv/bin/activate
pip install -e .[vllm]
playwright install
```

Note: If you plan on hosting with Azure Foundry only, you can skip the `[vllm]` and just do `pip install -e .`


## Windows

For Windows, we highly recommend using WSL2 (Windows Subsystem for Linux) to provide a Linux-like environment. However, if you prefer to run natively on Windows, follow these steps:

```bash
# 1. Clone repository
git clone https://github.com/microsoft/fara.git
cd fara

# 2. Setup environment
python3 -m venv .venv
.venv\Scripts\activate
pip install -e .
python3 -m playwright install
```

## Hosting the Model

**Recommended:** The easiest way to get started is using Azure Foundry hosting, which requires no GPU hardware or model downloads. Alternatively, you can self-host with vLLM if you have GPU resources available.

### Azure Foundry Hosting (Recommended)

Deploy Fara-7B on [Azure Foundry](https://ai.azure.com/explore/models/Fara-7B/version/2/registry/azureml-msr) without needing to download weights or manage GPU infrastructure.

**Setup:**

1. Deploy the Fara-7B model on Azure Foundry and obtain your endpoint URL and API key

Then create a endpoint configuration JSON file (e.g., `azure_foundry_config.json`):

```json
{
    "model": "Fara-7B",
    "base_url": "https://your-endpoint.inference.ml.azure.com/",
    "api_key": "YOUR_API_KEY_HERE"
}
```

Then you can run Fara-7B using this endpoint configuration.

2. Run the Fara agent:

```bash
fara-cli --task "how many pages does wikipedia have" --endpoint_config azure_foundry_config.json [--headful]
```

Note: you can also specify the endpoint config with the args `--base_url [your_base_url] --api_key [your_api_key] --model [your_model_name]` instead of using a config JSON file. 

Note: If you see an error that the `fara-cli` command is not found, then try:

```bash
python -m fara.run_fara --task "what is the weather in new york now"
```

That's it! No GPU or model downloads required.

### Self-hosting with vLLM or LM Studio / Ollama

If you have access to GPU resources, you can self-host Fara-7B using vLLM. This requires a GPU machine with sufficient VRAM (e.g., 24GB or more).

Only on Linux: all that is required is to run the following command to start the vLLM server:

```bash
vllm serve "microsoft/Fara-7B" --port 5000 --dtype auto 
```
For quantized models or lower VRAM GPUs, please see [Fara-7B GGUF on HuggingFace](https://huggingface.co/bartowski/microsoft_Fara-7B-GGUF).

For Windows/Mac, vLLM is not natively supported. You can use WSL2 on Windows to run the above command or LM Studio / Ollama as described below.

Otherwise, you can use [LM Studio](https://lmstudio.ai/) or [Ollama](https://ollama.com/) to host the model locally. We currently recommend the following GGUF versions of our models [Fara-7B GGUF on HuggingFace](https://huggingface.co/bartowski/microsoft_Fara-7B-GGUF) for use with LM Studio or Ollama. Select the largest model that fits your GPU. Please ensure that context length is set to at least 15000 tokens and temperature to 0 for best results.

Then you can run Fara-7B pointing to your local  server:

Run the test script to see Fara in action:

```bash
fara-cli --task "what is the weather in new york now"
```

If you didn't use vLLM to host, please specify the correct `--base_url [your_base_url] --api_key [your_api_key] --model [your_model_name]`

If you see an error that the `fara-cli` command is not found, then try:

```bash
python -m fara.run_fara --task "what is the weather in new york now"
```

# Reproducibility

We provide a framework in `webeval/` to reproduce our results on WebVoyager and OnlineMind2Web. 
Agentic evaluations on live websites present unique challenges due to day-to-day changes. We implement several measures to ensure reliable and comparable evaluations:

**BrowserBase Integration**
We employ BrowserBase to manage browser session hosting, enabling reliable browser instance management.

**Time-sensitive Task Updates**
Tasks in benchmarks like WebVoyager can become stale or impossible. We:
- Removed ~48 impossible tasks from the original WebVoyager benchmark
- Updated ~50 tasks with future dates to keep them achievable
- Example: *"Search for a hotel in Bali from Jan 1 to Jan 4, 2024"* → *"Search for a hotel in Bali from Jan 1 to Jan 4, 2026"*
- Our updated WebVoyager benchmark is available at `webeval/data/webvoyager/WebVoyager_data_08312025.jsonl`

**Environment Error Handling**
Browser errors (connection drops, page timeouts) are handled robustly:
- Trajectories are retried up to 5 times when environment errors occur
- Complete yet incorrect trajectories are never retried
- Each retry starts with a fresh browser session, with no retained state

**Step Budget**
Each trajectory is capped at a maximum of 100 actions across all online benchmarks. Trajectories exceeding this budget without choosing to stop are considered incorrect.

## WebEval Package Installation

```bash
conda create --name fara_webeval python=3.12
conda activate fara_webeval

# Install fara package (with vllm extras for GPU hosting)
pip install -e .[vllm]

# Install webeval
cd webeval
pip install -e .

# Install playwright
playwright install
```

The webeval package no longer depends on `autogen-core` / `autogen-ext` —
all chat completion clients are vendored under `webeval/src/webeval/oai_clients/`
(see `GracefulRetryClient`, `OpenAIClientWrapper`, `AzureOpenAIClientWrapper`,
etc.). You no longer need to clone or install the autogen submodule.

> **Always activate the `fara_webeval` env before running any of the eval
> scripts below.** It pins `vllm==0.10.0` + `torch==2.7.1`; running with
> a newer `vllm` (≥ 0.19) under `torch` ≥ 2.10 trips a known
> `torch._dynamo.symbolic_convert` crash during CUDA-graph capture, which
> can be worked around with `--enforce_eager` but at a meaningful
> throughput cost. Stick to the pinned env.

## Running Evaluations

Navigate to the scripts directory:

```bash
cd webeval/scripts
```

Make sure you set a valid OpenAI GPT-4o endpoint in `endpoint_configs_gpt4o/dev` in order to run the WebVoyager LLM-as-a-judge! 

**Option 1: Self-hosted vLLM**

```bash
python webvoyager.py --model_url /path/where/you/want/to/download/model/ --model_port 5000 --eval_oai_config ../endpoint_configs_gpt4o/dev/ --out_url /path/to/save/eval/files --device_id 0,1 --processes 1 --run_id 1 --max_rounds 100
python om2w.py --model_url /path/where/you/want/to/download/model/ --model_port 5000 --eval_oai_config ../endpoint_configs_o4/dev/ --eval_model o4-mini --out_url /path/to/save/eval/files --device_id 0,1 --processes 1 --run_id 1 --max_rounds 100

# WebTailBench almost always needs --browserbase: a meaningful share of
# the benchmark's task websites (airlines, retailers, ticketing, …)
# block bot traffic from a vanilla playwright browser. Without
# --browserbase you'll see a high rate of trajectories that abort on
# Page.goto / navigation / captcha errors. Set BROWSERBASE_API_KEY and
# BROWSERBASE_PROJECT_ID in the environment first.
export BROWSERBASE_API_KEY=<your_browserbase_api_key>
export BROWSERBASE_PROJECT_ID=<your_browserbase_project_id>

# --success controls which Universal Verifier signal counts as the
# top-line score: ``outcome`` (default; binary outcome_success — what the
# Fara-7B numbers in the README above are reported against), ``process``
# (rubric_is_success := rubric_score >= --rubric_score_threshold; a more
# lenient gate, expect slightly higher numbers), or ``both``.
python webtailbench.py \
    --model_url /data/data/Fara/model_checkpoints \
    --model_port 5000 \
    --device_id 0,1 \
    --eval_oai_config ../../endpoint_configs/judge_active/prod \
    --judge_eval_model gpt-5.2 \
    --judge_o4_eval_model o4-mini \
    --rubric_score_threshold 0.8 \
    --success outcome \
    --out_url /data/data/Fara/eval \
    --processes 4 \
    --run_id 1 \
    --max_rounds 100 \
    --browserbase

python webtailbench.py \
    --model_url /data/data/Fara/model_checkpoints \
    --eval_oai_config ../../endpoint_configs/judge_active/prod \
    --judge_eval_model gpt-5.2 --judge_o4_eval_model o4-mini \
    --out_url /data/data/Fara/eval \
    --split flights --subsample 0.1 \
    --device_id 0,1 --processes 1 --run_id smoke --max_rounds 30 \
    --browserbase

python verify_trajectories.py \
    --input /data/data/Fara/eval/runs/.../<benchmark>/<run_id>/traj \
    --task-data ../data/om2w/Online_Mind2Web_06042025.json \
    --task-data-format om2w \
    --eval-config ../../endpoint_configs/judge_active/prod \
    --judge-model gpt-5.2 --o4mini-model o4-mini \
    --processes 8
```

**Option 2: Azure Foundry Deployment**

Deploy [Fara-7B on Foundry endpoint(s)](https://ai.azure.com/explore/models/Fara-7B/version/2/registry/azureml-msr), then place endpoint URLs and keys in JSONs under `endpoint_configs/`:

```bash
python webvoyager.py --model_endpoint ../../endpoint_configs/ --eval_oai_config ../endpoint_configs_gpt4o/dev/ --out_url /path/to/save/eval/files --processes 1 --run_id 1_endpoint --max_rounds 100
python om2w.py --model_endpoint ../../endpoint_configs/ --eval_oai_config ../endpoint_configs_o4/dev/ --eval_model o4-mini --out_url /path/to/save/eval/files --processes 1 --run_id 1_endpoint --max_rounds 100
python webtailbench.py --model_endpoint ../../endpoint_configs/ --eval_oai_config ../../endpoint_configs/judge_active/prod --judge_eval_model gpt-5.2 --judge_o4_eval_model o4-mini --out_url /data/data/Fara/eval --processes 1 --run_id 1_endpoint --max_rounds 100
```

### Notes


- We use the same LLM-as-a-judge prompts and model (GPT-4o) as WebVoyager, hence the `--eval_oai_config` argument
- Set `--browserbase` for browser session management (requires exported API key and project ID environment variables)
- Avoid overloading a single vLLM deployment with more than ~10 concurrent processes due to known issues
- See debugging output in `fara/webeval/scripts/stdout.txt`

---

## Analyzing Evaluation Results

### Evaluation Output Structure

Evaluation results are stored under `--out_url` in folders organized by:
- Model name
- Dataset
- Username
- Run ID

Example path:
```
/runs/WebSurfer-fara-100-max_n_images-3/fara-7b/<username>/WebVoyager_WebVoyager_data_08312025.jsonl/<run_id>
```

Each evaluation folder contains:
- `gpt_eval/` - LLM-as-a-judge evaluation results
- `traj/` - Per-task trajectory subdirectories containing:
  - `*-final_answer.json` (e.g., `Amazon--1_final_answer.json`) - `<no_answer>` indicates abortion or step budget exceeded
  - `scores/*_eval.json` - LLM judge scores (`gpt_eval.json` for WebVoyager, `WebJudge_Online_Mind2Web_eval-3.json` for Online-Mind2Web)
  - `web_surfer.log` - Action history and errors
  - `screenshot_X.png` - Screenshots captured before each action X
  - `times.json` - contains start and end times of task
  - `core.log` - contains high level logs like if the trajectory needs to be started or is already cached/done, the evaluation score, duration, and errors encountered

### Running Analysis

Use the analysis notebook to compute metrics:

```bash
cd webeval/scripts/analyze_eval_results/
jupyter notebook analyze.ipynb
```

The script:
- Identifies trajectories aborted mid-execution and diagnostic reasons
- Computes average scores across non-aborted trajectories
- Distinguishes between aborted trajectories (errors during sampling) and completed trajectories (with terminate() call or step budget exceeded)

To re-run failed tasks, execute the evaluation script again with the same `run_id` and `username` - it will skip non-aborted tasks.

<details>
<summary>Example WebVoyager GPT Eval Result</summary>

```json
{
  "score": 1.0,
  "gpt_response_text": "To evaluate the task, we need to verify if the criteria have been met:\n\n1. **Recipe Requirement**: A vegetarian lasagna recipe with zucchini and at least a four-star rating.\n\n2. **Search and Results**:\n   - The screenshots show that the search term used was \"vegetarian lasagna zucchini.\"\n   - Among the search results, \"Debbie's Vegetable Lasagna\" is prominently featured.\n   \n3. **Evaluation of the Recipe**:\n   - Rating: \"Debbie's Vegetable Lasagna\" has a rating of 4.7, which satisfies the requirement of being at least four stars.\n   - The presence of zucchini in the recipe is implied through the search conducted, though the screenshots do not explicitly show the ingredients list. However, the result response confirms the match to the criteria.\n\nGiven the information provided, the task seems to have fulfilled the requirement of finding a vegetarian lasagna recipe with zucchini and a four-star rating or higher. \n\n**Verdict: SUCCESS**"
}
```

</details>

<details>
<summary>Example of web_surfer.log (action history)</summary>

```json
{"message": "\nThought #1: To begin locating the closest GameStop store, I will click on the 'Trade-In' button, which typically leads to a store locator feature on retail websites.\nAction #1: executing tool 'left_click' with arguments {\"action\": \"left_click\", \"coordinate\": [1074, 37]}", "timestamp": "2025-11-25T22:23:42.788320", "type": "OtherEvent"}
{"message": "Observation#1: I clicked at coordinates (1083.0252100840337, 37.16517857142858).", "timestamp": "2025-11-25T22:23:47.616613", "type": "OtherEvent"}
{"message": "\nThought #2: Since access to Gamestop's website has been blocked, I will search for alternative sources of store location and hours for GameStop locations near zip code 90028. This could include cached or third-party website information.\nAction #2: executing tool 'web_search' with arguments {\"action\": \"web_search\", \"query\": \"GameStop store locator zip code 90028\"}", "timestamp": "2025-11-25T22:23:50.579702", "type": "OtherEvent"}
{"message": "Observation#2: I typed 'GameStop store locator zip code 90028' into the browser search bar.", "timestamp": "2025-11-25T22:23:55.119495", "type": "OtherEvent"}
{"message": "\nThought #3: The closest GameStop store to zip code 90028 is located at 5533 W Sunset Blvd, Los Angeles, CA 90028. It is currently closed and will open tomorrow at 11 AM. The contact number is (323) 466-3714.\nAction #3: executing tool 'terminate' with arguments {\"action\": \"terminate\", \"status\": \"success\"}", "timestamp": "2025-11-25T22:23:58.729038", "type": "OtherEvent"}
{"message": "Observation#3: The closest GameStop store to zip code 90028 is located at 5533 W Sunset Blvd, Los Angeles, CA 90028. It is currently closed and will open tomorrow at 11 AM. The contact number is (323) 466-3714.", "timestamp": "2025-11-25T22:24:02.379069", "type": "OtherEvent"}
```
</details>

## Citation

If you use Fara-7B in your research, please use the following BibTeX entry.
```bibtex
@article{fara7b2025,
  title={Fara-7B: An Efficient Agentic Model for Computer Use},
  author={Awadallah, Ahmed and Lara, Yash and Magazine, Raghav and Mozannar, Hussein and Nambi, Akshay and Pandya, Yash and Rajeswaran, Aravind and Rosset, Corby and Taymanov, Alexey and Vineet, Vibhav and Whitehead, Spencer and Zhao, Andrew},
  journal={arXiv:2511.19663},
  year={2025}
}
```
