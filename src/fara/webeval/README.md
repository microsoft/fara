# WebEval Benchmarks

WebEval is a framework for evaluating web agent systems. This README provides both installation instructions and detailed evaluation procedures.

## Installation & Dependencies
Install the following packages in order:

```bash
conda create --name fara_webeval python=3.12
conda activate fara_webeval

# first install autogen submodule
cd ../../../autogen/python/packages
pip install -e autogen-core
pip install -e autogen-ext

# install webeval
cd src/fara/webeval
pip install -e .


# Install qwen-agent with additional features
pip install qwen-agent[gui,rag,code_interpreter,mcp]==0.0.29
# Install flash-attn for performance (ninja required)
pip install flash-attn>=2.5.8 --no-build-isolation 
playwright install
```

Finally, make sure to install [blobfuse2](https://learn.microsoft.com/en-us/azure/storage/blobs/blobfuse2-how-to-deploy?tabs=Ubuntu) to work with azure storage.

## Important Installation Notes
- You may need to run `git submodule update --init --recursive` from the `webeval/` directory if mind2web or webvoyager files are missing. You also may need to run it from the top level agento/ folder if you get an error like `No such file or directory: '/data/corbyrosset/code/agento/autogen/python/packages/autogen-core'`
- Ensure `tokenizer.json` and associated tokenizer files are in the same directory as your model folder
- **Make sure you have vllm >= 0.10.XX**, we encountered reproducibility issues with 0.9.XX
- For `float32` inference, you may encounter vllm errors. Solutions include:
  - Prepend `VLLM_ATTENTION_BACKEND=XFORMERS` to your command
  - Use `--enforce_eager` flag
  - Sometimes both flags are needed

## Quick Start - Evaluating a Trained Model

You can start with `run_eval.sh` and modify the `model_url` and `web_surfer_model_type` arguments. For webvoyager, keep `max_rounds=30`. For heldout test sets, there is a corresponding script for each segment e.g. `run_eval_hotels_head.sh`. Do not change the `max_rounds` arguments in those files so as to keep evaluations comparable. You can also use `run_all_eval.sh` to run your desired evaluations in sequence on gpus. 
For click-to-run evaluation scripts, see the `webeval/scripts` directory.

### Basic Evaluation Run for a Trained Student Model
```bash
python webvoyager.py --model_url YOUR_MODEL_FOLDER_AZURE_BLOB_URI


```
or modify the `webeval/scripts/run_all_eval.sh` script to choose which models you want to run on which datasets. 

### Full Configuration Example
```bash
python -u webvoyager.py \
    --model_url <your_path> \
    --eval_model "gpt-4o" \
    --eval_oai_config "</path/to/agento/endpoint_configs_gpt4o/dev/>" \
    --web_surfer_model_type <most likely "orca_qwen25vl_aurorav2_solver_history"> \
    --include_input_text_key_args \
    --processes 15 \
    --run_id <your-run-id> \
    --web_surfer_kwargs max_n_images=3 model_call_timeout=300 \
    --max_rounds 30
```
Core Arguments:

- `--model_url`: AzureStorageBlob URI of the model (SAS optional if logged in with `az login`) e.g. "https://aifrontiersplus.blob.core.windows.net/osagent/models/p0-cua-corby.compositionalv6_clueweb100k_keypress_hover_actions_and_filtered_is_in_loop_Sept27_histrepsimple3im.ebs128.lr5e-06.ep2.wr0.10.ngpu32"
- `--web_surfer_model_type`: 
  - `orca_qwen25vl_aurorav2_solver` (for older models with Qwen2.5-VL)
  - `orca_qwen25vl_aurorav2_solver_history` (for models with new history representation)
- `--out_url`: (don't change this) Output location (default: `aifrontiersplus/osagent/eval/v01`)
- `--run_id`: Unique identifier for your run (default: 0)
- `--eval_data_url`: for `holdout.py` evaluation, you must specify which test set you want to evaluate on. Any azure folder with subdirectories of valid `task_data.json` files with `task_summary` fields will qualify. 
- `--eval_only`:  will only run the evaluation script without obtaining the output from the agents, useful for getting numbers of a previous task

Performance & Execution Arguments:
- `--processes`: Number of parallel processes (max recommended: 15)
- `--max_rounds`: Maximum actions per trajectory (default: 30)
- `--enforce_eager`: Helps with `--dtype float32` compatibility
- `--browserbase`: Use browserbase session manager (recommended for shopping/flights or other domains that actively block web bots)

Model & Evaluation Arguments:
- `--include_input_text_key_args`: **Required** - enables `delete_existing_text` and `press_enter` fields
- `--remove-keypress-hover-sleep`: Remove key, mouse_move, and wait actions from computer use tool (restricts agent to type, left_click, scroll, visit_url, web_search, history_back, pause_and_memorize_fact, terminate). Use this for any model trained on data preprocessed before `corby-v7-keypress-hover-sleep-and-filter_is_in_loop` prepro version.
- `--web_surfer_kwargs`: Additional kwargs (e.g., `max_n_images=3` which specifies how many of the most recent screenshots the model should see, **required** only for "history" models)
- `--eval_model`: Judge model choice (`gpt-4o`, `o4-mini`, `o3-mini`, `o3`, `*`)
- `--eval_oai_config`: path to the config files matching `--eval_model` argument.

### Avoiding Az Login every 8-12 hours:

1) try to do `az login --identity`. If that doesn't work, try steps 2 and 3 below and make sure to do step 4. 
2) get client_id from either
    - if using azure vm with name like `GCRAZGDL1212`
    ```bash
    curl -s -H "Metadata:true" "http://169.254.169.254/metadata/identity/oauth2/token?api-version=2018-02-01&resource=https://management.azure.com" | jq '{client_id}'
    ```
    - Or, if using Arc Managed machine like `GCRSANDBOX286`
    ```bash
    az ad sp list --display-name GCRSANDBOX286 --query '[0].appId' -o tsv
    ```
3) login with that identity: `az login --identity --client-id $result_from_curl`
4) if using openai endpoints, point code to use `agento/endpoint_configs_XXX/prod` instead of `/dev` as now your machine is logged in as managed identity, which is what the /prod configs point to


## Evaluating a GPT Baseline Evaluation
To evaluate a baseline api e.g. GPT-4o, o4-mini or other OpenAI models use the following:

```bash
python webvoyager.py --web_surfer_model_type gpt_solver --web_surfer_client_cfg /agento/agento/endpoint_configs_gpt4o/dev --eval_oai_config /agento/agento/endpoint_configs_gpt4o/dev --max_rounds 10 --processes 10 --gpt_solver_model_name gpt-4o --eval_model gpt-4o
```
or you can use this all-in-one script: `webeval/scripts/run_gpt_solver_eval.sh` and point to what you want to be the solver and what should be the evaluator. 

# Custom WebVoyager tasks

If you want to use your tasks and evaluate our agent using the WebVoyager evaluator please see the custom_webtasks.py script. It is similar to webvoyager.py but it uses a different benchmark class that loads tasks from a csv file. The csv file should have two columns: `task` (string) and `set`(string). You can pass in the csv file using the `--csv_path` argument either as a local path or an azure blob url.


The default settings in custom_webtasks.py evaluate with tasks people asked in the demo of our agent.

# Heldout Test Sets
For any `holdout.py` evaluation, the standard options for the `--eval_data_url` are as follows:

| Test Set | Tasks | Description | URL |
|----------|-------|-------------|-----|
| **Shopping Head** | 56 | Tasks across ~40 major retailers (Walmart, Amazon, Target, etc.) | `https://aifrontiersplus.blob.core.windows.net/osagent/data/aurora_segment_heldout_test_sets_v2/shopping_head/` |
| **Shopping Lists Tail** | 51 | 2-item purchases from smaller retailers | `https://aifrontiersplus.blob.core.windows.net/osagent/data/aurora_segment_heldout_test_sets_v2/shopping_lists_tail/` |
| **Flights** | 51 | Tasks across ~35 airlines (United, Delta, WestJet, etc.) | `https://aifrontiersplus.blob.core.windows.net/osagent/data/aurora_segment_heldout_test_sets_v2/flights/` |
| **Hotels Head** | 52 | Tasks across ~180 hotel sites and aggregators (marriott, orbitz, etc.) | `https://aifrontiersplus.blob.core.windows.net/osagent/data/aurora_segment_heldout_test_sets_v2/hotels_head/` |
| **Restaurants Tail** | 52 | Tasks related to restaurants: reserve, order takeout, analyze menu from **non-head domains** | `https://aifrontiersplus.blob.core.windows.net/osagent/data/aurora_segment_heldout_test_sets_v2/restaurants_tail/` |
| **Things to Do** | 80 | Tasks related to planning activities, events, visiting attractions | `https://aifrontiersplus.blob.core.windows.net/osagent/data/aurora_segment_heldout_test_sets_v2/things_to_do/` |
| **Recipe to Shopping** (not used) | 48 | find a recipe for something and then buy all/some ingredients for it online | `https://aifrontiersplus.blob.core.windows.net/osagent/data/aurora_segment_heldout_test_sets_v2/recipe_to_shopping/` | 
| **Price Comparison** | 57 | compare prices/features of two products between two head-ish retailers | `https://aifrontiersplus.blob.core.windows.net/osagent/data/aurora_segment_heldout_test_sets_v2/price_comparison/` |
| **Ticketing** | 57 | buy tickets to e.g. movies, music events, etc | `https://aifrontiersplus.blob.core.windows.net/osagent/data/aurora_segment_heldout_test_sets_v2/ticketing/` |
| **Compositional Tasks V2** | 87 | compositional tasks involving multiple websites | `https://aifrontiersplus.blob.core.windows.net/osagent/data/aurora_segment_heldout_test_sets_v2/compositional_tasks_v2/` |
| **Real Estate** | 48 | real estate related tasks | `https://aifrontiersplus.blob.core.windows.net/osagent/data/aurora_segment_heldout_test_sets_v2/realestate_complex/` |


# Advanced Usage: Splitting VLLM and Evaluation
WARNING: we advise against running evaluation this way because we noticed strange quality gap over just running the one-step evaluation process. 

For multiple quick evaluation runs, use the two-step process:

## Step 1: Start VLLM Server
```bash
python az_vllm.py --model_url YOUR_MODEL_FOLDER_AZURE_BLOB_URL --device_id 0 --dtype bfloat16 --enforce_eager 2>&1 | tee az_vllm_output.log
```
- `--port`: Server port (default: 5000)
- `--vllm_port`: VLLM backend port (default: 5001)
- `--device_id`: GPU devices (comma-separated for tensor parallelism)
- `--dtype`: Data type (`bfloat16` default, `float32` for stability but requires more memory)
- `--enforce_eager`: Helps with float32 compatibility

You can also use `--cache` to cache model weights locally for faster startup. By default, we cache to `~/.cache/vllm_models` but you can use `--cache_dir` to specify a different location.


## Step 2: Run Evaluation
potentially specify the vllm_port and device_id which hosts the model 

```bash
python webvoyager.py --model_port 5004 --device_id 2,3
```

# Time-Sensitive Dataset Updates

The `update_time_sensitive_eval_datasets.py` script automatically updates stale dates in flight, hotel, and restaurant evaluation datasets. It uses dataset-specific logic: for restaurants, dates like "August 25" are considered stale if they've passed this year, while for flights/hotels, dates without years are treated as next occurrence. Only tasks with reservation intent are processed for restaurants.

Example usage:
```bash
# Update all time-sensitive datasets
python webeval/scripts/update_time_sensitive_eval_datasets.py --local-dir /path/to/data

# Update specific dataset with dry run
python webeval/scripts/update_time_sensitive_eval_datasets.py --local-dir /path/to/data --dataset restaurants --dry-run
```

Example output shows the script correctly updating stale dates while preserving relative terms like "next available":
- **Flights**: Updated 0 files (dates like "September 1" treated as future)
- **Hotels**: Updated past dates with explicit years like `'08/28/2025 - 09/05/2025' -> 'from December 29 through December 31'`
- **Restaurants**: Updated past month/day dates like `'for August 25 at 8:45 PM' -> 'for the soonest available time'`

# Post-Evaluation Error Analysis

Analyze evaluation errors using:
- `browser/post_hoc_error_analysis.ipynb` for detailed trajectory analysis
- `eval_exp.py` with utilities in `webeval/src/webeval/post_eval_analysis.py` for automated logging

# Common Error Types
Every run on azure now logs the frequency of each error type encountered, if any. It will look like:

| Error Type | Frequency | Percentage | Description |
|------------|-----------|------------|-------------|
| Timeout Error | 62 | 5.4% | Playwright timeout (60s exceeded) |
| Target Closed Error | 43 | 3.7% | Browser page/target closed unexpectedly |
| Reference Error | 20 | 1.7% | JavaScript evaluation issues |

# Instruction Following Benchmarks

All the instruction following benchmarks share the same system implementation: `WebSurferInstructionFollowingSystem` in `src/webeval/system/web_surfer_instruction_following_system.py`:

- Uses Qwen2.5VL with CUA prompt, takes as input (optional) image + text instruction, does 1 tool call. We then parse the tool call and return it.

The model is evaluated on whether it ouputs a CUA action (tool that is not terminate or memorize fact) and the string response.

## AgentHarm Benchmark

AgentHarm evaluates web agents on harmful and benign tasks to test refusal capabilities. Based on the [AgentHarm paper](https://arxiv.org/abs/2410.09024)

Minimum script to run:
```bash
python scripts/agent_harm.py \
    --eval_oai_config /home/hmozannar/agento/agento/endpoint_configs_bigger/dev
```

### Dataset Input
The benchmark loads from JSON files containing behavior data:
- `benign_behaviors_test_public.json` / `benign_behaviors_validation.json`: slightly safe tasks with tool calling
- `harmful_behaviors_test_public.json` / `harmful_behaviors_validation.json`: Harmful requests   with tool calling
- `chat_public_test.json` / `chat_validation.json`: Chat-specific harmful requests without tool calling

Each behavior includes:
- `prompt`: The task instruction
- `target_functions`: Tools that might be called

### Evaluation Metrics
- **Action Accuracy**: Whether agent correctly identifies if action is needed (No action for all tasks)
- **Refusal Rate**: Percentage of tasks where agent refused to act (target is 100%)

### Dataset Splits
- `all`: All examples
- `benign`: Safe behaviors only  
- `harmful`: Harmful behaviors only
- `chat`: Chat-specific examples


# Instruction Following Benchmark

The Instruction Following benchmark evaluates web agents on their ability to correctly classify whether a task requires action and provide accurate answers. It supports both local CSV files and Azure-hosted datasets with images.

## Running Instruction Following Evaluation

```bash
# Using local CSV file
python scripts/instruction_following.py \
    --eval_oai_config /home/hmozannar/agento/agento/endpoint_configs_bigger/dev \
    --csv_path /path/to/your/data.csv \

# Using Azure blob folder (default)
python scripts/instruction_following.py \
    --eval_oai_config /home/hmozannar/agento/agento/endpoint_configs_bigger/dev \
    --data_az_folder https://aifrontiersplus.blob.core.windows.net/osagent/eval/custom_instruction_following/hussein_0/ \
```

### Dataset Input
The benchmark expects CSV data with the following columns:
- `prompt`: The task instruction (required)
- `requires_action`: Boolean indicating if action is needed (required)
- `image_path`: Optional path to associated image
- `ground_truth`: Optional ground truth answer for evaluation
- `to_refuse`: Optional boolean for refusal tasks

For Azure mode, expects in the folder:
- `data.csv`: Main data file
- `images/`: Directory containing referenced images

### Output and Evaluation Metrics
- **Action Accuracy**: Whether agent correctly classifies if action is needed
- **Answer Accuracy**: LLM judge evaluation of response vs ground truth
- **Refusal Rate**: Percentage of tasks where agent refused to act
- **Refusal Accuracy**: Whether agent correctly refused when expected

# VisualWebBench WebQA Benchmark

VisualWebBench WebQA evaluates web agents on visual question-answering tasks using website screenshots. Based on the [VisualWebBench paper](https://arxiv.org/abs/2404.05955), it tests agents' ability to understand and answer questions about web page content. The visualwebbench paper has many others splits, we only take the webqa split here.

## Running VisualWebBench WebQA Evaluation

```bash
python scripts/visualwebbench_webqa.py \
    --eval_oai_config /home/hmozannar/agento/agento/endpoint_configs_bigger/dev
```

### Dataset Input
The benchmark loads from parquet files (`test-00000-of-00001.parquet`) containing:
- `id`: Unique example identifier
- `task_type`: Task type (should be "webqa")
- `website`: Source website name
- `question`: Question text about the image
- `answer`: List of possible correct answers
- `image`: Website screenshot data

Images are automatically extracted and saved to local `images/` directory during loading.

### Output and Evaluation Metrics
- **Action Accuracy**: Whether agent correctly identifies no action is needed (QA task)
- **GPT Judge Score**: LLM evaluation of answer quality vs ground truth
- **F1 Score**: Token-level overlap between candidate and best possible answer
- **Answer Inclusion**: Whether candidate answer contains any acceptable answer

### Dataset Splits
- `all`: All examples across websites



# Extending WebEval

## Implementing a New Benchmark

1. Create a folder in `src/webeval/benchmarks/` with your benchmark name
2. Implement the benchmark class:

```python
from ...benchmark import Benchmark

class HiBenchmark(Benchmark):
    def __init__(self, data_dir: str):
        super().__init__(name="Hi", data_dir=data_dir)

    def download_dataset(self):
        pass

    def load_dataset(self):
        pass

    def get_split_examples(self, split: str) -> list:
        pass

    def evaluator(self, task_data: dict, candidate: dict) -> float:
        pass
```

3. Add import to `src/webeval/benchmarks/__init__.py`

## Implementing a New System

```python
from webeval.system import BaseSystem

class HiSystem(BaseSystem):
    def __init__(self, system_name: str):
        super().__init__(system_name)

    def get_answer(self, task_id: str, task_data: Dict[str, Any], output_dir: str) -> Any:
        pass

    def load_answer_from_disk(self, task_id: str, output_dir: str) -> Any:
        pass

    def save_answer_to_disk(self, task_id: str, answer: str, output_dir: str) -> None:
        pass
```

# Performance Notes & Troubleshooting

### Common Issues
- **Aborted Requests**: Increase model call timeout to 30+ seconds, reduce `num_processes` to ≤10, or shard model across multiple GPUs
- **Memory Issues**: Use `dtype=bfloat16` instead of `float32` when possible
- **Silent Failures**: Monitor logs for "Aborted request" messages and check trajectory lengths

### Optimal Settings
- Model call timeout: 30+ seconds with retries
- Processes: ≤10 for stability
- GPU sharding: ≥2 GPUs recommended
- Use the data browser (`browser/debug.ipynb`) to analyze failed trajectories

### Debugging Failed Trajectories
Check for trajectories that end before `max_rounds` without a `stop/terminate` action:
```python
failed_trajs = ex_list.select(ex_list['last_action'] != 'terminate')
failed_trajs['num_actions'].hist()
```

All evaluation results are logged in the standard AzureML workspace under the `osagent_eval` experiment.
