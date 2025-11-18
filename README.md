# Install:

```bash
uv sync --all-extras
```
or 
```bash
pip install -e .
```

Then install playwright browsers:

```bash
playwright install
``` 

# Hosting model:

## downloading model:
We also released the model checkpoints as git-lfs files under model_checkpoints/

`git lfs install`
`git lfs pull`

## VLLM
1. If hosting a model whose weights you have downloaded locally on a GPU machine:
`python az_vllm.py --model_url /path/to/model_checkpoints/ --device_id 0,1` which will default to port 5000. We prefer to host across two devices depending on how many gpus you have and how much memory they have: `--device_id 0,1`. 

Then run `test_fara_agent.py` for an example of how to run the Fara agent. You will see a `client_config` which points to `"base_url": "http://localhost:5000/v1"` which is from above. 

```bash
python test_fara_agent.py --task "how many pages does wikipedia have" --start_page "https://www.bing.com" [--headful] [--downloads_folder "/path/to/downloads"] [--save_screenshots] [--max_rounds 100] [--browserbase]
```
If you set `--browserbase`, you need to export environment variables for the api key and project id. 

## Expected Output:
You will see it output 

```
[fara_agent] Wikipedia currently has approximately 64,394,387 pages.
<tool_call>
{"name": "computer_use", "arguments": {"action": "terminate", "status": "success"}}
</tool_call>

[fara_agent] Wikipedia currently has approximately 64,394,387 pages.
INFO:__main__:Closing browser...
```

# Benchmark Evaluation and Inference at Scale:

## installation of webeval package
```bash
conda create --name fara_webeval python=3.12
conda activate fara_webeval

# install fara package if you haven't already from fara/
pip install -e . 

# first install autogen submodule
git submodule update --init --recursive
cd autogen/python/packages
pip install -e autogen-core
pip install -e autogen-ext

# cd back up and install webeval
cd src/fara/webeval
pip install -e .

# install playwright if you haven't already
playwright install
```

Then launch e.g. webvoyager evaluation, `cd src/fara/webeval/scripts` and do:

```bash
python webvoyager.py --model_url ../../../../model_checkpoints/fara-7b/ --model_port 5000 --eval_oai_config ../endpoint_configs_gpt4o/dev/ --out_url /data/data/Fara/eval --device_id 0,1 --processes 1 --run_id 1 --max_rounds 100
```

Notes:

You can also set `--browserbase`, but again you need to export environment variables for the api key and project id. 
Be careful not to overload a single VLLM deployment with more than `--processes 10` or so requests concurrently because of weirdness like https://github.com/vllm-project/vllm/issues/19491

## Analyze Eval Run
Beneath the `--out_url` there will be an Eval folder like `/runs/WebSurfer-fara-100-max_n_images-3/fara-7b/<your_username>/WebVoyager_WebVoyager_data_08312025.jsonl/<run_id>` based on unique properties like `--run_id`, version of webvoyager data found in `webeval/data/webvoyager`, etc. 

After you ran the above `python webvoyager.py` command, you can enter that Eval folder into the `webeval/scripts/analyze_eval_results/analyze.ipynb` script and it will print out diagnostics of which tasks were aborted mid-trajectory and why, as well as the average score across non-aborted trajectories. 

A trajectory is aborted if and only if an error was raised during trajectory sampling. Trajectories that finished with a final terminate() call or those that exceeded the step budget are not considered aborted (though they may still receive a score of 0 and considered failures). Aborted trajectories are removed from this script's average computation. 

If you see lots of aborted trajectories, you should re-run the webvoyager.py script again (it will skip tasks that were not aborted, but only if you refer to the same run_id, username, etc). 

### Structure of Eval Folders:
The Eval folder is unique to the model, dataset, username who ran the command, and run_id. The folder contains `gpt_eval` and `traj` subdirectories. The latter contains another subdir for each task ID in webvoyager dataset, which itself contains 
- final_answer.json e.g. `Amazon--1_final_answer.json`. If you see `<no_answer>` it means either the trajectory was aborted, or the step budget was exceeded without terminate(). 
- `scores` containing the llm-as-a-judge score file
- `web_surfer.log` which contains a history of all the actions and errors 
- all the `screenshot_X.png` captured immediately before each action X. 