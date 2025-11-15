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

# Inference at Scale:

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
```

Then launch e.g. webvoyager evaluation, `cd src/fara/webeval/scripts` and do:

```bash
python webvoyager.py --model_url ../../../../model_checkpoints/fara-7b/ --model_port 5000 --eval_oai_config ../endpoint_configs_gpt4o/dev/ --out_url /data/data/Fara/eval --device_id 0,1 --processes 1 --run_id 1 --max_rounds 100
```

Be careful not to overload a single VLLM deployment with more than `--processes 10` or so requests concurrently because of weirdness like https://github.com/vllm-project/vllm/issues/19491