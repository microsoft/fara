Install:

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

see `test_fara_agent.py` for an example of how to run the Fara agent.

```bash
python test_fara_agent.py --task "Your task here" --start_page "https://www.example.com" [--headful] [--downloads_folder "/path/to/downloads"] [--save_screenshots] [--max_rounds 100]