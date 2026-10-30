"""Send the general-LLM table-prompting tasks to a model through OpenRouter.

Each task in `data/llm_table_probe/tasks_*.json` is one prompt (60 labelled support molecules,
40 query molecules). This sends it with the settings used for the paper: the task's repeat number
as `seed`, a strict JSON schema for the answer, a fixed per-model output cap, and one pinned
provider without fallbacks. Each raw response is saved to --out-dir (an existing response there
is never re-requested). The default, `results/llm_responses/`, already holds the paper's responses,
so a fresh run needs a new directory; score it with `scripts/score_llm_responses.py --responses`.

Requires an OpenRouter key in the environment variable OPENROUTER_API_KEY. Requests are paid.

    python scripts/query_llm.py --model openai/gpt-6-astra --repeats 0-4 --out-dir results/my_responses --dry-run
"""

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

import click

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fit_head import parse_repeats  # noqa: E402

API = "https://openrouter.ai/api/v1/chat/completions"
TASKS = sorted((ROOT / "data/llm_table_probe").glob("tasks_*.json"))
RESPONSES = ROOT / "results/llm_responses"
# model -> (pinned provider, max output tokens)
MODELS = {
    "openai/gpt-6-astra": ("openai", 24000),
    "anthropic/claude-opus-5.5": ("anthropic", 64000),
    "google/gemini-3.5-flash": ("google-vertex", 65536),
    "google/gemini-4-argon": ("google-vertex", 65536),
}
SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "predictions",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"predictions": {"type": "array", "items": {
                "type": "object",
                "properties": {"id": {"type": "string"}, "value": {"type": "number"}},
                "required": ["id", "value"], "additionalProperties": False}}},
            "required": ["predictions"], "additionalProperties": False,
        },
    },
}


def response_path(model, task, out_dir=RESPONSES):
    name = "__".join([model.replace("/", "__"), task["dataset"], task["target"], str(task["seed"]), task["mode"]])
    return Path(out_dir) / f"{name}.json"


def post(body, key):
    req = urllib.request.Request(API, data=json.dumps(body).encode(), headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.load(r)


@click.command()
@click.option("--model", type=click.Choice(sorted(MODELS)), required=True)
@click.option("--repeats", default="0-14", show_default=True)
@click.option("--dataset", multiple=True, help="Restrict to these datasets (repeatable).")
@click.option("--mode", type=click.Choice(["normal", "shuffled"]), multiple=True)
@click.option("--out-dir", type=click.Path(file_okay=False), default=str(RESPONSES), show_default=True)
@click.option("--dry-run", is_flag=True, help="List what would be sent; send nothing.")
def main(model, repeats, dataset, mode, out_dir, dry_run):
    provider, max_tokens = MODELS[model]
    wanted = set(parse_repeats(repeats))
    tasks = [t for p in TASKS for t in json.loads(p.read_text())
             if t["seed"] in wanted and (not dataset or t["dataset"] in dataset) and (not mode or t["mode"] in mode)]
    todo = [t for t in tasks if not response_path(model, t, out_dir).exists()]
    print(f"{len(tasks)} tasks, {len(tasks) - len(todo)} already answered, {len(todo)} to send")
    if dry_run or not todo:
        return
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise click.ClickException("set OPENROUTER_API_KEY")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for task in todo:
        body = {"model": model, "messages": task["messages"], "seed": task["seed"], "response_format": SCHEMA,
                "max_tokens": max_tokens, "usage": {"include": True},
                "provider": {"only": [provider], "allow_fallbacks": False}}
        try:
            response = post(body, key)
        except (urllib.error.URLError, TimeoutError) as exc:
            # Not retried: a failed request may still have been charged.
            print(f"{task['id']}: request failed ({type(exc).__name__}); not retried", flush=True)
            continue
        response_path(model, task, out_dir).write_text(json.dumps(response) + "\n")
        print(f"{task['id']}: saved", flush=True)


if __name__ == "__main__":
    main()
