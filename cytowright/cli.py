"""CytoWright model development, parameter correction and response prediction."""
from pathlib import Path
import argparse
import json
import sys

TASKS = ("bbbc036", "bbbc047")
CONFIG = "configs/model_development.json"


def _provider_options(parser):
    parser.add_argument("--model", help="model-service identifier")
    parser.add_argument("--base-url", help="Chat Completions API base URL")
    parser.add_argument("--api-key-env", help="environment variable containing the API key")
    parser.add_argument("--provider", default="default", help="model-panel entry")
    parser.add_argument("--model-panel", help="model-panel JSON file")
    parser.add_argument("--local-endpoints", help="local endpoint configuration")
    parser.add_argument("--stream", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--reasoning-effort", help="model-service reasoning setting")


def _backend(args):
    from .paths import project_path
    from .provider import resolve_campaign_provider
    config = json.loads(project_path(CONFIG).read_text())
    return resolve_campaign_provider(config["provider"], model=args.model, base_url=args.base_url,
        api_key_env=args.api_key_env, stream=args.stream, reasoning_effort=args.reasoning_effort,
        panel_key=args.provider, model_panel_path=args.model_panel, local_endpoints=args.local_endpoints)


def _discover(args):
    from .bounded_discovery import run_bounded_discovery
    from .workflow import TASKS as registry
    seed = args.seed or {"bbbc036": 2026080401, "bbbc047": 2026180401}[args.task]
    return run_bounded_discovery(task_id=registry[args.task], output_root=args.output, mode="formal",
        device=args.device, config_path=CONFIG, trajectory_seed=seed, provider_parameters=_backend(args))


def _campaign(args):
    from .paths import project_path
    root = project_path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    start = args.seed or {"bbbc036": 2026080401, "bbbc047": 2026180401}[args.task]
    completed = []
    for index in range(args.searches):
        entry = argparse.Namespace(**vars(args))
        entry.seed = start+index
        entry.output = str(root/f"search_{index+1:02d}")
        result = _discover(entry)
        completed.append({"seed": entry.seed, "selected_candidate": result["selected_candidate_id"], "output": entry.output})
    report = {"status": "complete", "task": args.task, "searches": completed}
    (root/"summary.json").write_text(json.dumps(report, indent=2)+"\n")
    return report


def _validate(args):
    from .joint_response_runtime import load_runtime_config, load_starting_candidate
    from .tasks import get_task
    from .workflow import TASKS as registry
    config = load_runtime_config("configs/joint_response_runtime.json")
    candidate = load_starting_candidate("h0", config)
    tasks = {}
    for alias, task_id in registry.items():
        task = get_task(task_id)
        tasks[alias] = {"source_tables": {key: Path(value).is_file() for key, value in task["sources"].items()}}
    return {"status": "configured", "initial_model": candidate.candidate_hash, "tasks": tasks}


def _model_arguments(parser):
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--normalizer", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)


def parser():
    root = argparse.ArgumentParser(prog="cytowright", description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("demo", help="run the synthetic CPU feedback example", add_help=False)
    commands.add_parser("correct", help="fit and save a parameter correction", add_help=False)
    validation = commands.add_parser("validate", help="inspect task and model configuration")
    validation.set_defaults(handler=_validate)
    prepare = commands.add_parser("prepare-data", help="prepare measurement tables and named NPZ subsets")
    prepare.add_argument("--task", choices=TASKS, required=True)
    prepare.add_argument("--output", type=Path, help="array output directory")
    for name in ["discover", "campaign"]:
        search = commands.add_parser(name, help="develop a predictor" if name=="discover" else "run repeated development searches")
        search.add_argument("--task", choices=TASKS, required=True)
        search.add_argument("--output", "--output-root", dest="output", required=True)
        search.add_argument("--device", default="cuda:0")
        search.add_argument("--seed", type=int)
        if name=="campaign":
            search.add_argument("--searches", type=int, default=10)
        _provider_options(search)
        search.set_defaults(handler=_discover if name=="discover" else _campaign)
    train = commands.add_parser("train", help="fit a predictor program")
    train.add_argument("--task", choices=TASKS, required=True)
    train.add_argument("--source", type=Path, required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--seed", type=int, default=41)
    train.add_argument("--device", default="cuda:0")
    train.add_argument("--epochs", type=int, help="training epoch budget")
    prediction = commands.add_parser("predict", help="generate response predictions")
    _model_arguments(prediction)
    analysis = commands.add_parser("analyze", help="measure compound use with matched input replacement")
    _model_arguments(analysis)
    analysis.add_argument("--maps", type=int, default=32)
    analysis.add_argument("--seed", type=int, default=41)
    return root


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] in {"correct", "demo"}:
        if arguments[0]=="correct":
            from .feedback.editing import main as command
        else:
            from .demo import main as command
        return command(arguments[1:]) or 0
    args = parser().parse_args(arguments)
    if args.command=="prepare-data":
        from .workflow import prepare_data
        result = prepare_data(args.task, args.output)
    elif args.command=="train":
        from .workflow import train_program
        result = train_program(args.task, args.source, args.output, seed=args.seed, device=args.device, epochs=args.epochs)
    elif args.command in {"predict", "analyze"}:
        from .inference import analyze_file, predict_file
        shared = (args.source, args.checkpoint, args.normalizer, args.data, args.output)
        result = predict_file(*shared) if args.command=="predict" else analyze_file(*shared, maps=args.maps, seed=args.seed)
    else:
        result = args.handler(args)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
