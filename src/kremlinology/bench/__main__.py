"""Run with python -m kremlinology.bench or kremlinology-bench."""

import argparse
import asyncio
import json
from pathlib import Path
import sys
from .config import load_config
from .runner import plan_benchmark, run_benchmark, verify_run


def main(argv=None):
    parser = argparse.ArgumentParser(prog="kremlinology-bench")
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="validate and print a plan; no API calls")
    plan.add_argument("config")
    run = commands.add_parser("run", help="execute only with --execute")
    run.add_argument("config")
    run.add_argument("--output", required=True)
    run.add_argument("--execute", action="store_true")
    report = commands.add_parser("report", help="read an existing summary")
    report.add_argument("run_directory")
    verify = commands.add_parser("verify", help="verify a run manifest")
    verify.add_argument("run_directory")
    args = parser.parse_args(argv)
    try:
        if args.command in ("plan", "run"):
            config = load_config(args.config)
            if args.command == "plan" or not args.execute:
                result = plan_benchmark(config)
            else:
                result = asyncio.run(run_benchmark(config, args.output))
        elif args.command == "report":
            result = json.loads((Path(args.run_directory) / "summary.json").read_text())
        else:
            result = verify_run(args.run_directory)
        print(json.dumps(result, indent=2, allow_nan=False))
        if args.command == "verify":
            return 0 if result["valid"] else 1
        if args.command == "run" and args.execute:
            return 0 if result["run"]["stop_reason"] == "completed" else 1
        return 0
    except (ValueError, OSError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
