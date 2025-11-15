import argparse
from .core import download_benchmark_func, run_benchmark_func, evaluate_benchmark_func, run_evaluate_benchmark_func


# ----------------------------------------------------------------------
# CLI Subcommand: download_benchmark
# ----------------------------------------------------------------------
def download_benchmark(args):
    download_benchmark_func(args.benchmark_name)


# ----------------------------------------------------------------------
# CLI Subcommand: run_benchmark
# ----------------------------------------------------------------------
def run_benchmark(args):
    run_benchmark_func(
        args.benchmark_name, args.system_name, args.parallel, args.benchmark_dir, args.runs_dir, args.split, args.run_id
    )


# ----------------------------------------------------------------------
# CLI Subcommand: evaluate_benchmark
# ----------------------------------------------------------------------
def evaluate_benchmark(args):
    evaluate_benchmark_func(
        args.benchmark_name, args.system_name, args.benchmark_dir, args.runs_dir, args.split, args.run_id
    )


# ----------------------------------------------------------------------
# CLI Subcommand: run_evaluate_benchmark
# ----------------------------------------------------------------------
def run_evaluate_benchmark(args):
    run_evaluate_benchmark_func(
        args.benchmark_name, args.system_name, args.parallel, args.benchmark_dir, args.runs_dir, args.split, args.run_id
    )


# ----------------------------------------------------------------------
# Main CLI
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Benchmark CLI")
    subparsers = parser.add_subparsers(dest="command", help="Sub-command help")
    subparsers.required = True

    # download_benchmark
    download_parser = subparsers.add_parser("download_benchmark", help="Download dataset files.")
    download_parser.add_argument("--benchmark_name", type=str, required=True)
    download_parser.set_defaults(func=download_benchmark)

    # run_benchmark
    run_parser = subparsers.add_parser("run_benchmark", help="Run system on benchmark.")
    run_parser.add_argument("--benchmark_name", type=str, required=True)
    run_parser.add_argument("--system_name", type=str, required=True)
    run_parser.add_argument("--parallel", type=int, default=2, help="Number of processes to use.")
    run_parser.add_argument("--benchmark_dir", type=str, required=True)
    run_parser.add_argument("--runs_dir", type=str, required=True)
    run_parser.add_argument("--split", type=str, default="dev")
    run_parser.add_argument("--run_id", type=int, default=0)
    run_parser.set_defaults(func=run_benchmark)

    # evaluate_benchmark
    eval_parser = subparsers.add_parser("evaluate_benchmark", help="Evaluate results.")
    eval_parser.add_argument("--benchmark_name", type=str, required=True)
    eval_parser.add_argument("--system_name", type=str, required=True)
    eval_parser.add_argument("--benchmark_dir", type=str, required=True)
    eval_parser.add_argument("--runs_dir", type=str, required=True)
    eval_parser.add_argument("--split", type=str, default="dev")
    eval_parser.add_argument("--run_id", type=int, default=0)
    eval_parser.set_defaults(func=evaluate_benchmark)

    # run_evaluate_benchmark
    run_eval_parser = subparsers.add_parser("run_evaluate_benchmark", help="Run and evaluate system on benchmark.")
    run_eval_parser.add_argument("--benchmark_name", type=str, required=True)
    run_eval_parser.add_argument("--system_name", type=str, required=True)
    run_eval_parser.add_argument("--parallel", type=int, default=2, help="Number of processes to use.")
    run_eval_parser.add_argument("--benchmark_dir", type=str, required=True)
    run_eval_parser.add_argument("--runs_dir", type=str, required=True)
    run_eval_parser.add_argument("--split", type=str, default="dev")
    run_eval_parser.add_argument("--run_id", type=int, default=0)
    run_eval_parser.set_defaults(func=run_evaluate_benchmark)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
