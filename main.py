"""Entry point: ``python main.py --config ./exps/cifar.json``."""

import argparse
import logging
import os
import sys

from utils.config import load_config, validate_config


def setup_logging(save_dir, log_name):
    log_dir = os.path.join(save_dir, "exp")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "{}.log".format(log_name))

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.info("Experiment started, logging to %s", log_path)


def setup_parser():
    parser = argparse.ArgumentParser(
        description="G-FSCIL: learning to curate generated knowledge for class-incremental learning."
    )
    parser.add_argument("--config", dest="experiment_config", type=str,
                        default="./exps/cifar.json",
                        help="Path to the experiment JSON config.")
    parser.add_argument(
        "--check-config", action="store_true",
        help="Load and validate the experiment config, then exit without importing PyTorch.",
    )
    parser.add_argument(
        "--print-config", action="store_true",
        help="Print the fully resolved flat config before checking or training.",
    )
    return parser


def main():
    # Silences the HuggingFace tokenizer fork warning triggered by the LDM text encoder.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    cli = setup_parser().parse_args()
    args = load_config(cli.experiment_config)
    validate_config(args)

    if cli.print_config:
        import json
        print(json.dumps(args, indent=2, ensure_ascii=False, sort_keys=True))
    if cli.check_config:
        print("Config OK: {}".format(os.path.abspath(cli.experiment_config)))
        return

    setup_logging(args.get("save_dir", "outputs"), args.get("log_name", "debug_run"))
    # Keep heavy ML imports out of --check-config so first-time users can inspect
    # configs before installing the full diffusion environment.
    from trainer import train
    train(args)


if __name__ == "__main__":
    main()
