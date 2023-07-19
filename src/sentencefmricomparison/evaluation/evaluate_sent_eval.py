"""Evaluate sentence embedding models with the SentEval benchmark.

This script is based on the SimCSE repository: https://github.com/princeton-nlp/SimCSE/blob/main/evaluation.py
"""

import argparse
import logging
import os
import sys

import senteval
import torch
from prettytable import PrettyTable

from sentencefmricomparison.constants import (
    PATH_TO_SENTEVAL,
    PATH_TO_SENTEVAL_DATA,
    SENTEVAL_OUTPUT_DIR,
)
from sentencefmricomparison.models.sentence_embedding_base import SentenceEmbeddingModel

# Set up logger
logging.basicConfig(format="%(asctime)s : %(message)s", level=logging.DEBUG)

# Import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)


def print_table(
    task_names,
    scores,
) -> PrettyTable:
    """Print a table of the SentEval results."""
    tb = PrettyTable()
    tb.field_names = task_names
    tb.add_row(scores)
    print(tb)
    return tb


def main():
    """Run the SentEval benchmark."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Transformers' model name or path",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=SENTEVAL_OUTPUT_DIR,
        help="Output directory to save the senteval benchmark results to",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["dev", "test", "fasttest"],
        default="fasttest",
        help="What evaluation mode to use (dev: fast mode, dev results; test: full mode, test results); fasttest: "
        "fast mode, test results",
    )
    parser.add_argument(
        "--task_set",
        type=str,
        choices=["sts", "transfer", "full", "na"],
        default="full",
        help="What set of tasks to evaluate on. If not 'na', this will override '--tasks'",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=[
            "STS12",
            "STS13",
            "STS14",
            "STS15",
            "STS16",
            "MR",
            "CR",
            "MPQA",
            "SUBJ",
            "SST2",
            "TREC",
            "MRPC",
            "SICKRelatedness",
            "STSBenchmark",
        ],
        help="Tasks to evaluate on. If '--task_set' is specified, this will be overridden",
    )

    args = parser.parse_args()

    # Load transformers' model checkpoint
    sent_model = SentenceEmbeddingModel(args.model_name_or_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    try:
        sent_model.model = sent_model.model.to(device)
    except:  # noqa
        pass

    # Set up the tasks
    if args.task_set == "sts":
        args.tasks = [
            "STS12",
            "STS13",
            "STS14",
            "STS15",
            "STS16",
            "STSBenchmark",
            "SICKRelatedness",
        ]
    elif args.task_set == "transfer":
        args.tasks = ["MR", "CR", "MPQA", "SUBJ", "SST2", "TREC", "MRPC"]
    elif args.task_set == "full":
        args.tasks = [
            "STS12",
            "STS13",
            "STS14",
            "STS15",
            "STS16",
            "STSBenchmark",
            "SICKRelatedness",
        ]
        args.tasks += ["MR", "CR", "MPQA", "SUBJ", "SST2", "TREC", "MRPC"]

    # Set params for SentEval
    if args.mode == "dev" or args.mode == "fasttest":
        # Fast mode
        params = {
            "task_path": PATH_TO_SENTEVAL_DATA,
            "usepytorch": True,
            "kfold": 5,
            "classifier": {
                "nhid": 0,
                "optim": "rmsprop",
                "batch_size": 128,
                "tenacity": 3,
                "epoch_size": 2,
            },
        }
    elif args.mode == "test":
        # Full mode
        params = {
            "task_path": PATH_TO_SENTEVAL_DATA,
            "usepytorch": True,
            "kfold": 10,
            "classifier": {
                "nhid": 0,
                "optim": "adam",
                "batch_size": 64,
                "tenacity": 5,
                "epoch_size": 4,
            },
        }
    else:
        raise NotImplementedError

    # SentEval prepare and batcher
    def prepare(params, samples):  # noqa
        return

    def batcher(params, batch, max_length=None):  # noqa
        # Handle rare token encoding issues in the dataset
        if len(batch) >= 1 and len(batch[0]) >= 1 and isinstance(batch[0][0], bytes):
            batch = [[word.decode("utf-8") for word in s] for s in batch]

        sentences = [" ".join(s) for s in batch]

        # Get raw embeddings
        with torch.no_grad():
            # Use the sentence embedding model encode sentences function to get pooled sentence embeddings
            pooled_output = torch.stack(sent_model.encode_sentences(sentences))
            return pooled_output

    results = {}

    for task in args.tasks:
        se = senteval.engine.SE(params, batcher, prepare)
        result = se.eval(task)
        results[task] = result

    # Print evaluation results
    if args.mode == "dev":
        print("------ %s ------" % args.mode)

        task_names = []
        scores = []
        for task in ["STSBenchmark", "SICKRelatedness"]:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]["dev"]["spearman"][0] * 100))
            else:
                scores.append("0.00")
        tb_sts = print_table(task_names, scores)
        with open(
            os.path.join(
                SENTEVAL_OUTPUT_DIR,
                args.model_name_or_path.split("/")[-1] + "_sts_dev.csv",
            ),
            "w",
            newline="",
        ) as f_output:
            f_output.write(tb_sts.get_csv_string())

        task_names = []
        scores = []
        for task in ["MR", "CR", "SUBJ", "MPQA", "SST2", "TREC", "MRPC"]:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]["devacc"]))
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        tb_transfer = print_table(task_names, scores)
        with open(
            os.path.join(
                SENTEVAL_OUTPUT_DIR,
                args.model_name_or_path.split("/")[-1] + "_transfer_dev.csv",
            ),
            "w",
            newline="",
        ) as f_output:
            f_output.write(tb_transfer.get_csv_string())

    elif args.mode == "test" or args.mode == "fasttest":
        print("------ %s ------" % args.mode)

        task_names = []
        scores = []
        for task in [
            "STS12",
            "STS13",
            "STS14",
            "STS15",
            "STS16",
            "STSBenchmark",
            "SICKRelatedness",
        ]:
            task_names.append(task)
            if task in results:
                if task in ["STS12", "STS13", "STS14", "STS15", "STS16"]:
                    scores.append(
                        "%.2f" % (results[task]["all"]["spearman"]["mean"] * 100)
                    )
                else:
                    # scores.append("%.2f" % (results[task]['test']['spearman'].correlation * 100))
                    scores.append("%.2f" % (results[task]["spearman"] * 100))
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        tb_sts = print_table(task_names, scores)
        with open(
            os.path.join(
                SENTEVAL_OUTPUT_DIR,
                args.model_name_or_path.split("/")[-1] + "_sts_test.csv",
            ),
            "w",
            newline="",
        ) as f_output:
            f_output.write(tb_sts.get_csv_string())

        task_names = []
        scores = []
        for task in ["MR", "CR", "SUBJ", "MPQA", "SST2", "TREC", "MRPC"]:
            task_names.append(task)
            if task in results:
                scores.append("%.2f" % (results[task]["acc"]))
            else:
                scores.append("0.00")
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        print_table(task_names, scores)
        tb_transfer = print_table(task_names, scores)
        with open(
            os.path.join(
                SENTEVAL_OUTPUT_DIR,
                args.model_name_or_path.split("/")[-1] + "_transfer_test.csv",
            ),
            "w",
            newline="",
        ) as f_output:
            f_output.write(tb_transfer.get_csv_string())


if __name__ == "__main__":
    main()
