"""Create a correlation plot between the RSA correlations and neural encoding performances of different models."""

# Imports
import logging
import os
from glob import glob

import click
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import linregress

from sentencefmricomparison.constants import PEREIRA_OUTPUT_DIR, SENTEVAL_OUTPUT_DIR
from sentencefmricomparison.models.sentence_embedding_base import SentenceEmbeddingModel

# Initialize the logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@click.command()
@click.option(
    "--rsa-input-file",
    type=str,
    default=os.path.join(PEREIRA_OUTPUT_DIR, "rsa_correlations_spearman_cosine_paragraphs.csv"),
)
@click.option(
    "--neural-enc-input-file",
    type=str,
    default=os.path.join(PEREIRA_OUTPUT_DIR, "pereira_neural_enc_passages_pairwise_accuracy.csv"),
)
@click.option("--color", type=str, default="#69BDB2")
@click.option("--output-dir", type=str, default=PEREIRA_OUTPUT_DIR)
def plot_corr_rsa_neural_enc(
    rsa_input_file: str = os.path.join(PEREIRA_OUTPUT_DIR, "rsa_correlations_spearman_cosine_paragraphs.csv"),
    neural_enc_input_file: str = os.path.join(
        PEREIRA_OUTPUT_DIR,
        "pereira_neural_enc_sentences_pairwise_accuracy.csv",
    ),
    color: str = "#69BDB2",
    output_dir: str = PEREIRA_OUTPUT_DIR,
) -> plt.Figure:
    """ Create a figure with lineplots of correlations between RSA and neural encoding scores.

    Each data point is representing a sentence embedding model for which an RSA score and a neural encoding score
    exists.

    :param rsa_input_file: Input file for the RSA correlations between sentence embeddings and fMRI features
    :type rsa_input_file: str
    :param neural_enc_input_file: Input file for neural encoding scores for sentence embedding models
    :type neural_enc_input_file: str
    :param output_dir: Output directory to save the plot to
    :type output_dir: str
    :param color: Color of the regression line
    :type color: str
    :return: Figure with lineplots of correlations between RSA and neural encoding scores
    :rtype: plt.Figure
    """
    rsa_corr = pd.read_csv(rsa_input_file, index_col=0)
    neural_enc = pd.read_csv(neural_enc_input_file, index_col=0)

    intersecting_models = rsa_corr.index.intersection(neural_enc.index)
    intersecting_rois = rsa_corr.columns.intersection(neural_enc.columns).drop("mean")

    rsa_corr = rsa_corr.loc[intersecting_models, intersecting_rois].reset_index()
    neural_enc = neural_enc.loc[intersecting_models, intersecting_rois].reset_index()

    rsa_corr = rsa_corr.melt(id_vars=["index"], value_vars=[col for col in rsa_corr.columns if "index" not in col])
    rsa_corr = rsa_corr.rename(columns={"variable": "roi", "value": "rsa_corr"})
    neural_enc = neural_enc.melt(
        id_vars=["index"], value_vars=[col for col in neural_enc.columns if "index" not in col]
    )
    neural_enc = neural_enc.rename(columns={"variable": "roi", "value": "neural_enc"})
    concat_df = rsa_corr.merge(neural_enc)

    # Get the parameters for a regression line
    slope, intercept, r_value, p_value, std_err = linregress(concat_df["neural_enc"], concat_df["rsa_corr"])

    sns.set(font="Calibri", style="whitegrid", rc={'figure.figsize': (16, 8)})
    fig, ax = plt.subplots()
    sns.regplot(concat_df["neural_enc"], concat_df["rsa_corr"], ax=ax, scatter=False, color=color)
    sns.scatterplot(
        data=concat_df,
        x="neural_enc",
        y="rsa_corr",
        hue="roi",
        palette="Set2",
        ax=ax,
    )
    ax.set_xlabel("Pairwise accuracy (neural encoding)")
    ax.set_ylabel("Spearman correlation (RSA)")
    ax.set_title(
        f"Correlation between RSA and neural encoding: y= {str(round(slope, 2))}x + {str(round(intercept, 2))}\n"
        + f"with resulting p={round(p_value, 5)}",
    )
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)
    plt.tight_layout()
    save_fig = fig.get_figure()
    save_fig.savefig(os.path.join(output_dir, "corr_rsa_neural_enc.png"))

    return fig


@click.command()
@click.option(
    "--rsa-input-file",
    type=str,
    default=os.path.join(PEREIRA_OUTPUT_DIR, "rsa_correlations_spearman_cosine_paragraphs.csv"),
)
@click.option(
    "--sent-eval-results-dir",
    type=str,
    default=SENTEVAL_OUTPUT_DIR,
)
@click.option("--sent_eval_mode", type=click.Choice(["sts", "transfer"]), default="transfer")
@click.option("--color", type=str, default="#CEDEB2")
@click.option("--output-dir", type=str, default=PEREIRA_OUTPUT_DIR)
def plot_corr_rsa_sent_eval(
    rsa_input_file: str = os.path.join(PEREIRA_OUTPUT_DIR, "rsa_correlations_spearman_cosine_paragraphs.csv"),
    sent_eval_results_dir: str = SENTEVAL_OUTPUT_DIR,
    sent_eval_mode: str = "transfer",
    color: str = "#CEDEB2",
    output_dir: str = PEREIRA_OUTPUT_DIR,
) -> plt.Figure:
    """Create a correlation plot between RSA correlations and SentEval scores.

    :param rsa_input_file: Input file for the RSA correlations between sentence embeddings and fMRI features
    :type rsa_input_file: str
    :param sent_eval_results_dir: Directory of the SentEval results
    :type sent_eval_results_dir: str
    :param sent_eval_mode: Whether to use "transfer" or "sts" SentEval results
    :type sent_eval_mode: str
    :param output_dir: Output directory to save the plot to
    :type output_dir: str
    :param color: Color of the regression line
    :type color: str
    :return: Figure with lineplots of correlations between RSA and SentEval scores
    :rtype: plt.Figure
    """
    # 1. Extend the RSA results DataFrame with the SentEval results for the listed models
    rsa_corr_df = pd.read_csv(rsa_input_file, index_col=0, usecols=lambda x: "p-value" not in x)
    for model_name in rsa_corr_df.index:
        file_path = glob(
            os.path.join(sent_eval_results_dir, model_name.split("/")[-1] + f"_{sent_eval_mode}_test.csv"),
        )[0]
        rsa_corr_df.loc[model_name, "sent_eval_avg"] = pd.read_csv(file_path)["Avg."][0]

    # Melt all different ROI scores
    rsa_corr_df = rsa_corr_df.melt(
        id_vars=["sent_eval_avg"],
        value_vars=[col for col in rsa_corr_df.columns if "sent_eval_avg" not in col or "mean" not in col],
    )
    rsa_corr_df = rsa_corr_df.rename(columns={"variable": "roi", "value": "rsa_corr"})
    rsa_corr_mean_df = rsa_corr_df[rsa_corr_df["roi"] == "mean"]

    # 2. Apply linear regression
    slope, intercept, r_value, p_value, std_err = linregress(
        rsa_corr_mean_df["sent_eval_avg"],
        rsa_corr_mean_df["rsa_corr"],
    )

    # 3. Plot it
    sns.set(font="Calibri", style="whitegrid", rc={'figure.figsize': (16, 8)})
    fig, ax = plt.subplots()
    sns.regplot(rsa_corr_mean_df["sent_eval_avg"], rsa_corr_mean_df["rsa_corr"], ax=ax, scatter=False, color=color)
    sns.scatterplot(
        data=rsa_corr_df,
        x="sent_eval_avg",
        y="rsa_corr",
        hue="roi",
        palette="Set2",
        ax=ax,
    )
    ax.set_xlabel(f"SentEval {sent_eval_mode} scores")
    ax.set_ylabel("Spearman correlation (RSA)")
    ax.set_title(
        f"Correlation between RSA and SentEval scores: y= {str(round(slope, 5))}x + {str(round(intercept, 5))}\n"
        + f"with resulting p={round(p_value, 5)}",
    )
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)
    plt.tight_layout()
    save_fig = fig.get_figure()
    save_fig.savefig(os.path.join(output_dir, "corr_rsa_sent_eval.png"))

    return fig


@click.command()
@click.option(
    "--rsa-input-file",
    type=str,
    default=os.path.join(PEREIRA_OUTPUT_DIR, "rsa_correlations_spearman_cosine_paragraphs.csv"),
)
@click.option("--color", type=str, default="#DBAD6A")
@click.option("--output-dir", type=str, default=PEREIRA_OUTPUT_DIR)
def plot_corr_rsa_model_size(
    rsa_input_file: str = os.path.join(PEREIRA_OUTPUT_DIR, "rsa_correlations_spearman_cosine_paragraphs.csv"),
    color: str = "#DBAD6A",
    output_dir: str = PEREIRA_OUTPUT_DIR,
):
    """Create a correlation plot between RSA correlations and model size

    :param rsa_input_file: Input file for the RSA correlations between sentence embeddings and fMRI features
    :type rsa_input_file: str
    :param color: Color for the lineplot
    :type color: str
    :param output_dir: Output directory to save the plot to
    :type output_dir: str
    :return: Figure with lineplots of correlations between RSA and model sizes
    :rtype: plt.Figure
    """
    # 1. Get the model size for each model
    rsa_corr_df = pd.read_csv(rsa_input_file, index_col=0, usecols=lambda x: "p-value" not in x)
    for model_name in rsa_corr_df.index:
        model = SentenceEmbeddingModel(model_name).model
        sum_with_grad = sum(p.numel() for p in model.parameters() if p.requires_grad)
        sum_wo_grad = sum(p.numel() for p in model.parameters())
        model_size = sum_with_grad if sum_with_grad > 0 else sum_wo_grad
        rsa_corr_df.loc[model_name, "model_size"] = model_size

    rsa_corr_df["model_size"] = rsa_corr_df["model_size"] / 1e7

    # Melt all different ROI scores
    rsa_corr_df = rsa_corr_df.melt(
        id_vars=["model_size"],
        value_vars=[col for col in rsa_corr_df.columns if "model_size" not in col or "mean" not in col],
    )
    rsa_corr_df = rsa_corr_df.rename(columns={"variable": "roi", "value": "rsa_corr"})
    rsa_corr_mean_df = rsa_corr_df[rsa_corr_df["roi"] == "mean"]

    # 2. Apply linear regression
    slope, intercept, r_value, p_value, std_err = linregress(
        rsa_corr_mean_df["model_size"],
        rsa_corr_mean_df["rsa_corr"],
    )

    # 3. Plot it
    sns.set(font="Calibri", style="whitegrid", rc={'figure.figsize': (16, 8)})
    fig, ax = plt.subplots()
    sns.regplot(rsa_corr_mean_df["model_size"], rsa_corr_mean_df["rsa_corr"], ax=ax, scatter=False, color=color)
    sns.scatterplot(
        data=rsa_corr_df,
        x="model_size",
        y="rsa_corr",
        hue="roi",
        palette="Set2",
        ax=ax,
    )
    ax.set_xlabel(f"Model size (*1e7)")
    ax.set_ylabel("Spearman correlation (RSA)")
    ax.set_title(
        f"Correlation between RSA and model size: y= {str(round(slope, 5))}x + {str(round(intercept, 5))}\n"
        + f"with resulting p={round(p_value, 5)}",
    )
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)
    plt.tight_layout()
    save_fig = fig.get_figure()
    save_fig.savefig(os.path.join(output_dir, "corr_rsa_model_size.png"))

    return fig


def plot_scatter_regplot(
    data_df: pd.DataFrame,
    color: str = "#69BDB2",
    output_path: str = os.path.join(PEREIRA_OUTPUT_DIR, "corr_rsa_neural_enc.png"),
    x_axis: str = "neural_enc",
    x_label: str = "Pairwise accuracy (neural encoding)",
    y_axis: str = "rsa_corr",
    y_label: str = "Spearman correlation (RSA)",
) -> plt.Figure:
    # TODO documentation
    # TODO use this method in the three functions above

    # Apply linear regression
    slope, intercept, r_value, p_value, std_err = linregress(
        data_df[x_axis],
        data_df[y_axis],
    )

    # Plot it
    sns.set(font="Calibri", style="whitegrid", rc={'figure.figsize': (16, 8)})
    fig, ax = plt.subplots()
    sns.regplot(data_df[x_axis], data_df[y_axis], ax=ax, scatter=False, color=color)
    sns.scatterplot(
        data=data_df,
        x=x_axis,
        y=y_axis,
        hue="roi",
        palette="Set2",
        ax=ax,
    )
    ax.set_xlabel(f"Model size (*1e7)")
    ax.set_ylabel("Spearman correlation (RSA)")
    ax.set_title(
        f"Correlation between {x_label} and {y_label}:\n y= {str(round(slope, 5))}x + {str(round(intercept, 5))}\n"
        + f"with resulting p={round(p_value, 5)}",
    )
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)
    plt.tight_layout()
    save_fig = fig.get_figure()
    save_fig.savefig(output_path)

    return fig


@click.group()
def cli() -> None:
    """
    This script generates a correlation plot between RSA correlations and neural encoding scores for different sentence
    embedding models.
    """


if __name__ == "__main__":
    cli.add_command(plot_corr_rsa_neural_enc)
    cli.add_command(plot_corr_rsa_sent_eval)
    cli.add_command(plot_corr_rsa_model_size)
    cli()
