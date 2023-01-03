"""Create a correlation plot between the RSA correlations and neural encoding performances of different models."""

# Imports
import logging
import os
from itertools import combinations
from glob import glob
from typing import List

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torchmetrics.functional.pairwise import pairwise_cosine_similarity

from sentencefmricomparison.constants import (
    PEREIRA_OUTPUT_DIR,
    PEREIRA_PERMUTED_SENTENCES_PATH,
    SENTEVAL_OUTPUT_DIR,
    SENT_EMBED_MODEL_LIST_EN,
    SENT_EMBED_MODEL_NAMES_EN,
)
from sentencefmricomparison.models.sentence_embedding_base import SentenceEmbeddingModel
from sentencefmricomparison.rsa.rsa_pereira import get_sim_vector
from sentencefmricomparison.utils import CORRELATION_MEASURES

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
@click.option("--per-brain-network", type=bool, default=False)
@click.option("--color", type=str, default="#C44536")
@click.option("--output-dir", type=str, default=PEREIRA_OUTPUT_DIR)
def plot_corr_rsa_neural_enc(
    rsa_input_file: str = os.path.join(PEREIRA_OUTPUT_DIR, "rsa_correlations_spearman_cosine_paragraphs.csv"),
    neural_enc_input_file: str = os.path.join(
        PEREIRA_OUTPUT_DIR,
        "pereira_neural_enc_sentences_pairwise_accuracy_ridge.csv",
    ),
    per_brain_network: bool = False,
    color: str = "#C44536",
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
    :param per_brain_network: Whether to plot separate regression lines per brain network
    :type per_brain_network: bool
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

    fig = plot_scatter_regplot(
        data_df=concat_df,
        color=color,
        output_path=os.path.join(output_dir, "corr_rsa_neural_enc.png"),
        x_axis="neural_enc",
        x_label="pairwise accuracy (neural encoding)",
        y_axis="rsa_corr",
        y_label="spearman correlation (RSA)",
        per_brain_network=per_brain_network,
    )

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
@click.option("--per-brain-network", type=bool, default=False)
@click.option("--color", type=str, default="#DBAD6A")
@click.option("--output-dir", type=str, default=PEREIRA_OUTPUT_DIR)
def plot_corr_rsa_sent_eval(
    rsa_input_file: str = os.path.join(PEREIRA_OUTPUT_DIR, "rsa_correlations_spearman_cosine_paragraphs.csv"),
    sent_eval_results_dir: str = SENTEVAL_OUTPUT_DIR,
    sent_eval_mode: str = "transfer",
    per_brain_network: bool = False,
    color: str = "#DBAD6A",
    output_dir: str = PEREIRA_OUTPUT_DIR,
) -> plt.Figure:
    """Create a correlation plot between RSA correlations and SentEval scores.

    :param rsa_input_file: Input file for the RSA correlations between sentence embeddings and fMRI features
    :type rsa_input_file: str
    :param sent_eval_results_dir: Directory of the SentEval results
    :type sent_eval_results_dir: str
    :param sent_eval_mode: Whether to use "transfer" or "sts" SentEval results
    :type sent_eval_mode: str
    :param per_brain_network: Whether to plot separate regression lines per brain network
    :type per_brain_network: bool
    :param output_dir: Output directory to save the plot to
    :type output_dir: str
    :param color: Color of the regression line
    :type color: str
    :return: Figure with lineplots of correlations between RSA and SentEval scores
    :rtype: plt.Figure
    """
    # 1. Extend the RSA results DataFrame with the SentEval results for the listed models (if respective SentEval scores
    # exist)
    rsa_corr_df = pd.read_csv(rsa_input_file, index_col=0, usecols=lambda x: "p-value" not in x)
    for model_name in rsa_corr_df.index:
        try:
            file_path = glob(
                os.path.join(sent_eval_results_dir, model_name.split("/")[-1] + f"_{sent_eval_mode}_test.csv"),
            )[0]
            rsa_corr_df.loc[model_name, "sent_eval_avg"] = pd.read_csv(file_path)["Avg."][0]
        except IndexError:
            continue
    rsa_corr_df = rsa_corr_df.dropna()

    # 2. Melt all different ROI scores
    rsa_corr_df = rsa_corr_df.melt(
        id_vars=["sent_eval_avg"],
        value_vars=[col for col in rsa_corr_df.columns if "sent_eval_avg" not in col],
    )
    rsa_corr_df = rsa_corr_df[rsa_corr_df["variable"] != "mean"]
    rsa_corr_df = rsa_corr_df.rename(columns={"variable": "roi", "value": "rsa_corr"})

    # 3. Plot it
    fig = plot_scatter_regplot(
        data_df=rsa_corr_df,
        color=color,
        output_path=os.path.join(output_dir, "corr_rsa_sent_eval.png"),
        x_axis="sent_eval_avg",
        x_label=f"SentEval {sent_eval_mode} scores",
        y_axis="rsa_corr",
        y_label="spearman correlation (RSA)",
        per_brain_network=per_brain_network,
    )

    return fig


def plot_scatter_regplot(
    data_df: pd.DataFrame,
    color: str = "#CF995F",
    output_path: str = os.path.join(PEREIRA_OUTPUT_DIR, "corr_rsa_neural_enc.png"),
    x_axis: str = "neural_enc",
    x_label: str = "Pairwise accuracy (neural encoding)",
    y_axis: str = "rsa_corr",
    y_label: str = "Spearman correlation (RSA)",
    per_brain_network: bool = False,
) -> plt.Figure:
    """Plotting helper function for line/regression plots between various variables from various analyses.

    :param data_df: Dataframe with the data for the regression plot
    :type data_df: pd.DataFrame
    :param color: Color for the lineplot
    :type color: str
    :param output_path: Output path to save the plot to
    :type output_path: str
    :param x_axis: Column name of the dataframe for the data to use for the x-axis
    :type x_axis: str
    :param x_label: x-label of the plot
    :type x_label: str
    :param y_axis: Column name of the dataframe for the data to use for the y-axis
    :type y_axis: str
    :param y_label: y-label of the plot
    :type y_label: str
    :param per_brain_network: Whether to plot separate regression lines per brain network
    :type per_brain_network: bool
    :return: Figure object of the plot
    :rtype: plt.Figure
    """

    # Calcualte the correlation
    corr, p_value = CORRELATION_MEASURES['pearson'](data_df[x_axis], data_df[y_axis])

    # Brain networks
    brain_networks = np.unique(data_df["roi"])

    # Plot it
    sns.set(font="Calibri", style="whitegrid", rc={'figure.figsize': (12, 8)},  font_scale=1.5)
    fig, ax = plt.subplots()

    sns.regplot(
        data_df[x_axis],
        data_df[y_axis],
        ax=ax,
        scatter=False,
        color=color,
        line_kws={'linewidth': 1.5},
    )

    labels = ["mean (r=" + str(round(corr, 2)) + ", p=" + str(round(p_value, 4)) + ")"]

    if per_brain_network:
        for i, b in enumerate(brain_networks):
            network_specific_x = data_df[data_df["roi"] == b][x_axis]
            network_specific_y = data_df[data_df["roi"] == b][y_axis]
            sns.regplot(
                network_specific_x,
                network_specific_y,
                ax=ax,
                scatter=False,
                color=sns.color_palette("Set2")[i],
                ci=None,
                line_kws={'linewidth': 1},
                truncate=False,
            )
            # Calcualte the correlation
            corr, p_value = CORRELATION_MEASURES['pearson'](network_specific_x, network_specific_y)
            labels.append(b + " (r=" + str(round(corr, 2)) + ")")

    sns.scatterplot(
        data=data_df,
        x=x_axis,
        y=y_axis,
        hue="roi",
        palette="Set2",
        ax=ax,
        s=75,
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.legend(loc="upper left", labels=labels if len(labels) > 0 else None)
    plt.tight_layout()
    fig.savefig(output_path)

    return fig


@click.command()
@click.option("--paragraph-input-path", type=str, default=PEREIRA_PERMUTED_SENTENCES_PATH)
@click.option("--color", type=str, default="#DBAD6A")
@click.option("--sent-model-names", multiple=True, default=SENT_EMBED_MODEL_LIST_EN)
@click.option("--output-dir", default=PEREIRA_OUTPUT_DIR)
def plot_correlogram_sent_models(
    paragraph_input_path: str = PEREIRA_PERMUTED_SENTENCES_PATH,  # One could also use the hf dataset here
    sent_model_names: List[str] = SENT_EMBED_MODEL_LIST_EN,
    output_dir: str = PEREIRA_OUTPUT_DIR,
) -> plt.Figure:
    """Plot a correlogram to see how the sentence embedding models are correlated to each other.

    :param paragraph_input_path: Input path to the file containing the paragraphs
    :type paragraph_input_path: str
    :param sent_model_names: List of sentence embedding model names
    :type sent_model_names: List[str]
    :param output_dir: Output directory to save the plot to
    :type output_dir: str
    :return: Correlogram plot
    :rtype: plt.Figure
    """
    # 1. Get vectorized RDMs for each model
    # Load the paragraphs
    paragraphs = pd.read_csv(paragraph_input_path)["paragraphs"]
    # Create RDMs based on the paragraphs for each model
    sent_model_rdms = {}
    for sent_model_name in sent_model_names:
        model = SentenceEmbeddingModel(sent_model_name)
        sent_model_rdms[sent_model_name] = get_sim_vector(model.generate_rdm(paragraphs))
        del model

    # 2. Get all pairwise correlations
    pair_corr_df = pd.DataFrame(
        data=pairwise_cosine_similarity(torch.stack([torch.tensor(t) for t in sent_model_rdms.values()])).numpy(),
        index=sent_model_names,
        columns=sent_model_names,
    )
    pair_corr_df = pair_corr_df.rename(columns=SENT_EMBED_MODEL_NAMES_EN, index=SENT_EMBED_MODEL_NAMES_EN)

    # 3. Plot them
    # Plot settings
    sns.set(font="Calibri", style="whitegrid", rc={'figure.figsize': (12, 8)},  font_scale=1)
    fig, ax = plt.subplots()
    # Only use the lower triangular matrix, excluding the diagonal
    mask = np.zeros_like(pair_corr_df)
    mask[np.triu_indices_from(mask)] = True
    # Draw the heatmap with the mask
    heatmap = sns.heatmap(
        pair_corr_df,
        mask=mask,
        square=True,
        cmap=sns.diverging_palette(220, 20, as_cmap=True),
        # vmin=0.0,
        # vmax=1.0,
    )
    heatmap.set_xticklabels(heatmap.get_yticklabels(), rotation=45)
    # Save the plot
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "corr_plot_sent_embed_models.png"))

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
    cli.add_command(plot_correlogram_sent_models)
    cli()
