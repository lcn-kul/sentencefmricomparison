"""Visualize the confidence intervals obtained with bootstrapping the correlation values."""

# Imports
import os

import click
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sentencefmricomparison.constants import CUSTOM_COLOR_PALETTE, PEREIRA_OUTPUT_DIR, SENT_EMBED_MODEL_NAMES_EN


@click.command()
@click.option(
    "--subj_specific-file-path",
    type=str,
    default=os.path.join(PEREIRA_OUTPUT_DIR, "subj_rsa_correlations_spearman_cosine_paragraphs.csv"),
)
@click.option("--model-wise", is_flag=True, default=False)
@click.option("--output-dir", type=str, default=PEREIRA_OUTPUT_DIR)
def plot_conf_intervals(
    subj_specific_file_path: str = os.path.join(
        PEREIRA_OUTPUT_DIR,
        "subj_rsa_correlations_spearman_cosine_paragraphs.csv",
    ),
    model_wise: bool = False,
    output_dir: str = PEREIRA_OUTPUT_DIR,
) -> plt.Figure:
    """Plots violin plots and point plots with 95% CIs against RSA correlations, grouped by models or paradigms.

    :param subj_specific_file_path: Path to the subject-specific correlations
    :type subj_specific_file_path: str
    :param model_wise: Whether to group by models instead of paradigms, defaults to False
    :type model_wise: str
    :param output_dir: Output directory to save the plot to
    :type output_dir: str
    :return: Figure object of the plot
    :rtype: plt.Figure
    """
    col = "model" if model_wise else "paradigm"

    # 1. Load the file
    subj_specific_df = pd.read_csv(subj_specific_file_path)

    # 2. Plot a violin plot with a point plot showing the mean and 95% CI
    sns.set(font="Calibri", style="whitegrid", rc={'figure.figsize': (12, 8)},  font_scale=1.5)
    fig, ax = plt.subplots()
    sns.violinplot(
        ax=ax,
        x=col,
        y="correlation",
        hue="roi",
        data=subj_specific_df,
        hue_order=subj_specific_df["roi"].unique(),
        palette=CUSTOM_COLOR_PALETTE,
        cut=0,
        inner=None,
    )
    sns.pointplot(
        dodge=0.6,
        ax=ax,
        x=col,
        y="correlation",
        hue="roi",
        hue_order=subj_specific_df["roi"].unique(),
        data=subj_specific_df,
        join=False,
        capsize=0.05,
        palette=["#6e6e6e"] * len(subj_specific_df[col].unique()),
    )
    ax.set_xticklabels(
        labels=[
            SENT_EMBED_MODEL_NAMES_EN[i.get_text()] for i in list(ax.get_xticklabels())
        ] if model_wise else ax.get_xticklabels(),
        rotation=45 if model_wise else 15,
    )
    ax.set_ylabel("spearman correlation (RSA)")
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles[0:len(subj_specific_df["roi"].unique())], labels[0:len(subj_specific_df["roi"].unique())])
    plt.tight_layout()

    # Save the figure
    fig.savefig(os.path.join(output_dir, f"conf_intervals_{col}.png"))

    return fig


@click.group()
def cli() -> None:
    """
    Visualize the confidence intervals obtained with bootstrapping the correlation values.
    """


if __name__ == "__main__":
    cli.add_command(plot_conf_intervals)
    cli()
