"""Create a stripplot to visualize differences in mapping models used to obtain the neural encoding results."""

# Imports
import os

import click
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from glob import glob

from sentencefmricomparison.constants import PEREIRA_OUTPUT_DIR


@click.command()
@click.option("--neural-enc-dir", type=str, default=PEREIRA_OUTPUT_DIR)
@click.option("--output-dir", type=str, default=PEREIRA_OUTPUT_DIR)
@click.option("--strip-plot", is_flag=True, default=False)
def plot_mappings_from_neural_enc(
    neural_enc_dir: str = PEREIRA_OUTPUT_DIR,
    output_dir: str = PEREIRA_OUTPUT_DIR,
    strip_plot: bool = False,
) -> plt.Figure:
    """Plot the distribution of neural encoding pairwise accuracy scores obtained with different mapping models.

    :param neural_enc_dir: Directory of the neural encoding results
    :type neural_enc_dir: str
    :param output_dir: Output directory to save the plot to
    :type output_dir: str
    :param strip_plot: Whether to plot a strip plot instead of a boxplot, defaults to False
    :type strip_plot: bool
    :return: Figure object of the plot
    :rtype: plt.Figure
    """
    # 1. Load the relevant neural encoding result csvs
    total_df = pd.DataFrame()
    files = glob(neural_enc_dir + "/pereira_neural_enc_passages_pairwise_accuracy*.csv")
    for f in files:
        result_df = pd.read_csv(f)
        result_df["mapping"] = f.split("_")[-1].strip(".csv")
        total_df = total_df.append(result_df)
    total_df = total_df.rename(columns={"Unnamed: 0": "model"})
    total_df = total_df.drop(columns=["mean"])
    total_df = total_df.melt(
        id_vars=["model", "mapping"],
        value_vars=[col for col in total_df.columns if col not in ["model", "mapping"]],
    )

    # 2. Seaborn box or strip plot
    sns.set(font="Calibri", style="whitegrid", rc={'figure.figsize': (12, 8)},  font_scale=1.5)
    fig, ax = plt.subplots()

    if strip_plot:
        sns.stripplot(
            data=total_df,
            x="model",
            y="value",
            hue="variable",
            palette="Set2",
            dodge=True,
            jitter=False,
        )
    else:
        sns.boxplot(
            data=total_df,
            x="model",
            y="value",
            hue="variable",
            palette="Set2",
            dodge=True,
        )

    # 3. Save the plot
    plt.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "strip_plot_mapping_comp.png"))

    return fig


@click.group()
def cli() -> None:
    """
    This script generates a correlation plot between RSA correlations and neural encoding scores for different sentence
    embedding models.
    """


if __name__ == "__main__":
    cli.add_command(plot_mappings_from_neural_enc)
    cli()
