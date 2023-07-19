"""Additional functions to analyze the RSA results."""

# Imports
import os

import click
import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd  # noqa

from sentencefmricomparison.constants import PEREIRA_OUTPUT_DIR


@click.command()
@click.option(
    "--input-path",
    type=str,
    default=os.path.join(
        PEREIRA_OUTPUT_DIR, "subj_rsa_correlations_spearman_cosine_paragraphs.csv"
    ),
)
@click.option("--model-wise", is_flag=True, default=False)
@click.option("--sig", type=float, default=0.05)
def perform_anova_tukey(
    input_path: os.path.join(
        PEREIRA_OUTPUT_DIR, "subj_rsa_correlations_spearman_cosine_paragraphs.csv"  # noqa
    ),
    model_wise: bool = False,
    sig: float = 0.05,
):
    """Test for significant differences across models or paradigms using subject-specific RSA correlations.

    :param input_path: Input path to the subject-specific RSA results
    :type input_path: str
    :param model_wise: Whether to perform the analyses group by models rather than paradigms, defaults to False
    :type model_wise: bool
    :param sig: Significance level for both ANOVA and post-hoc Tukey's HSD
    :type sig: float
    :return: None
    :rtype: -
    """
    # 1. Load the file
    subj_corr = pd.read_csv(input_path)
    col = "model" if model_wise else "paradigm"

    for roi in subj_corr["roi"].unique():
        # 2. Perform ANOVA to see whether the means across models or paradigms are significantly different
        roi_subj_corr = subj_corr[subj_corr["roi"] == roi]
        groups = [
            np.array(roi_subj_corr[roi_subj_corr[col] == i]["correlation"])
            for i in roi_subj_corr[col].unique()
        ]
        anova = f_oneway(*groups)
        pvalue = anova.pvalue
        logger.info(f"{roi}: {pvalue}")

        if pvalue < sig:
            # 3. Perform post-hoc tukey HSD testing if the means are significantly different from each other
            tukey = pairwise_tukeyhsd(
                endog=roi_subj_corr["correlation"],
                groups=roi_subj_corr[col],
                alpha=sig,
            )
            logger.info(tukey)
            logger.info("\n\n")

    return


@click.group()
def cli() -> None:
    """Analyze the RSA results."""


if __name__ == "__main__":
    cli.add_command(perform_anova_tukey)
    cli()
