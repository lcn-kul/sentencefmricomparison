"""Command line interface for sentencefmricomparison."""
# Imports
import click

from sentencefmricomparison.data.preprocess_pereira import (
    convert_to_hf_dataset,
    generate_permuted_passages,
    get_subject_data,
    save_pereira_sentences,
)
from sentencefmricomparison.models.neural_encoder import (
    calculate_brain_scores_cv_wrapper,
    hpo_neural_encoder,
)
from sentencefmricomparison.rsa.rsa_corr_analysis import (
    perform_anova_tukey,
)
from sentencefmricomparison.rsa.rsa_pereira import (
    perform_rsa,
    perform_rsa_text_permutations,
)
from sentencefmricomparison.visualization.generate_corr_plots import (
    plot_corr_rsa_neural_enc,
    plot_corr_rsa_sent_eval,
    plot_correlogram_sent_models,
    plot_corr_neural_enc_embed_size,
)
from sentencefmricomparison.visualization.visualize_brain_networks import (
    visualize_brain_networks,
)


@click.group()
def cli() -> None:
    """
    This is the command line interface for sentencefmricomparison.
    """


if __name__ == "__main__":
    cli.add_command(calculate_brain_scores_cv_wrapper)
    cli.add_command(convert_to_hf_dataset)
    cli.add_command(generate_permuted_passages)
    cli.add_command(get_subject_data)
    cli.add_command(hpo_neural_encoder)
    cli.add_command(perform_anova_tukey)
    cli.add_command(perform_rsa)
    cli.add_command(plot_corr_neural_enc_embed_size)
    cli.add_command(plot_corr_rsa_neural_enc)
    cli.add_command(plot_corr_rsa_sent_eval)
    cli.add_command(plot_correlogram_sent_models)
    cli.add_command(perform_rsa_text_permutations)
    cli.add_command(save_pereira_sentences)
    cli.add_command(visualize_brain_networks)
    cli()
