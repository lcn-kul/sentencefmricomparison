"""Visualize the brain networks used in the analysis."""

# Imports
import os
from typing import List

import click
import matplotlib.pyplot as plt
import nibabel as nib
from matplotlib import colors
from nilearn import image
from nilearn.plotting import plot_roi

from sentencefmricomparison.constants import NETWORK_NIFTI_DIR, PEREIRA_OUTPUT_DIR

NETWORK_NAMES = {
    "DMN": "DMN",
    "language": "Language",
    "MD": "Task",
    "visual": "Vision",
}


@click.option("--network-colors", multiple=True, default=["#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3"])
@click.option("--network-nifti-path", multiple=True, default=NETWORK_NIFTI_DIR)
@click.option("--output-dir", type=str, default=PEREIRA_OUTPUT_DIR)
@click.command()
def visualize_brain_networks(
    network_colors: List[str] = ["#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3"],  # noqa
    network_nifti_path: str = NETWORK_NIFTI_DIR,
    output_dir: str = PEREIRA_OUTPUT_DIR,
):
    """Visualize the brain networks used in the analysis using an MNI template.

    :param network_colors: List of colors to use for each network
    :type network_colors: List[str]
    :param network_nifti_path: Path to the preprocessed nifti files
    :type network_nifti_path: str
    :param output_dir: Output directory to save the plot to
    :type output_dir: str
    :return: -
    :rtype: -
    """
    fig, axes = plt.subplots(2, 2, figsize=(48, 24), gridspec_kw={'wspace': 0.1, 'hspace': -0.2})
    for ax in axes.flatten():
        # Hide grid lines
        ax.grid(False)
        ax.axis('off')
        # Hide axes ticks
        ax.set_xticks([])
        ax.set_yticks([])

    # Iterate through all networks
    for i, (network, color) in enumerate(zip(NETWORK_NAMES.keys(), network_colors)):
        if network == "language":
            # In the case of language, plot left and right hemispheres together by combining the images
            img_rh = image.smooth_img(os.path.join(network_nifti_path, f'{network}RH.nii'), fwhm=1)
            img_lh = image.smooth_img(os.path.join(network_nifti_path, f'{network}LH.nii'), fwhm=1)
            img = nib.Nifti1Image(dataobj=img_lh.dataobj + img_rh.dataobj, affine=img_lh.affine)
        else:
            # Load the network NIfTI files
            img = image.smooth_img(os.path.join(network_nifti_path, f'{network}.nii'), fwhm=1)

        # Plot the brain network
        plot_roi(
            img,
            output_file=os.path.join(output_dir, "network_vis.png"),
            cmap=colors.LinearSegmentedColormap.from_list("", [color] * 10),
            axes=axes.flatten()[i],
        )
        axes.flatten()[i].set_title(NETWORK_NAMES[network], y=0.85, fontsize=18)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "network_vis.png"))

    return


@click.group()
def cli() -> None:
    """Visualize the brain networks used in the analysis."""


if __name__ == "__main__":
    cli.add_command(visualize_brain_networks)
    cli()
