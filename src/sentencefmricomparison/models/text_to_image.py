"""Generate images from text using a text-to-image model."""
import os

# Imports
import click
import torch
from datasets import load_dataset
from diffusers import StableDiffusionXLPipeline

from sentencefmricomparison.constants import (
    LARGE_DATASET_STORAGE_PATH,
    LARGE_MODELS_STORAGE_PATH,
    TEXT_TO_IMAGE_OUTPUT_DIR,
)


@click.command()
@click.option(
    "--stimuli_ds_name",
    default="helena-balabin/pereira_fMRI_passages",
    help="Name of the stimuli dataset to use",
    type=str,
)
@click.option(
    "--model_name",
    default="segmind/SSD-1B",
    help="Name of the text-to-image model to use",
    type=str,
)
@click.option("--sentence_key", default="paragraphs", type=str)
@click.option("--neg_prompt", default="ugly, blurry, poor quality", type=str)
@click.option("--large_model_dir", default=LARGE_MODELS_STORAGE_PATH, type=str)
@click.option("--large_dataset_dir", default=LARGE_DATASET_STORAGE_PATH, type=str)
@click.option("--output_dir", default=TEXT_TO_IMAGE_OUTPUT_DIR, type=str)
def text_to_image(
    stimuli_ds_name: str,
    model_name: str,
    sentence_key: str = "paragraphs",
    neg_prompt: str = "ugly, blurry, poor quality",
    large_model_dir: str = LARGE_MODELS_STORAGE_PATH,
    large_dataset_dir: str = LARGE_DATASET_STORAGE_PATH,
    output_dir: str = TEXT_TO_IMAGE_OUTPUT_DIR,
) -> None:
    """Generate images from the text stimuli in the dataset using a text-to-image model.

    :param stimuli_ds_name: Name of the stimuli dataset to use
    :type stimuli_ds_name: str
    :param model_name: Name of the text-to-image model to use
    :type model_name: str
    :param sentence_key: Name of the feature in the dataset that contains the sentences, defaults to "sentences"
    :type sentence_key: str
    :param neg_prompt: Negative prompt to use for the text-to-image model, defaults to "ugly, blurry, poor quality"
    :type neg_prompt: str
    :param large_model_dir: Directory for saving large models, defaults to LARGE_MODELS_STORAGE_PATH
    :type large_model_dir: str
    :param large_dataset_dir: Directory for saving large datasets, defaults to LARGE_DATASET_STORAGE_PATH
    :type large_dataset_dir: str
    :param output_dir: Output directory for saving the generated images, defaults to TEXT_TO_IMAGE_OUTPUT_DIR
    :type output_dir: str
    """
    # 1. Load the stimuli dataset
    stimuli_ds = load_dataset(
        stimuli_ds_name,
        cache_dir=large_dataset_dir,
    )["train"]
    # Get one test example from the stimuli dataset
    examples = stimuli_ds[sentence_key][0]

    # 2. Load the text-to-image model
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
        cache_dir=large_model_dir,
        device_map="auto",
    )
    pipe.to("cuda")

    # 3. Generate the image from the text
    for i, example in enumerate(examples):
        image = pipe(prompt=example, negative_prompt=neg_prompt).images[0]

        # 4. Save the generated image
        image.save(os.path.join(output_dir, f"stimulus_{i}.png"))


@click.group()
def cli() -> None:
    """Generate images from the text stimuli in the dataset using a text-to-image model."""


if __name__ == "__main__":
    cli.add_command(text_to_image)
    cli()
