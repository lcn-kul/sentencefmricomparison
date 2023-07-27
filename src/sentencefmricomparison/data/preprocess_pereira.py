"""Loading script of the Pereira et al. 2018 dataset based on their publicly available MATLAB data files.

Code is partially based on Oota et al. 2022 (https://tinyurl.com/langTask).
"""

# Imports
import logging
import os
import pickle  # noqa
from glob import glob
from typing import Dict, List

import click
import numpy as np
import pandas as pd
from scipy.io import loadmat

from datasets import Dataset
from sentencefmricomparison.constants import (
    PEREIRA_EXAMPLE_FILE,
    PEREIRA_INPUT_DIR,
    PEREIRA_RAW_DIR,
)

# Initialize logger
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# These indices are based on the code from Oota et al. 2022 (https://tinyurl.com/langTask)
ROI_INDICES = {
    "language_lh": 7,
    "language_rh": 8,
    "vision_body": 9,
    "vision_face": 10,
    "vision_object": 11,
    "vision_scene": 12,
    "vision": 13,
    "dmn": 6,
    "task": 5,
}


def generate_indices(
    data: Dict,
) -> Dict:
    """Get the right indices for the ROIs used to subset the set of voxels later on.

    Based on Oota et al. 2022 (https://tinyurl.com/langTask).

    :param data: MATLAB data that has been read with scipy's loadmat function
    :type data: Dict
    :return: Dictionary with the right indices to select voxels according to each predefined language network
    :rtype: Dict
    """
    # Load the right field from the MATLAB file
    meta_data = data["meta"][0][0][11][0]
    roi_indices: Dict = {name: [] for name in ROI_INDICES.keys()}

    # Extract the right indices to retrieve the right subset of voxels for each language network
    for name, idx in ROI_INDICES.items():
        index_list = [[int(k) for k in j[0]] for j in meta_data[idx]]
        flattened_list = [element for sublist in index_list for element in sublist]
        roi_indices[name] = flattened_list

    return roi_indices


@click.command()
@click.option("--pereira-input-dir", type=str, default=PEREIRA_RAW_DIR)
@click.option("--pereira-output-dir", type=str, default=PEREIRA_INPUT_DIR)
@click.option("--passage-wise-processing", is_flag=True)
def get_subject_data(
    pereira_input_dir: str = PEREIRA_RAW_DIR,
    pereira_output_dir: str = PEREIRA_INPUT_DIR,
    passage_wise_processing: bool = False,
) -> List[Dict]:
    """Retrieve the subject-specific data based on the Pereira data files and ROIs to subset the voxels.

    Based on Oota et al. 2022 (https://tinyurl.com/langTask).

    :param pereira_input_dir: Path to the base directory with the raw Pereira data
    :type pereira_input_dir: str
    :param pereira_output_dir: Path where the processed data should be saved
    :type pereira_output_dir: str
    :param passage_wise_processing: Whether to get the brain activation for passages instead of single sentences,
        defaults to False
    :type passage_wise_processing: bool
    :return: List of Dicts with the preprocessed fMRI data, each field is based on a different brain region/network
    :rtype: List[Dict]
    """
    # Get all the available subjects
    all_subject_paths = [f.path for f in os.scandir(pereira_input_dir) if f.is_dir()]
    all_fmri = []

    # Get the correct key/file name depending on whether to preprocess sentences or paragraphs
    examples_key = (
        "examples_passages" if passage_wise_processing else "examples_passagesentences"
    )
    file_name_suffix = "_passages" if passage_wise_processing else ""

    # Retrieve the preprocessed images for each subject
    for subject_path in all_subject_paths:
        logger.info(f"Processing {subject_path.split('/')[-1]}")
        # Load the MATLAB file into python
        data_pic = loadmat(os.path.join(subject_path, "data_384sentences.mat"))
        # Get the right indices
        roi_indices = generate_indices(data_pic)

        fmri = {}
        # For each predefined language network (9 in total)
        for roi, indices in roi_indices.items():
            # Subset the overall MRI data according to the language networks
            fmri[roi] = data_pic[examples_key][0:, np.array(indices) - 1]
        # Also save one entry with all voxels
        fmri["all"] = data_pic[examples_key][0:, :]

        # Save as pickled dictionary for each subject, together with the sentences or paragraphs
        if passage_wise_processing:
            fmri["paragraphs"] = np.array(
                [
                    " ".join(sent[0][0] for sent in data_pic["keySentences"][i: i + 4])
                    for i in range(0, len(data_pic["keySentences"]), 4)
                ]
            )
            fmri["topic_indices"] = np.array([i[0] for i in data_pic["labelsPassageCategory"]])
        else:
            fmri["sentences"] = np.array(
                [
                    data_pic["keySentences"][i][0][0]
                    for i in range(len(data_pic["keySentences"]))
                ]
            )
        with open(
            os.path.join(
                pereira_output_dir,
                subject_path.split("/")[-1] + file_name_suffix + "_fmri.pkl",
            ),
            "wb",
        ) as handle:
            pickle.dump(fmri, handle, protocol=pickle.HIGHEST_PROTOCOL)

        all_fmri.append(fmri)

    # Return a list of all fMRI data for all subjects
    return all_fmri


@click.command()
@click.option("--input-filepath", type=str, default=PEREIRA_EXAMPLE_FILE)
@click.option("--output-dir", type=str, default=PEREIRA_INPUT_DIR)
@click.option("--passage-wise-processing", is_flag=True)
def save_pereira_sentences(
    input_filepath: str = PEREIRA_EXAMPLE_FILE,
    output_dir: str = PEREIRA_INPUT_DIR,
    passage_wise_processing: bool = False,
) -> pd.DataFrame:
    """Save the sentences from the Pereira paper to a separate csv file.

    :param input_filepath: fMRI MATLAB data input file, defaults to PEREIRA_EXAMPLE_FILE
    :type input_filepath: str
    :param output_dir: Output directory for the sentence csv file, defaults to PEREIRA_INPUT_DIR
    :type output_dir: str
    :param passage_wise_processing: Whether to get the text for passages instead of single sentences,
        defaults to False
    :type passage_wise_processing: bool
    :return: Data frame with the sentences
    :rtype: pd.DataFrame
    """
    # Load the MATLAB data file
    input_data = loadmat(input_filepath)

    # Set the filename
    file_ending = "passages" if passage_wise_processing else "sentences"

    # Extract the sentences
    sentences = [
        str(input_data["keySentences"][i][0]).strip("'[]")
        for i in range(len(input_data["keySentences"]))
    ]

    if passage_wise_processing:
        sentences = [
            " ".join(str(sent) for sent in sentences[i: i + 4])
            for i in range(0, len(sentences), 4)
        ]

    sent_df = pd.DataFrame(sentences)

    # Save the sentences
    sent_df.to_csv(
        os.path.join(output_dir, "sentences", "pereira_" + file_ending + ".csv"),
        index=False,
    )

    return sent_df


@click.command()
@click.option("--processed-mri-dir", type=str, default=PEREIRA_INPUT_DIR)
@click.option("--passage-wise-processing", is_flag=True, default=False)
@click.option("--push-to-hub", is_flag=True, default=False)
def convert_to_hf_dataset(
    processed_mri_dir: str = PEREIRA_INPUT_DIR,
    push_to_hub: bool = False,
    passage_wise_processing: bool = False,
):
    """Convert the preprocessed MRI data into a huggingface dataset and optionally pushes it to the hub.

    Preliminaries: The original MATLAB file is preprocessed using the get_subject_data function and for pushing to the
    hub, it is assumed that you're logged in using 'huggingface-cli login'

    :param processed_mri_dir: Directory with the preprocessed MRI data
    :type processed_mri_dir: str
    :param push_to_hub: Whether to push to the hub or not, defaults to False
    :type push_to_hub: bool
    :param passage_wise_processing: Whether to work on passages instead of sentences, defaults to False
    :type passage_wise_processing: bool
    :return: None
    :rtype: -
    """
    # Load the right files for passage or sentence-wise processing
    mri_file_names = glob(os.path.join(processed_mri_dir, "*.pkl"))
    if passage_wise_processing:
        mri_file_names = [f for f in mri_file_names if "passage" in f]
    else:
        mri_file_names = [f for f in mri_file_names if "passage" not in f]

    # Add the permuted sentences to the dataset as well
    permuted_file = None
    if passage_wise_processing:
        permuted_file = pd.read_csv(
            os.path.join(processed_mri_dir, "pereira_permuted_passages.csv")
        )

    # Load all the MRI data from the files
    mri_files = []
    for file in mri_file_names:
        with open(file, "rb") as f:
            mri_data_dict = pickle.load(f)  # noqa
            if passage_wise_processing:
                mri_data_dict["permuted_paragraphs"] = permuted_file[  # type: ignore
                    "permuted_sents"
                ].tolist()
            mri_files.append(mri_data_dict)

    # Convert into a huggingface dataset
    dataset = Dataset.from_list(mri_files)

    # Optional: Push to the Hub
    if push_to_hub:
        dataset.push_to_hub(
            "pereira_fMRI_passages"
            if passage_wise_processing
            else "pereira_fMRI_sentences"
        )

    return


@click.command()
@click.option("--pereira-input-file", type=str, default=PEREIRA_EXAMPLE_FILE)
@click.option("--output-dir", type=str, default=PEREIRA_INPUT_DIR)
def generate_permuted_passages(
    pereira_input_file: str = PEREIRA_EXAMPLE_FILE,
    output_dir: str = PEREIRA_INPUT_DIR,
) -> pd.DataFrame:
    """Create a dataframe with 3 types of paragraphs based on the Pereira dataset (not using any fMRI data).

    1) Only 2 middle sentences 2) Original paragraphs (correct first + last sentences) 3) Permuted paragraphs (random
    first + last sentences)

    :param pereira_input_file: (Any) raw Pereira input MATLAB file containing the sentences
    :type pereira_input_file: str
    :param output_dir: Output directory to save the passages to
    :type output_dir: str
    :return: Data frame with 3 types of paragraphs: 1) Only 2 middle sentences 2) Original paragraphs (correct first +
        last sentences) 3) Permuted paragraphs (random first + last sentences)
    :rtype: pd.DataFrame
    """
    # Initialize the result df
    permuted_passages_df = pd.DataFrame()

    # Extract the sentences from the Pereira input file
    data_pic = loadmat(pereira_input_file)
    sentences = [
        data_pic["keySentences"][i][0][0] for i in range(len(data_pic["keySentences"]))
    ]
    # Only take the middle two sentences
    permuted_passages_df["center_sents"] = [
        " ".join(sent for sent in sentences[i + 1: i + 3])
        for i in range(0, len(sentences), 4)
    ]
    # Create the original paragraphs with correct first and last sentences
    permuted_passages_df["paragraphs"] = [
        " ".join(sent for sent in sentences[i: i + 4])
        for i in range(0, len(sentences), 4)
    ]
    # Create paragraphs in which the two middle sentences are paired with random first and last sentences
    permuted_sents = []
    for i in range(0, len(sentences), 4):
        # Get the middle two sentences
        middle = " ".join(sent for sent in sentences[i + 1: i + 3])
        # Get a random first sentence and a random last sentence
        rand_idx_front = np.random.choice(
            [j for j in range(0, len(sentences), 4) if j != i], size=1
        )[0]
        rand_idx_last = np.random.choice(
            [j + 3 for j in range(0, len(sentences), 4) if j != i], size=1
        )[0]
        permuted = " ".join(
            [sentences[rand_idx_front], middle, sentences[rand_idx_last]]
        )
        permuted_sents.append(permuted)

    permuted_passages_df["permuted_sents"] = permuted_sents

    # Save to csv
    permuted_passages_df.to_csv(
        os.path.join(output_dir, "pereira_permuted_passages.csv"), index=False
    )

    return permuted_passages_df


@click.group()
def cli() -> None:
    """Preprocess the Pereira MATLAB files."""


if __name__ == "__main__":
    cli.add_command(get_subject_data)
    cli.add_command(save_pereira_sentences)
    cli.add_command(convert_to_hf_dataset)
    cli.add_command(generate_permuted_passages)
    cli()
