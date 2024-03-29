"""Apply RSA to sentence embedding models and fMRI data from Pereira et al. (2018)."""

# Imports
import os
from glob import glob
from typing import List, Union

import click
import numpy as np
import pandas as pd
import pickle  # noqa
from loguru import logger
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import pairwise_distances
from tqdm import tqdm

from sentencefmricomparison.constants import (
    PEREIRA_INPUT_DIR,
    PEREIRA_OUTPUT_DIR,
    PEREIRA_PERMUTED_SENTENCES_PATH,
    SENT_EMBED_MODEL_LIST_EN,
    SENT_EMBED_MODEL_PARADIGMS,
)
from sentencefmricomparison.data.preprocess_pereira import ROI_INDICES
from sentencefmricomparison.models.sentence_embedding_base import SentenceEmbeddingModel
from sentencefmricomparison.utils import (
    CORRELATION_MEASURES,
    PAIRWISE_DISTANCES,
    permutation_test,
)


def get_sim_vector(
    sim_matrix: Union[np.array, pd.DataFrame],
) -> np.array:
    """Get the upper triangle from a similarity matrix and vectorize it.

    :param sim_matrix: Similarity matrix from which to extract the upper triangle
    :type sim_matrix: np.array
    :return: Upper triangle (no diagonal) as a vector
    :rtype: np.array
    """
    if isinstance(sim_matrix, pd.DataFrame):
        sim_matrix = sim_matrix.values
    upper_triangle_values = sim_matrix[np.triu_indices(len(sim_matrix), k=1)]
    return upper_triangle_values


@click.command()
@click.option("--pereira-input-path", type=str, default=PEREIRA_INPUT_DIR)
@click.option(
    "--sent-model-names",
    multiple=True,
    default=SENT_EMBED_MODEL_LIST_EN,
)
@click.option("--pairwise-metric", type=str, default="cosine")
@click.option(
    "--correlation-metric",
    type=click.Choice(CORRELATION_MEASURES.keys()),
    default="spearman",
)
@click.option("--n-resamples-permutation", type=int, default=10000)
@click.option("--num-sentences", type=int, default=-1)
@click.option("--alpha", type=float, default=0.05)
@click.option("--passage-wise-processing", type=bool, default=True)
@click.option("--individual-correlations", type=bool, default=False)
@click.option("--output-dir", type=str, default=PEREIRA_OUTPUT_DIR)
def perform_rsa(
    pereira_input_path: str = PEREIRA_INPUT_DIR,
    sent_model_names: List[str] = SENT_EMBED_MODEL_LIST_EN,
    pairwise_metric: str = "cosine",
    correlation_metric: str = "spearman",
    n_resamples_permutation: int = 10000,
    num_sentences: int = -1,
    alpha: float = 0.05,
    passage_wise_processing: bool = True,
    individual_correlations: bool = False,
    output_dir: str = PEREIRA_OUTPUT_DIR,
) -> pd.DataFrame:
    """Perform simple correlation analyses on multiple sentence embedding models that are compared to the fMRI data.

    :param pereira_input_path: Input path to the Pereira fMRI data
    :type pereira_input_path: str
    :param sent_model_names: Sentence model name per sentence embedding model
    :type sent_model_names: List[str]
    :param pairwise_metric: Metric used for calculating the similarity matrices (in-between ROIs/LMs)
    :type pairwise_metric: str
    :param correlation_metric: Metric used for calculating the correlation across ROIs and LMs
    :type correlation_metric: str
    :param n_resamples_permutation: Number of permutations for the permutation test that measures statistical
        significance for the calculated correlations
    :type n_resamples_permutation: int
    :param num_sentences: Number of sentences used in the overall analysis, defaults to -1 (all)
    :type num_sentences: int
    :param alpha: Alpha level for the CIs based on permutation testing, defaults to 0.05
    :type alpha: float
    :param passage_wise_processing: Whether to process passages instead of sentences, defaults to True
    :type passage_wise_processing: bool
    :param individual_correlations: Whether to also calculate correlations for individual subjects, defaults to False
    :type individual_correlations: bool
    :param output_dir: Output directory where the DataFrame with all correlations should be saved to
    :type output_dir: str
    :return: DataFrame with all correlations between ROIs and LMs
    :rtype: pd.DataFrame
    """
    # Define the text key (sentences or passages)
    text_key = "paragraphs" if passage_wise_processing else "sentences"

    # 1. Read in all the preprocessed fMRI data from all subjects from all ROIs
    all_subject_files = glob(os.path.join(pereira_input_path, "*.pkl"))
    if passage_wise_processing:
        all_subject_files = [i for i in all_subject_files if "passages" in i]
    else:
        all_subject_files = [i for i in all_subject_files if "passages" not in i]
    all_subject_ids = [
        file.split("/")[-1].replace("_fmri.pkl", "") for file in all_subject_files
    ]
    all_subject_data = {}

    # Iterate through the subjects
    for file, subject_id in zip(all_subject_files, all_subject_ids):
        logger.info(f"Processing subject {subject_id}")

        with open(file, "rb") as f:
            subject_data = pickle.load(f)  # noqa

        subject_data_no_sent = {
            i: subject_data[i][:num_sentences] for i in subject_data if i != text_key
        }

        # Iterate through all ROIs and get all fMRI data from all ROIs
        all_subject_data[subject_id] = {
            # Get the dissimilarity matrices for the ROIs
            k: pairwise_distances(v, v, metric=pairwise_metric)
            for k, v in subject_data_no_sent.items()
        }

    # Get the sentences or paragraphs (just use the last subject data file)
    sentences = subject_data[text_key][:num_sentences]
    sent_rdms = {}

    # 2. Load the data for the sentence embeddings for the different models
    for sent_model_name in sent_model_names:
        logger.info(f"Generating sentence embeddings for {sent_model_name}")

        # Some models need to be initialized differently as they are not compatible with huggingface
        if "sentence-transformers" in sent_model_name:
            model = SentenceTransformer(sent_model_name)
            encoded_sentences = np.stack(model.encode(sentences))
            del model
        # Initialize the other huggingface-based models here
        else:
            sent_model = SentenceEmbeddingModel(sent_model_name)
            embeddings = [t.numpy() for t in sent_model.encode_sentences(sentences)]
            encoded_sentences = np.stack(embeddings)
            del sent_model

        # Create the dissimilarity matrix for the sentence models
        sentences_dissim_mat = pd.DataFrame(
            pairwise_distances(
                encoded_sentences, encoded_sentences, metric=pairwise_metric
            ),
        )
        sent_rdms[sent_model_name] = sentences_dissim_mat

    # Get a vectorized representation of the similarity matrices
    sent_rdm_vectors = [
        get_sim_vector(sent_rdms[sent_model_name])
        for sent_model_name in sent_model_names
    ]

    # Generate a result matrix for all ROIs and all sent embed models
    result_df = pd.DataFrame(
        index=sent_model_names
    )  # columns=sorted(ROI_INDICES.keys())

    # Don't include the vision subnetworks and deal differently with language
    rois = {
        k
        for k in ROI_INDICES.keys()
        if k
        not in [
            "vision_object",
            "vision_face",
            "vision_scene",
            "vision_body",
            "language_lh",
            "language_rh",
        ]
    }
    rois.add("language")

    # Also create a dataframe for correlations to individual subjects
    ind_corr_df = pd.DataFrame(columns=["paradigm", "model", "roi", "correlation"])

    # 3. Calculate the correlation between each sentence embedding model and each ROI
    # (all subjects -> average corr matrix)
    for roi in sorted(rois):
        logger.info(f"Processing ROI: {roi}")
        if roi == "language":
            # Average across subjects and left/right in this case
            roi_specific_sim_mat = [
                all_subject_data[subj][r]
                for r in ["language_lh", "language_rh"]
                for subj in all_subject_ids
            ]
        else:
            roi_specific_sim_mat = [
                all_subject_data[subj][roi] for subj in all_subject_ids
            ]
        # Average correlation matrices across all subjects for a given ROI
        av_roi_specific_sim_mat = np.mean(roi_specific_sim_mat, axis=0)
        av_roi_specific_sim_vector = get_sim_vector(av_roi_specific_sim_mat)

        # Also calculate individual correlations if specified
        if individual_correlations:
            if roi == "language":
                # Average rh and lh in the case of language
                roi_specific_sim_mat = [
                    roi_specific_sim_mat[i] + roi_specific_sim_mat[i + 1] / 2
                    for i in range(0, len(roi_specific_sim_mat), 2)
                ]
            ind_corrs = [
                CORRELATION_MEASURES[correlation_metric](
                    sent_rdm_vector,
                    get_sim_vector(subj_roi_specific_sim_mat),
                )[0]
                for subj_roi_specific_sim_mat in roi_specific_sim_mat
                for sent_rdm_vector in sent_rdm_vectors
            ]
            temp_df = pd.DataFrame(
                {
                    "correlation": ind_corrs,
                    "model": np.repeat([sent_model_names], len(roi_specific_sim_mat)),
                    "paradigm": np.repeat(
                        list(map(SENT_EMBED_MODEL_PARADIGMS.get, sent_model_names)),
                        len(roi_specific_sim_mat),
                    ),
                    "roi": np.repeat(
                        [roi], len(sent_model_names) * len(roi_specific_sim_mat)
                    ),
                }
            )
            ind_corr_df = pd.concat([ind_corr_df, temp_df])

        # Spearman correlation to the average ROI correlation matrix for each sent model
        result_df.loc[:, roi] = [
            CORRELATION_MEASURES[correlation_metric](
                sent_rdm_vector,
                av_roi_specific_sim_vector,
            )[0]
            for sent_rdm_vector in sent_rdm_vectors
        ]

        # p-value for the Spearman correlation based on a permutation test
        perm_tests = [
            permutation_test(
                sent_rdm_vector.reshape(1, -1),
                permutation_type="pairings",
                statistic=lambda data: CORRELATION_MEASURES[correlation_metric](
                    data,
                    av_roi_specific_sim_vector,  # noqa
                )[0],
                n_resamples=n_resamples_permutation,
                alternative="greater",
            )
            for sent_rdm_vector in sent_rdm_vectors
        ]
        result_df.loc[:, roi + " p-value"] = [p.pvalue for p in perm_tests]
        result_df.loc[:, roi + " CI"] = [
            str(
                np.percentile(
                    p.null_distribution, [alpha / 2 * 100, (1 - alpha / 2) * 100]
                )
            )
            for p in perm_tests
        ]

    # Add an average column to average the correlations from all ROIs
    result_df["mean"] = result_df[
        [i for i in result_df.columns if "p-value" not in i and "CI" not in i]
    ].mean(axis=1)

    logger.info("Correlations between LMs and ROIs")
    logger.info(result_df)
    # Save the result
    result_df.to_csv(
        os.path.join(
            output_dir,
            f"rsa_correlations_{correlation_metric}_{pairwise_metric}_{text_key}.csv",
        )
    )

    # Also save individual correlation results if specified
    if individual_correlations:
        ind_corr_df.to_csv(
            os.path.join(
                output_dir,
                f"subj_rsa_correlations_{correlation_metric}_{pairwise_metric}_{text_key}.csv",
            )
        )

    return result_df


@click.command()
@click.option(
    "--permut-paragraph-input-path", type=str, default=PEREIRA_PERMUTED_SENTENCES_PATH
)
@click.option("--sent-models", multiple=True, default=SENT_EMBED_MODEL_LIST_EN)
@click.option(
    "--pairwise-metric", type=click.Choice(PAIRWISE_DISTANCES.keys()), default="cosine"
)
@click.option(
    "--correlation-metric",
    type=click.Choice(CORRELATION_MEASURES.keys()),
    default="spearman",
)
@click.option("--n-resamples", type=int, default=10000)
@click.option("--output-dir", type=str, default=PEREIRA_OUTPUT_DIR)
def perform_rsa_text_permutations(
    permut_paragraph_input_path: str = PEREIRA_PERMUTED_SENTENCES_PATH,
    sent_models: List[str] = SENT_EMBED_MODEL_LIST_EN,
    correlation_metric: str = "spearman",
    pairwise_metric: str = "cosine",
    n_resamples: int = 10000,
    output_dir: str = PEREIRA_OUTPUT_DIR,
) -> pd.DataFrame:
    """Perform text-based Representational Similarity Analysis (RSA) to test linguistic hypothesis.

    More specifically, this function is based on the method presented in Lepori and McCoy (2020). It is creating three
    different models based on the paragraphs from Pereira et al. (2018):
    A) A reference model (two middle sentences from each paragraph)
    B) A hypothesis model (full paragraphs, i.e., middle sentences + correct front/back sentence) representing the
    "contextual" sentence processing idea
    C) An alternative hypothesis model (two middle sentences + random front/back sentences from other paragraphs),
    representing the idea that context sentences do not make too much of a difference.

    :param permut_paragraph_input_path: Input path for the preprocessed paragraph data created with the
        generate_permuted_passages function in the preprocess_pereira script
    :type permut_paragraph_input_path: str
    :param sent_models: List of sentence embedding model names, defaults to SENT_EMBED_MODEL_LIST_EN
    :type sent_models: List[str]
    :param correlation_metric: Metric used to calculate the correlation between reference and (alt) hyp models
    :type correlation_metric: str
    :param pairwise_metric: Metric used for calculating the distance matrices (in-between a given (alt) hyp/ref model)
    :type pairwise_metric: str
    :param n_resamples: Number of resamples for the permutation test to determine significance for the correlations
    :type n_resamples: int
    :param output_dir: Output directory to save the results to
    :type output_dir: str
    :return: DataFrame with RSA correlation and p-values for each sentence embedding model and hypothesis-reference pair
    :rtype: pd.DataFrame
    """
    # Read the paragraph data
    permut_paragraphs = pd.read_csv(permut_paragraph_input_path, index_col=None)

    # Create the results df
    results = pd.DataFrame(index=["paragraphs", "permuted_sents"])

    # Iterate through each sentence embedding model
    for model_name in tqdm(
        sent_models, desc="Performing RSA for all sentence embedding models"
    ):
        # Create the model-specific results and p-value dictionary
        res = {}
        p_value = {}

        # Initialize the model
        sent_embed_model = SentenceEmbeddingModel(
            model_name=model_name,
            distance_measure=PAIRWISE_DISTANCES[pairwise_metric],
        )

        # Generate representational similarity matrices (RDMs): A) reference model (=2 center sentences)
        # These RDMs are converted into a vectorized representation using get_sim_vector
        ref_rdm = sent_embed_model.generate_rdm(permut_paragraphs["center_sents"])
        ref_model = get_sim_vector(ref_rdm)

        # B) hypothesis model (=two surrounding context sentences, "paragraphs") C) alternative hypothesis
        # (=two random sentences added to the front and back, "permuted_sents")
        for hypothesis in ["paragraphs", "permuted_sents"]:
            hyp_rdm = sent_embed_model.generate_rdm(permut_paragraphs[hypothesis])
            hyp_model = get_sim_vector(hyp_rdm)
            res[hypothesis] = CORRELATION_MEASURES[correlation_metric](
                ref_model, hyp_model
            )[0]
            # Calculate the significance of the correlation based on a permutation test
            p_value[hypothesis] = permutation_test(
                ref_model.reshape(1, -1),
                permutation_type="pairings",
                statistic=lambda data: CORRELATION_MEASURES[correlation_metric](
                    data,
                    hyp_model,  # noqa
                )[0],
                n_resamples=n_resamples,
                alternative="greater",
            ).pvalue

        # Append the correlation results and p-values for both hypotheses to the results df
        results[model_name] = pd.Series(res)
        results[model_name + " p-value"] = pd.Series(p_value)

    # Save the results to the output_dir
    results.to_csv(os.path.join(output_dir, "pereira_text_based_rsa.csv"))

    return results


@click.group()
def cli() -> None:
    """Perform a preliminary simple correlation analysis between fMRI data and LMs using the Pereira dataset."""


if __name__ == "__main__":
    cli.add_command(perform_rsa)
    cli.add_command(perform_rsa_text_permutations)
    cli()
