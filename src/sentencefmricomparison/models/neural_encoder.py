"""This script is performing neural encoding, i.e., the prediction of fMRI features from sentence embeddings."""

# Imports
import logging
import os
from typing import List

import click
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score
from tqdm import tqdm

from sentencefmricomparison.constants import PEREIRA_OUTPUT_DIR, SENT_EMBED_MODEL_LIST_EN
from sentencefmricomparison.models.sentence_embedding_base import SentenceEmbeddingModel
from sentencefmricomparison.utils import NEURAL_ENC_MODELS

# Initialize logger
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def pairwise_accuracy(
    estimator: BaseEstimator = None,
    X: torch.Tensor = None,  # noqa
    y: torch.Tensor = None,
) -> float:
    """Calculate the average pairwise accuracy of all pairs of true and predicted vectors.

    Based on the pairwise accuracy as defined in Oota et al. 2022, Sun et al. 2021, Pereira et al. 2018.

    :param estimator: Estimator object (e.g., a Ridge regression)
    :type estimator: BaseEstimator
    :param X: Sentence embeddings used as a basis to predict MRI vectors with the estimator
    :type X: torch.Tensor
    :param y: True MRI vectors
    :type y: torch.Tensor
    :return: Average pairwise accuracy from all possible sentence pairs
    :rtype: float
    """
    pred = estimator.predict(X)

    # See for all possible sentence pairings: Is the distance between the correct matches of predicted and X
    # sentences smaller than the distance between pairings of X and predicted vectors from different sentences?
    res = [
        cosine(pred[i], y[i]) + cosine(pred[j], y[j]) < cosine(pred[i], y[j]) + cosine(pred[j], y[i])
        for i in range(len(X)) for j in range(i + 1, len(X))
    ]

    # Return the fraction of instances for which the condition holds versus all possible pairs
    return sum(res) / len(res)


def pearson_scoring(
    estimator: BaseEstimator = None,
    X: torch.Tensor = None,  # noqa
    y: torch.Tensor = None,
) -> float:
    """Calculate the average pearson correlation for the given set of true and predicted MRI vectors.

    :param estimator: Estimator object (e.g., a Ridge regression)
    :type estimator: BaseEstimator
    :param X: Sentence embeddings used as a basis to predict MRI vectors with the estimator
    :type X: torch.Tensor
    :param y: True MRI vectors
    :type y: torch.Tensor
    :return: Average pearson correlation from all pairs of predicted and true MRI vectors
    :rtype: float
    """
    pred = estimator.predict(X)

    # See for all possible sentence pairings: Is the distance between the correct matches of predicted and X
    # sentences smaller than the distance between pairings of X and predicted vectors from different sentences?
    res = [pearsonr(t, p).statistic for t, p in zip(y, pred)]

    # Return the fraction of instances for which the condition holds versus all possible pairs
    return np.mean(res)


@click.command()
@click.option("--dataset-hf-name", type=str, default="helena-balabin/pereira_fMRI_passages")
@click.option("--sent-embed-models", type=str, multiple=True, default=SENT_EMBED_MODEL_LIST_EN)
@click.option("--region-based", type=bool, default=True)
@click.option("--sentence-key", type=str, default="")
@click.option("--mapping", type=str, default="ridge")
@click.option("--cv", type=int, default=5)
@click.option("--scoring", type=str, default="pairwise_accuracy")
@click.option("--output-dir", type=str, default=PEREIRA_OUTPUT_DIR)
def calculate_brain_scores_cv(
    dataset_hf_name: str = "helena-balabin/pereira_fMRI_passages",
    sent_embed_models: List[str] = SENT_EMBED_MODEL_LIST_EN,
    region_based: bool = True,
    sentence_key: str = "",
    mapping: str = "ridge",
    cv: int = 5,
    scoring: str = "pairwise_accuracy",
    output_dir: str = PEREIRA_OUTPUT_DIR,
) -> pd.DataFrame:
    """Calculates brain scores (neural encoding performances) averaged across cross-validation folds and subjects.

    :param dataset_hf_name: Name of the huggingface dataset used for the sentence/paragraph and MRI data, defaults to
        "helena-balabin/pereira_fMRI_passages"
    :type dataset_hf_name: str
    :param sent_embed_models: List of sentence embedding models used to encode the sentences
    :type sent_embed_models: List[str]
    :param region_based: Whether to calculate separate brain scores for each region of interest rather than the whole
        brain at once, defaults to True
    :type region_based: bool
    :param sentence_key: Optional key for the name of the text data column in the dataset
    :type sentence_key: str
    :param mapping: What kind of mapping model to use, defaults to "ridge"
    :type mapping: str
    :param cv: Number of folds used for cross-validation included in the neural encoding procedure
    :type cv: int
    :param scoring: Scoring function used to evaluate the performance of the neural encoding approach, defaults to
        "pairwise_accuracy"
    :type scoring: str
    :param output_dir: Output directory to save the brain scores to
    :type output_dir: str
    :return: DataFrame with mean values for the "brain score(s)" (neural encoding performance), either all regions of
        interest or all brain voxels, averaged across cross-validation folds and subjects
    :rtype: pd.DataFrame
    """
    # 1. Load the dataset from huggingface datasets
    dataset = load_dataset(dataset_hf_name)
    if len(sentence_key) == 0:
        sentence_key = "paragraphs" if "passages" in dataset_hf_name else "sentences"

    # 2. Initialize the regression model
    mapping_model = NEURAL_ENC_MODELS[mapping]

    # 3. Get the sentences/paragraphs
    sents = dataset["train"][0][sentence_key]
    # And try to encode them for each sentence embedding model
    sents_encoded_all_models = {}
    for model in sent_embed_models:
        try:
            sents_encoded_all_models[model] = torch.stack(SentenceEmbeddingModel(model).encode_sentences(sents))
        except OSError:
            # Some models might not be able to be initialized because they are part of a private repository requiring
            # an access token
            logger.info(f"{model} was skipped (likely due to being part of a private repository)")

    # 4. Set up the scoring function if needed
    scoring_func = None
    if "accuracy" in scoring or "2v2" in scoring:
        scoring_func = pairwise_accuracy
    if "pearson" in scoring:
        scoring_func = pearson_scoring
    # 5. Initialize the results
    results = []

    # 6. Get the scores for each model based on testing on each subject in a cross-validated manner
    for model, sents_encoded in tqdm(
        sents_encoded_all_models.items(),
        desc="Iterating through all sentence embedding models",
    ):
        # Create a result df that saves the results per subject first
        subj_results = []
        for i, subj in enumerate(dataset["train"]):
            # 6.1 Get the MRI features, perform either separate regression for each region of interest or all voxels
            # and get the respective cross validation scores
            if region_based:
                logger.info(f"{model}: Processing subj. {i + 1}/{len(dataset['train'])}")

                # Only use "vision" rather than all the vision sub-networks
                brain_rois = {
                    k: torch.tensor(v) for k, v in subj.items() if k not in
                    [sentence_key, "all", "vision_object", "vision_face", "vision_scene", "vision_body"]
                }

                roi_results = {}
                for roi_name, roi_features in brain_rois.items():
                    # Average across cross-validated results
                    brain_score = np.mean(
                        cross_val_score(
                            mapping_model,
                            sents_encoded.to("cpu").numpy(),
                            roi_features.to("cpu").numpy(),
                            cv=cv,
                            scoring=scoring_func or scoring,
                        )
                    )
                    roi_results[roi_name] = [brain_score]

                # Average results from language lh and language rh
                roi_results["language"] = np.mean([roi_results["language_lh"], roi_results["language_rh"]])
                roi_results.pop("language_lh")
                roi_results.pop("language_rh")

                subj_results.append(pd.DataFrame(roi_results))

            else:
                brain_voxels = torch.tensor(subj["all"])
                brain_score = np.mean(
                    cross_val_score(
                        mapping_model,
                        sents_encoded.to("cpu"),
                        brain_voxels.to("cpu"),
                        scoring=scoring_func or scoring,
                    )
                )
                subj_results.append(pd.DataFrame({"all": [brain_score]}))

        # 6.2 Calculate the average across all subjects for each sentence embedding model, append to the overall results
        results.append(pd.DataFrame(pd.concat(subj_results).mean(axis=0), columns=[model]).T)

    # Concatenate into an overall results dataframe
    results = pd.DataFrame(pd.concat(results))
    # And add a mean column if the scores are calculated region-wise
    if region_based:
        results["mean"] = results.mean(axis=1)

    # 7. Save the averaged results
    results.to_csv(
        os.path.join(
            output_dir,
            f"pereira_neural_enc_{dataset_hf_name.split('_')[-1]}_{scoring}_{mapping}.csv",
        ),
    )

    return results


@click.group()
def cli() -> None:
    """
    This is a preprocessing script for the preprocessed Pereira MATLAB files.
    """


if __name__ == "__main__":
    cli.add_command(calculate_brain_scores_cv)
    cli()
