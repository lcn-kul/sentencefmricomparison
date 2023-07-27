"""This script is performing neural encoding, i.e., the prediction of fMRI features from sentence embeddings."""

# Imports
import logging
import os
from typing import Dict, List, Union

import click
import numpy as np
import optuna
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.linear_model import LinearRegression, Ridge
from tqdm import tqdm

from sentencefmricomparison.constants import (
    PEREIRA_OUTPUT_DIR,
    SENT_EMBED_MODEL_LIST_EN,
)
from sentencefmricomparison.models.sentence_embedding_base import SentenceEmbeddingModel
from sentencefmricomparison.utils import cross_val_score_with_topic_ids, pairwise_accuracy, pearson_scoring

# Initialize logger
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def calculate_brain_scores_cv(
    dataset_hf_name: str = "helena-balabin/pereira_fMRI_passages",
    sent_embed_models: List[str] = SENT_EMBED_MODEL_LIST_EN,
    region_based: bool = True,
    combine_lh_rh: bool = True,
    rois: List[str] = ["dmn", "task", "vision", "language_lh", "language_rh"],  # noqa
    sentence_key: str = "",
    only_middle: bool = False,
    permuted_paragraphs: bool = False,
    mapping: str = "ridge",
    mapping_params: Union[Dict[str, Union[str, int, float]], None] = None,  # noqa
    cv: int = 5,
    scoring: str = "pairwise_accuracy",
    scoring_variation: str = None,  # type: ignore
    write_output: bool = True,
    output_dir: str = PEREIRA_OUTPUT_DIR,
) -> Union[pd.DataFrame, float]:
    """Calculate brain scores (neural encoding performances) averaged across cross-validation folds and subjects.

    :param dataset_hf_name: Name of the huggingface dataset used for the sentence/paragraph and MRI data, defaults to
        "helena-balabin/pereira_fMRI_passages"
    :type dataset_hf_name: str
    :param sent_embed_models: List of sentence embedding models used to encode the sentences
    :type sent_embed_models: List[str]
    :param rois: List of brain regions of interest to calculate brain scores for
    :type rois: List[str]
    :param region_based: Whether to calculate combine brain scores for each region of interest rather than the whole
        brain at once, defaults to True
    :type region_based: bool
    :param combine_lh_rh: Whether to calculate combine brain scores for the left and right hemisphere of the language
        network, defaults to False
    :param sentence_key: Optional key for the name of the text data column in the dataset
    :type sentence_key: str
    :param only_middle: Whether to only use the middle two sentences instead of all four (only works for paragraphs),
        defaults to False
    :type only_middle: bool
    :param permuted_paragraphs: Perform analysis on permuted paragraphs, defaults to False
    :type permuted_paragraphs: bool
    :param mapping: What kind of mapping model to use, defaults to "ridge"
    :type mapping: str
    :param mapping_params: Hyperparameters for the mapping model
    :type mapping_params: Dict[str, Union[str, int, float]]
    :param cv: Number of folds used for cross-validation included in the neural encoding procedure
    :type cv: int
    :param scoring: Scoring function used to evaluate the performance of the neural encoding approach, defaults to
        "pairwise_accuracy"
    :type scoring: str
    :param scoring_variation: Variation of the scoring function, e.g., "same-topic" or "different-topic" for
        "pairwise_accuracy", defaults to None
    :type scoring_variation: str
    :param write_output: Whether to write the output to a file, defaults to True
    :type write_output: bool
    :param output_dir: Output directory to save the brain scores to
    :type output_dir: str
    :return: DataFrame with mean values for the "brain score(s)" (neural encoding performance), either all regions of
        interest or all brain voxels, averaged across cross-validation folds and subjects, or a single float for HPO
    :rtype: Union[pd.DataFrame, float]
    """
    # 1. Load the dataset from huggingface datasets
    dataset = load_dataset(dataset_hf_name)
    if len(sentence_key) == 0:
        sentence_key = "paragraphs" if "passages" in dataset_hf_name else "sentences"

    # 2. Initialize the regression model
    if mapping_params:
        mapping_model = Ridge(**mapping_params)
    elif mapping == "ridge":
        mapping_model = Ridge()
    else:
        mapping_model = LinearRegression()

    # 3. Get the sentences/paragraphs
    sents = dataset["train"][0][sentence_key]

    # Only use the middle two sentences here for an alternative analysis
    if only_middle:
        sents = [". ".join(s.split(". ")[1:3]) + "." for s in sents]
    # Or use permuted paragraphs here for an alternative analysis
    if permuted_paragraphs:
        sents = dataset["train"][0]["permuted_paragraphs"]

    # And try to encode them for each sentence embedding model
    sents_encoded_all_models = {}
    for model in sent_embed_models:
        sents_encoded_all_models[model] = torch.stack(
            SentenceEmbeddingModel(model).encode_sentences(sents)
        )

    # 4. Set up the scoring function if needed
    scoring_func = None
    if "accuracy" in scoring or "2v2" in scoring:
        scoring_func = pairwise_accuracy
    if "pearson" in scoring:
        scoring_func = pearson_scoring  # type: ignore
    # 5. Initialize the results
    res = []

    # 6. Get the scores for each model based on testing on each subject in a cross-validated manner
    for model, sents_encoded in tqdm(
        sents_encoded_all_models.items(),
        desc="Iterating through all sentence embedding models",
    ):
        # Create a result df that saves the results per subject first
        subj_results = []
        for i, subj in enumerate(dataset["train"]):
            # 6.1 Get the MRI features, perform either combine regression for each region of interest or all voxels
            # and get the respective cross validation scores
            if region_based:
                logger.info(
                    f"{model}: Processing subj. {i + 1}/{len(dataset['train'])}"
                )

                # Only use "vision" rather than all the vision sub-networks
                brain_rois = {k: torch.tensor(v) for k, v in subj.items() if k in rois}

                roi_results = {}
                for roi_name, roi_features in brain_rois.items():
                    # Average across cross-validated results
                    brain_score = np.mean(
                        cross_val_score_with_topic_ids(
                            mapping_model,
                            sents_encoded.to("cpu").numpy(),
                            roi_features.to("cpu").numpy(),
                            topic_ids=torch.tensor(dataset["train"][i]["topic_indices"]),
                            cv=cv,
                            scoring=scoring_func or scoring,  # type: ignore
                            scoring_variation=scoring_variation,
                        )
                    )
                    roi_results[roi_name] = [brain_score]

                if combine_lh_rh:
                    # Average results from language lh and language rh
                    roi_results["language"] = np.mean(
                        [roi_results["language_lh"], roi_results["language_rh"]]
                    )
                    roi_results.pop("language_lh")
                    roi_results.pop("language_rh")

                subj_results.append(pd.DataFrame(roi_results, index=[i]))

            else:
                brain_voxels = torch.tensor(subj["all"])
                brain_score = np.mean(
                    cross_val_score_with_topic_ids(
                        mapping_model,
                        sents_encoded.to("cpu").numpy(),
                        brain_voxels.to("cpu").numpy(),
                        topic_ids=torch.tensor(dataset["train"][i]["topic_indices"]),
                        cv=cv,
                        scoring=scoring_func or scoring,  # type: ignore
                        scoring_variation=scoring_variation,
                    )
                )
                subj_results.append(pd.DataFrame({"all": [brain_score]}))

        # 6.2 Calculate the average across all subjects for each sentence embedding model, append to the overall results
        res.append(
            pd.DataFrame(pd.concat(subj_results).mean(axis=0), columns=[model]).T
        )

    # Concatenate into an overall results dataframe
    results = pd.DataFrame(pd.concat(res))
    # And add a mean column if the scores are calculated region-wise
    if region_based:
        results["mean"] = results.mean(axis=1)

    # 7. Save the averaged results
    if scoring_variation:
        scoring = f"{scoring}_{scoring_variation}"
    if write_output:
        if only_middle:
            output_file = f"pereira_neural_enc_{dataset_hf_name.split('_')[-1]}_{scoring}_{mapping}_middle.csv"
        elif permuted_paragraphs:
            output_file = f"pereira_neural_enc_{dataset_hf_name.split('_')[-1]}_{scoring}_{mapping}_permuted.csv"
        else:
            output_file = f"pereira_neural_enc_{dataset_hf_name.split('_')[-1]}_{scoring}_{mapping}.csv"
        results.to_csv(os.path.join(output_dir, output_file))
        return results
    else:
        # For HPO, we are currently only interested in one numerical value (one model, one brain network)
        return results.iloc[0][0]


@click.command()
@click.option(
    "--dataset-hf-name", type=str, default="helena-balabin/pereira_fMRI_passages"
)
@click.option(
    "--sent-embed-models", type=str, multiple=True, default=SENT_EMBED_MODEL_LIST_EN
)
@click.option("--region-based", type=bool, default=True)
@click.option("--combine-lh-rh", type=bool, default=True)
@click.option("--sentence-key", type=str, default="")
@click.option("--only-middle", type=bool, default=False)
@click.option("--permuted-paragraphs", type=bool, default=False)
@click.option("--mapping", type=str, default="ridge")
@click.option("--cv", type=int, default=5)
@click.option("--scoring", type=str, default="pairwise_accuracy")
@click.option("--scoring-variation", type=str, default=None)
@click.option("--output-dir", type=str, default=PEREIRA_OUTPUT_DIR)
def calculate_brain_scores_cv_wrapper(
    dataset_hf_name: str = "helena-balabin/pereira_fMRI_passages",
    sent_embed_models: List[str] = SENT_EMBED_MODEL_LIST_EN,
    region_based: bool = True,
    combine_lh_rh: bool = True,
    sentence_key: str = "",
    only_middle: bool = False,
    permuted_paragraphs: bool = False,
    mapping: str = "ridge",
    cv: int = 5,
    scoring: str = "pairwise_accuracy",
    scoring_variation: str = None,  # type: ignore
    output_dir: str = PEREIRA_OUTPUT_DIR,
):
    """Use calculate_brain_scores_cv (wrapper function).

    :param dataset_hf_name: Name of the huggingface dataset used for the sentence/paragraph and MRI data, defaults to
    "helena-balabin/pereira_fMRI_passages"
    :type dataset_hf_name: str
    :param sent_embed_models: List of sentence embedding models used to encode the sentences
    :type sent_embed_models: List[str]
    :param region_based: Whether to calculate combine brain scores for each region of interest rather than the whole
        brain at once, defaults to True
    :type region_based: bool
    :param combine_lh_rh: Whether to calculate combine brain scores for the left and right hemisphere for the language
        network, defaults to True
    :type combine_lh_rh: bool
    :param sentence_key: Optional key for the name of the text data column in the dataset
    :type sentence_key: str
    :param only_middle: Whether to only use the middle two sentences instead of all four (only works for paragraphs),
        defaults to False
    :type only_middle: bool
    :param permuted_paragraphs: Perform analysis on permuted paragraphs, defaults to False
    :type permuted_paragraphs: bool
    :param mapping: What kind of mapping model to use, defaults to "ridge"
    :type mapping: str
    :param cv: Number of folds used for cross-validation included in the neural encoding procedure
    :type cv: int
    :param scoring: Scoring function used to evaluate the performance of the neural encoding approach, defaults to
        "pairwise_accuracy"
    :type scoring: str
    :param scoring_variation: Optional variation of the scoring function, defaults to None
    :type scoring_variation: str
    :param output_dir: Output directory to save the brain scores to
    :type output_dir: str
    """
    calculate_brain_scores_cv(
        dataset_hf_name=dataset_hf_name,
        sent_embed_models=sent_embed_models,
        sentence_key=sentence_key,
        combine_lh_rh=combine_lh_rh,
        only_middle=only_middle,
        permuted_paragraphs=permuted_paragraphs,
        mapping=mapping,
        region_based=region_based,
        cv=cv,
        scoring=scoring,
        scoring_variation=scoring_variation,
        write_output=True,
        output_dir=output_dir,
    )


def objective(
    optuna_trial: optuna.trial.Trial,
) -> float:
    """Objective function for the hyperparameter optimization."""
    # Define the hyperparameters to optimize
    alpha = optuna_trial.suggest_categorical("alpha", [0, 0.5, 1.0, 2.0])
    max_iter = optuna_trial.suggest_categorical("max_iter", [100, 500, 1000])
    tol = optuna_trial.suggest_categorical("tol", [1e-3, 1e-4, 1e-5])

    # Calculate a single brain score for the given hyperparameters
    score = calculate_brain_scores_cv(
        dataset_hf_name="helena-balabin/pereira_fMRI_passages",
        sent_embed_models=["gpt2"],
        rois=["language_rh", "language_lh"],
        mapping="ridge",
        mapping_params={"alpha": alpha, "max_iter": max_iter, "tol": tol},
        write_output=False,
    )

    return score


@click.command()
@click.option(
    "--output_dir",
    "-o",
    default=PEREIRA_OUTPUT_DIR,
    help="The output directory to save the HPO results to.",
)
def hpo_neural_encoder(
    output_dir: str = PEREIRA_OUTPUT_DIR,
):
    """Hyperparameter optimization for the neural encoder.

    :param output_dir: The output directory to save the results to.
    :type output_dir: str
    """
    # Define the search space
    search_space = {
        "alpha": [0, 0.5, 1.0, 2.0],
        "max_iter": [100, 500, 1000],
        "tol": [1e-3, 1e-4, 1e-5],
    }
    # Create the study and optimize
    study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space))
    study.optimize(objective, n_trials=48)

    logger.info(f"All trial results: {study.trials_dataframe()}")

    # Save the results
    study.trials_dataframe().to_csv(
        os.path.join(output_dir, "pereira_neural_enc_hpo.csv")
    )


@click.group()
def cli() -> None:
    """Use the neural encoder."""


if __name__ == "__main__":
    cli.add_command(calculate_brain_scores_cv_wrapper)
    cli.add_command(hpo_neural_encoder)
    cli()
