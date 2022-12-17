# -*- coding: utf-8 -*-

"""Constants."""

import os

from dotenv import load_dotenv

HERE = os.path.dirname(os.path.realpath(__file__))
SOURCE_DIR = os.path.join(os.path.abspath(os.path.join(HERE, os.pardir)))
PROJECT_DIR = os.path.join(os.path.abspath(os.path.join(HERE, os.pardir)))

# Directory for data, logs, models, notebooks
DATA_DIR = os.path.join(os.sep.join(PROJECT_DIR.split(os.sep)[:-1]), "data")
LOGS_DIR = os.path.join(os.sep.join(PROJECT_DIR.split(os.sep)[:-1]), "logs")
MODELS_DIR = os.path.join(os.sep.join(PROJECT_DIR.split(os.sep)[:-1]), "models")
NOTEBOOKS_DIR = os.path.join(os.sep.join(PROJECT_DIR.split(os.sep)[:-1]), "notebooks")

# Sub-directories of data
RAW_DIR = os.path.join(DATA_DIR, "raw")
INPUT_DIR = os.path.join(DATA_DIR, "input")
OUTPUT_DIR = os.path.join(DATA_DIR, "output")
MISC_DIR = os.path.join(DATA_DIR, "misc")
# Dirs/paths for Pereira et al. 2018 data
PEREIRA_RAW_DIR = os.path.join(RAW_DIR, "pereira")
PEREIRA_INPUT_DIR = os.path.join(INPUT_DIR, "pereira")
PEREIRA_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "pereira")
PEREIRA_SENTENCE_DIR = os.path.join(PEREIRA_INPUT_DIR, "sentences")
PEREIRA_EXAMPLE_FILE = os.path.join(PEREIRA_RAW_DIR, "P01/data_384sentences.mat")
PEREIRA_PREPROCESSED_PATH = os.path.join(PEREIRA_INPUT_DIR, "pereira_preprocessed.csv")
PEREIRA_PERMUTED_SENTENCES_PATH = os.path.join(PEREIRA_INPUT_DIR, "pereira_permuted_passages.csv")
# Use dotenv to properly load the device-dependent mlflow tracking URI
load_dotenv()
# Load a constant for distinguishing between local and cluster execution (default = True)
LOCAL_EXECUTION = os.getenv("LOCAL_EXECUTION") or "True"
# Constant for the maximum batch size that should be used during inference
INFERENCE_BATCH_SIZE = int(os.getenv("INFERENCE_BATCH_SIZE")) or 4
# Huggingface access token for the Pred-based sentence embedding models
HF_TOKEN_PRED_BERT = os.getenv("HF_TOKEN_PRED_BERT")
# Directory for saving large datasets or language models
LARGE_DATASET_STORAGE_PATH = os.path.join(
    os.getenv("LARGE_STORAGE_PATH") or "~/.cache/huggingface", "datasets"
)
LARGE_MODELS_STORAGE_PATH = os.path.join(
    os.getenv("LARGE_STORAGE_PATH") or "~/.cache/huggingface", "transformers"
)

# Sent-eval benchmark constants
PATH_TO_SENTEVAL = os.getenv("PATH_TO_SENTEVAL") or './SentEval'
PATH_TO_SENTEVAL_DATA = os.path.join(PATH_TO_SENTEVAL, "data")
SENTEVAL_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "senteval")

# Create directories
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PEREIRA_RAW_DIR, exist_ok=True)
os.makedirs(PEREIRA_INPUT_DIR, exist_ok=True)
os.makedirs(PEREIRA_OUTPUT_DIR, exist_ok=True)
os.makedirs(SENTEVAL_OUTPUT_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(NOTEBOOKS_DIR, exist_ok=True)

# Sentence embedding default
SENTENCE_EMBED_DEFAULT_EN = "sentence-transformers/stsb-roberta-large"

# List of sentence embedding models to test for RSA
# in English
SENT_EMBED_MODEL_LIST_EN = [
    # word averaging
    "sentence-transformers/average_word_embeddings_glove.6B.300d",  # GloVe averaging
    "roberta-large",  # RoBERTa-large averaging
    "gpt2",  # GPT-2 averaging
    # pragmatic coherence
    "helena-balabin/qt-roberta-large",  # QT based on RoBERTa-large
    "vgaraujov/PredBERT-T",  # Alteratives are: vgaraujov/PredBERT-T, vgaraujov/PredALBERT-T, vgaraujov/PredALBERT-R
    # TODO skipthoughts?
    # TODO switch quickthoughts to the original model
    # semantic comparison
    "sentence-transformers/roberta-large-nli-stsb-mean-tokens",  # S-BERT based on RoBERTa-large
    "princeton-nlp/sup-simcse-roberta-large",  # SUPERVISED SimCSE based on RoBERTa-large
    "sentence-transformers/sentence-t5-base"  # Sentence-T5 based on T5 (fine-tuned on QA + NLI)
    # contrastive learning
    "princeton-nlp/unsup-simcse-roberta-large",  # UNSUPERVISED SimCSE based on RoBERTa-large
    "voidism/diffcse-roberta-base-sts",  # DiffCSE based on RoBERTa-base
    "johngiorgi/declutr-base",  # DeCLUTR based on RoBERTa-base
]

# Create a custom color palette
CUSTOM_COLOR_PALETTE = [
    "#FFC9B5",
    "#B2CEDE",
    "#8CDFD6",
    "#6DC0D5",
    "#416788",
    "#DBAD6A",
    "#C44536",
    "#A24B64",
    "#15616D",
    "#993955"
]
