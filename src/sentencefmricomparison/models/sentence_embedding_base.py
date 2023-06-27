"""Base class for sentence embedding paradigms/models that are used."""

import os
from typing import Callable, Dict, List, Optional, Union

import pandas as pd
import pickle as pkl
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer, GPT2Config, GPT2Model, PreTrainedModel, PreTrainedTokenizer

from sentencefmricomparison.constants import (
    GPT3_EMBEDS_PATH,
    INFERENCE_BATCH_SIZE,
    SENTENCE_EMBED_DEFAULT_EN,
    SKIPTHOUGHTS_MODEL_DIR,
)
from sentencefmricomparison.utils import PAIRWISE_DISTANCES, POOLING_STRATEGIES


class SentenceEmbeddingModel:
    """Base class for all sentence embedding models."""

    model: Union[PreTrainedModel, SentenceTransformer, Dict]
    model_name: str
    tokenizer: Optional[PreTrainedTokenizer]
    distance_measure: Callable = PAIRWISE_DISTANCES["cosine"]
    inference_batch_size: int = 4

    def __init__(
        self,
        model_name: str = SENTENCE_EMBED_DEFAULT_EN,
        distance_measure: Callable = PAIRWISE_DISTANCES["cosine"],
        pooling_strategy: str = POOLING_STRATEGIES["avg"],
        inference_batch_size: int = INFERENCE_BATCH_SIZE,
        gpt3_embed_path: str = GPT3_EMBEDS_PATH,
    ):
        """Initialize a semantic comparison sentence embedding model.

        The pooling strategy is to take the regular pooled_output of the model.

        :param model_name: Name of the model that generates the token embeddings,
                defaults to 'princeton-nlp/unsup-simcse-roberta-large'
        :type model_name: str
        :param distance_measure: Similarity measure used for pairwise similairties in this model
        :type distance_measure: Callable
        :param inference_batch_size: Batch size used during inference (obtaining sentence embeddings)
        :type inference_batch_size: int
        :param gpt3_embed_path: Path to the precomputed GPT-3 embeddings (only relevant to GPT-3), defaults to None
        :type gpt3_embed_path: str, optional
        """
        # Specify the model type
        self.model_name = model_name

        # Some models don't offer a specific tokenizer, use the one of the SENTENCE_EMBED_DEFAULT in that case
        if "sentence-transformers" in self.model_name:
            # Some sentence transformers don't have a hf transformers tokenizer and model, use the sentence-transformers
            # package with its encode function in that case instead
            self.tokenizer = None
            self.model = SentenceTransformer(self.model_name)
        elif "gpt3" in self.model_name:
            self.tokenizer = None
            # For GPT-3, the "model" is just a dictionary of precomputed embeddings for all possible sentences in this
            # analysis
            with open(gpt3_embed_path, "rb") as f:
                self.model = {k.strip(): v for k,v in pkl.load(f).items()}
        # Skip-Thoughts setup
        elif self.model_name == "skipthoughts":
            from skip_thoughts import configuration
            from skip_thoughts.encoder_manager import EncoderManager

            self.tokenizer = None
            self.model = EncoderManager()
            self.model.load_model(
                configuration.model_config(),
                vocabulary_file=os.path.join(SKIPTHOUGHTS_MODEL_DIR, "vocab.txt"),
                embedding_matrix_file=os.path.join(SKIPTHOUGHTS_MODEL_DIR, "embeddings.npy"),
                checkpoint_path=os.path.join(SKIPTHOUGHTS_MODEL_DIR, "model.ckpt-501424"),
            )
        # Setup for a random GPT-2 based baseline
        elif self.model_name == "gpt2-random":
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            config = GPT2Config()
            self.model = GPT2Model(config)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)

        # Add a pseudo padding token to GPT2 (it's only needed to ensure a consistent output shape and it's
        # disregarded when averaging token embeddings)
        if "gpt2" in self.model_name:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.pooling_stategy = pooling_strategy
        self.distance_measure = distance_measure
        self.inference_batch_size = inference_batch_size

    def encode_sentences(
        self,
        sentences: Union[List[str], pd.Series],
    ) -> List[torch.tensor]:
        """Encode each (multi)sentence into an embedding vector.

        :param sentences: List of (multi)sentences to be converted into embeddings
        :type sentences: Union[List[str], pd.Series]
        :return: List of sentence embeddings
        :rtype: List[torch.tensor]
        """
        encoded_sentences = []
        sentence_dl = torch.utils.data.DataLoader(
            sentences, batch_size=self.inference_batch_size
        )

        for batch in sentence_dl:
            # Tokenize the sentences
            if self.tokenizer:
                inputs = self.tokenizer(
                    batch,
                    padding="max_length",
                    truncation=True,
                    max_length=self.model.config.max_length,
                    return_tensors="pt",
                )
                # Place the batch on the GPU if possible (if the model is on the GPU)
                inputs = inputs.to(self.model.device)
                with torch.no_grad():
                    output = self.pooling_stategy(
                        self.model(**inputs), inputs["attention_mask"]
                    ).to('cpu')
            else:
                # Use the SentenceTransformer/SkipThoughts syntax if there is no tokenizer
                with torch.no_grad():
                    if "sentence-transformers" in self.model_name:
                        output = self.pooling_stategy(
                            self.model.encode(
                                batch,
                                output_value="token_embeddings",
                                convert_to_numpy=False,
                            )
                        ).to('cpu')
                    # SkipThoughts
                    elif self.model_name == "skipthoughts":
                        output = torch.from_numpy(self.model.encode(batch))
                    # GPT-3
                    else:
                        # Use the pre-computed embeddings for GPT-3
                        output = [self.model[sent.strip()] for sent in batch]

            encoded_sentences += [sent_embed for sent_embed in output]

        return encoded_sentences

    def generate_rdm(
        self,
        sentences: Union[List[str], pd.Series],
    ) -> pd.DataFrame:
        """Generate a similarity matrix for a given list of (multi)sentences.

        :param sentences: List of (multi)sentences for which to calculate pairwise similarities
        :type sentences: Union[List[str], pd.Series]
        :return: DataFrame with pairwise sentence similarities
        :rtype: pd.DataFrame
        """
        # 1. Get the sentence embedding vectors
        encoded_sentences = torch.stack(self.encode_sentences(sentences))
        # 2. Create the similarity matrix
        sim_matrix = pd.DataFrame(
            self.distance_measure(encoded_sentences).to("cpu").numpy()
        )
        return sim_matrix
