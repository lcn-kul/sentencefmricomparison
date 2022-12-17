"""Base class for sentence embedding paradigms/models that are used."""

from typing import Callable, List, Optional, Union

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from sentencefmricomparison.constants import HF_TOKEN_PRED_BERT, INFERENCE_BATCH_SIZE, SENTENCE_EMBED_DEFAULT_EN
from sentencefmricomparison.utils import PAIRWISE_DISTANCES, POOLING_STRATEGIES


class SentenceEmbeddingModel:
    """Base class for all sentence embedding models."""

    model: Union[PreTrainedModel, SentenceTransformer]
    tokenizer: Optional[PreTrainedTokenizer]
    distance_measure: Callable = PAIRWISE_DISTANCES["cosine"]
    inference_batch_size: int = 4

    def __init__(
        self,
        model_name: str = SENTENCE_EMBED_DEFAULT_EN,
        distance_measure: Callable = PAIRWISE_DISTANCES["cosine"],
        pooling_strategy: str = POOLING_STRATEGIES["avg"],
        inference_batch_size: int = INFERENCE_BATCH_SIZE,
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
        """
        # Some models don't offer a specific tokenizer, use the one of the SENTENCE_EMBED_DEFAULT in that case
        if "sentence-transformers" in model_name:
            # Some sentence transformers don't have a hf transformers tokenizer and model, use the sentence-transformers
            # package with its encode function in that case instead
            self.tokenizer = None
            self.model = SentenceTransformer(model_name)
        # Use the authentification token for the Pred-BERT based models
        elif "vgaraujov" in model_name:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HF_TOKEN_PRED_BERT)
            self.model = AutoModel.from_pretrained(model_name, use_auth_token=HF_TOKEN_PRED_BERT)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)

            # Add a pseudo padding token to GPT2 (it's only needed to ensure a consistent output shape and it's
            # disregarded when averaging token embeddings)
            if model_name == "gpt2":
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

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
                # Use the SentenceTransformer syntax if there is no tokenizer
                with torch.no_grad():
                    output = self.pooling_stategy(
                        self.model.encode(batch, output_value="token_embeddings", convert_to_numpy=False)
                    ).to('cpu')

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
