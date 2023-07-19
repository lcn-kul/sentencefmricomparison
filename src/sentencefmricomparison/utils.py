"""Additional functions used in this repository."""

from dataclasses import make_dataclass
from itertools import combinations, permutations, product
from typing import Iterable, List, Optional, Union

import numpy as np
import torch
from scipy._lib._util import check_random_state  # noqa
from scipy.special import comb, factorial
from scipy.stats import kendalltau, pearsonr, spearmanr
from torchmetrics.functional.pairwise import (
    pairwise_cosine_similarity,
    pairwise_euclidean_distance,
    pairwise_manhattan_distance,
)
from transformers.file_utils import ModelOutput


def mean_pooling(
    model_output: Union[ModelOutput, List[torch.tensor]],
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply Mean Pooling - Take attention mask into account for correct averaging.

    Taken from: https://huggingface.co/sentence-transformers/stsb-xlm-r-multilingual
    :param model_output: Model output
    :type model_output: Union[ModelOutput, List[torch.tensor]]
    :param attention_mask: Attention mask
    :type attention_mask: Optional[torch.Tensor]
    :return: Mean pooled output
    :rtype: torch.Tensor
    """
    if isinstance(model_output, ModelOutput):
        token_embeddings = model_output[
            0
        ]  # First element of model_output contains all token embeddings
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()  # type: ignore
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
    else:
        # Deal with the alternative SentenceTransformer encode output (list of tensors, no attention mask)
        return torch.stack(
            [torch.mean(sent_embed, dim=0) for sent_embed in model_output]
        )


def max_pooling(
    model_output: ModelOutput,  # noqa
    attention_mask: torch.Tensor,  # noqa
):
    """Apply Max Pooling."""
    return NotImplementedError()


def cls_pooling(
    model_output: ModelOutput,  # noqa
    attention_mask: torch.Tensor,  # noqa
):
    """Apply [CLS] Pooling."""
    return NotImplementedError()


def pairwise_cosine_distance(
    embeds: torch.tensor,
    **kwargs,
) -> torch.tensor:
    """Convert cosine similarities into cosine distances.

    :param embeds: Embeddings to calculate pairwise cosine distances from
    :type embeds: torch.tensor
    :param kwargs: Other parameters passed to the underlying pairwise_cosine_similarity function (torch)
    :type kwargs: Dict
    :return: Tensor of cosine distances
    :rtype: torch.tensor
    """
    similarity = pairwise_cosine_similarity(embeds, zero_diagonal=False, **kwargs).to(
        "cpu"
    )
    distance = torch.ones(size=similarity.shape).to("cpu") - similarity
    return distance


POOLING_STRATEGIES = {
    "max": max_pooling,
    "avg": mean_pooling,
    "cls": cls_pooling,
}

PAIRWISE_DISTANCES = {
    "cosine": pairwise_cosine_distance,
    "euclidean": pairwise_euclidean_distance,
    "manhattan": pairwise_manhattan_distance,
}

CORRELATION_MEASURES = {
    "spearman": spearmanr,
    "pearson": pearsonr,
    "kendalltau": kendalltau,
}

# The following code is taken from scipy 1.9.3 (https://github.com/scipy/scipy/tree/v1.9.3/scipy/stats),
# which is needed for this project due to its permutation test. But it would unfortunately cause dependency conflicts
# with other packages used, that is why the functions are added directly here


def _broadcast_arrays(arrays, axis=None):
    """Broadcast shapes of arrays, ignoring incompatibility of specified axes."""
    new_shapes = _broadcast_array_shapes(arrays, axis=axis)
    if axis is None:
        new_shapes = [new_shapes] * len(arrays)
    return [
        np.broadcast_to(array, new_shape)
        for array, new_shape in zip(arrays, new_shapes)
    ]


def _broadcast_array_shapes(arrays, axis=None):
    """Broadcast shapes of arrays, ignoring incompatibility of specified axes."""
    shapes = [np.asarray(arr).shape for arr in arrays]
    return _broadcast_shapes(shapes, axis)


def _broadcast_shapes(shapes, axis=None):
    """Broadcast shapes, ignoring incompatibility of specified axes."""
    if not shapes:
        return shapes

    # input validation
    if axis is not None:
        axis = np.atleast_1d(axis)
        axis_int = axis.astype(int)
        if not np.array_equal(axis_int, axis):
            raise np.AxisError(
                "`axis` must be an integer, a " "tuple of integers, or `None`."
            )
        axis = axis_int

    # First, ensure all shapes have same number of dimensions by prepending 1s.
    n_dims = max([len(shape) for shape in shapes])
    new_shapes = np.ones((len(shapes), n_dims), dtype=int)
    for row, shape in zip(new_shapes, shapes):
        row[len(row) - len(shape):] = shape  # can't use negative indices (-0:)

    removed_shapes = None
    # Remove the shape elements of the axes to be ignored, but remember them.
    if axis is not None:
        axis[axis < 0] = n_dims + axis[axis < 0]
        axis = np.sort(axis)
        if axis[-1] >= n_dims or axis[0] < 0:
            message = f"`axis` is out of bounds " f"for array of dimension {n_dims}"
            raise np.AxisError(message)

        if len(np.unique(axis)) != len(axis):
            raise np.AxisError("`axis` must contain only distinct elements")

        removed_shapes = new_shapes[:, axis]
        new_shapes = np.delete(new_shapes, axis, axis=1)

    # If arrays are broadcastable, shape elements that are 1 may be replaced
    # with a corresponding non-1 shape element. Assuming arrays are
    # broadcastable, that final shape element can be found with:
    new_shape = np.max(new_shapes, axis=0)
    # except in case of an empty array:
    new_shape *= new_shapes.all(axis=0)

    # Among all arrays, there can only be one unique non-1 shape element.
    # Therefore, if any non-1 shape element does not match what we found
    # above, the arrays must not be broadcastable after all.
    if np.any(~((new_shapes == 1) | (new_shapes == new_shape))):
        raise ValueError("Array shapes are incompatible for broadcasting.")

    if axis is not None:
        # Add back the shape elements that were ignored
        new_axis = axis - np.arange(len(axis))
        new_shapes = [
            tuple(np.insert(new_shape, new_axis, removed_shape))
            for removed_shape in removed_shapes  # noqa: E231
        ]
        return new_shapes
    else:
        return tuple(new_shape)


def _broadcast_concatenate(arrays, axis):
    """Concatenate arrays along an axis with broadcasting."""
    arrays = _broadcast_arrays(arrays, axis)
    res = np.concatenate(arrays, axis=axis)
    return res


def _vectorize_statistic(statistic):
    """Vectorize an n-sample statistic."""
    # This is a little cleaner than np.nditer at the expense of some data copying: concatenate samples together,
    # then use np.apply_along_axis
    def stat_nd(*data, axis=0):
        """Broadcast the samples and apply the statistic."""
        lengths = [sample.shape[axis] for sample in data]
        split_indices = np.cumsum(lengths)[:-1]
        z = _broadcast_concatenate(data, axis)

        # move working axis to position 0 so that new dimensions in the output
        # of `statistic` are _prepended_. ("This axis is removed, and replaced
        # with new dimensions...")
        z = np.moveaxis(z, axis, 0)

        def stat_1d(z):  # noqa
            """Apply the statistic to a 1-D array of concatenated samples."""
            data = np.split(z, split_indices)  # noqa
            return statistic(*data)

        return np.apply_along_axis(stat_1d, 0, z)[()]

    return stat_nd


attributes = ("statistic", "pvalue", "null_distribution")
PermutationTestResult = make_dataclass("PermutationTestResult", attributes)


def _all_partitions_concatenated(
    ns: Iterable[int],
):
    """Generate all partitions of indices of groups of given sizes.

    :param ns: Iterable of group sizes.
    :type ns: Iterable[int]
    :yield: All partitions of indices of groups of given sizes.
    :rtype: Iterator[np.ndarray]
    """

    def all_partitions(z, n):  # noqa
        """Generate all partitions of a set into two subsets of given sizes."""
        for c in combinations(z, n):
            x0 = set(c)
            x1 = z - x0
            yield [x0, x1]

    def all_partitions_n(z, ns):  # noqa
        """Generate all partitions of a set into subsets of given sizes."""
        if len(ns) == 0:
            yield [z]
            return
        for c in all_partitions(z, ns[0]):
            for d in all_partitions_n(c[1], ns[1:]):
                yield c[0:1] + d

    z = set(range(np.sum(ns)))
    for partitioning in all_partitions_n(z, ns[:]):  # type: ignore
        x = np.concatenate([list(partition) for partition in partitioning]).astype(int)
        yield x


def _batch_generator(iterable, batch):
    """Yield batches of elements from an iterable."""
    iterator = iter(iterable)
    if batch <= 0:
        raise ValueError("`batch` must be positive.")
    z = [item for i, item in zip(range(batch), iterator)]
    while z:  # we don't want StopIteration without yielding an empty list
        yield z
        z = [item for i, item in zip(range(batch), iterator)]


def _calculate_null_both(data, statistic, n_permutations, batch, random_state=None):
    """Calculate null distribution for independent sample tests."""
    n_samples = len(data)

    # compute number of permutations
    # (distinct partitions of data into samples of these sizes)
    n_obs_i = [sample.shape[-1] for sample in data]  # observations per sample
    n_obs_ic = np.cumsum(n_obs_i)
    n_obs = n_obs_ic[-1]  # total number of observations
    n_max = np.prod(
        [comb(n_obs_ic[i], n_obs_ic[i - 1]) for i in range(n_samples - 1, 0, -1)]
    )

    # perm_generator is an iterator that produces permutations of indices
    # from 0 to n_obs. We'll concatenate the samples, use these indices to
    # permute the data, then split the samples apart again.
    if n_permutations >= n_max:
        exact_test = True
        n_permutations = n_max
        perm_generator = _all_partitions_concatenated(n_obs_i)
    else:
        exact_test = False
        # Neither RandomState.permutation nor Generator.permutation
        # can permute axis-slices independently. If this feature is
        # added in the future, batches of the desired size should be
        # generated in a single call.
        perm_generator = (
            random_state.permutation(n_obs) for _ in range(n_permutations)
        )

    batch = batch or int(n_permutations)
    null_distribution = []

    # First, concatenate all the samples. In batches, permute samples with
    # indices produced by the `perm_generator`, split them into new samples of
    # the original sizes, compute the statistic for each batch, and add these
    # statistic values to the null distribution.
    data = np.concatenate(data, axis=-1)
    for indices in _batch_generator(perm_generator, batch=batch):
        indices = np.array(indices)

        # `indices` is 2D: each row is a permutation of the indices.
        # We use it to index `data` along its last axis, which corresponds
        # with observations.
        # After indexing, the second to last axis of `data_batch` corresponds
        # with permutations, and the last axis corresponds with observations.
        data_batch = data[..., indices]

        # Move the permutation axis to the front: we'll concatenate a list
        # of batched statistic values along this zeroth axis to form the
        # null distribution.
        data_batch = np.moveaxis(data_batch, -2, 0)
        data_batch = np.split(data_batch, n_obs_ic[:-1], axis=-1)
        null_distribution.append(statistic(*data_batch, axis=-1))
    null_distribution = np.concatenate(null_distribution, axis=0)

    return null_distribution, n_permutations, exact_test


def _calculate_null_pairings(data, statistic, n_permutations, batch, random_state=None):
    """Calculate null distribution for association tests."""
    n_samples = len(data)

    # compute number of permutations (factorial(n) permutations of each sample)
    n_obs_sample = data[0].shape[-1]  # observations per sample; same for each
    n_max = factorial(n_obs_sample) ** n_samples

    # `perm_generator` is an iterator that produces a list of permutations of
    # indices from 0 to n_obs_sample, one for each sample.
    if n_permutations >= n_max:
        exact_test = True
        n_permutations = n_max
        # cartesian product of the sets of all permutations of indices
        perm_generator = product(
            *(permutations(range(n_obs_sample)) for _ in range(n_samples))
        )
    else:
        exact_test = False
        # Separate random permutations of indices for each sample.
        # Again, it would be nice if RandomState/Generator.permutation
        # could permute each axis-slice separately.
        perm_generator = (
            [random_state.permutation(n_obs_sample) for _ in range(n_samples)]
            for _ in range(n_permutations)
        )

    batch = batch or int(n_permutations)
    null_distribution = []

    for indices in _batch_generator(perm_generator, batch=batch):
        indices = np.array(indices)

        # `indices` is 3D: the zeroth axis is for permutations, the next is
        # for samples, and the last is for observations. Swap the first two
        # to make the zeroth axis correspond with samples, as it does for
        # `data`.
        indices = np.swapaxes(indices, 0, 1)

        # When we're done, `data_batch` will be a list of length `n_samples`.
        # Each element will be a batch of random permutations of one sample.
        # The zeroth axis of each batch will correspond with permutations,
        # and the last will correspond with observations. (This makes it
        # easy to pass into `statistic`.)
        data_batch = [None] * n_samples
        for i in range(n_samples):
            data_batch[i] = data[i][..., indices[i]]
            data_batch[i] = np.moveaxis(data_batch[i], -2, 0)  # noqa

        null_distribution.append(statistic(*data_batch, axis=-1))
    null_distribution = np.concatenate(null_distribution, axis=0)

    return null_distribution, n_permutations, exact_test


def _calculate_null_samples(data, statistic, n_permutations, batch, random_state=None):
    """Calculate null distribution for paired-sample tests."""
    n_samples = len(data)

    # By convention, the meaning of the "samples" permutations type for
    # data with only one sample is to flip the sign of the observations.
    # Achieve this by adding a second sample - the negative of the original.
    if n_samples == 1:
        data = [data[0], -data[0]]

    # The "samples" permutation strategy is the same as the "pairings"
    # strategy except the roles of samples and observations are flipped.
    # So swap these axes, then we'll use the function for the "pairings"
    # strategy to do all the work!
    data = np.swapaxes(data, 0, -1)

    # (Of course, the user's statistic doesn't know what we've done here,
    # so we need to pass it what it's expecting.)
    def statistic_wrapped(*data, axis):  # noqa
        """Wrap `statistic` to swap the first two axes."""
        data = np.swapaxes(data, 0, -1)  # noqa
        if n_samples == 1:
            data = data[0:1]  # noqa
        return statistic(*data, axis=axis)

    return _calculate_null_pairings(
        data, statistic_wrapped, n_permutations, batch, random_state
    )


def _permutation_test_iv(
    data,
    statistic,
    permutation_type,
    vectorized,
    n_resamples,
    batch,
    alternative,
    axis,
    random_state,
):
    """Input validation for `permutation_test`."""
    axis_int = int(axis)
    if axis != axis_int:
        raise ValueError("`axis` must be an integer.")

    permutation_types = {"samples", "pairings", "independent"}
    permutation_type = permutation_type.lower()
    if permutation_type not in permutation_types:
        raise ValueError(f"`permutation_type` must be in {permutation_types}.")

    if vectorized not in {True, False}:
        raise ValueError("`vectorized` must be `True` or `False`.")

    if not vectorized:
        statistic = _vectorize_statistic(statistic)

    message = "`data` must be a tuple containing at least two samples"
    try:
        if len(data) < 2 and permutation_type == "independent":
            raise ValueError(message)
    except TypeError:
        raise TypeError(message)

    data = _broadcast_arrays(data, axis)
    data_iv = []
    for sample in data:
        sample = np.atleast_1d(sample)
        if sample.shape[axis] <= 1:
            raise ValueError(
                "each sample in `data` must contain two or more "
                "observations along `axis`."
            )
        sample = np.moveaxis(sample, axis_int, -1)
        data_iv.append(sample)

    n_resamples_int = int(n_resamples) if not np.isinf(n_resamples) else np.inf
    if n_resamples != n_resamples_int or n_resamples_int <= 0:
        raise ValueError("`n_resamples` must be a positive integer.")

    if batch is None:
        batch_iv = batch
    else:
        batch_iv = int(batch)
        if batch != batch_iv or batch_iv <= 0:
            raise ValueError("`batch` must be a positive integer or None.")

    alternatives = {"two-sided", "greater", "less"}
    alternative = alternative.lower()
    if alternative not in alternatives:
        raise ValueError(f"`alternative` must be in {alternatives}")

    random_state = check_random_state(random_state)

    return (
        data_iv,
        statistic,
        permutation_type,
        vectorized,
        n_resamples_int,
        batch_iv,
        alternative,
        axis_int,
        random_state,
    )


def permutation_test(
    data,
    statistic,
    *,
    permutation_type="independent",
    vectorized=False,
    n_resamples=9999,
    batch=None,
    alternative="two-sided",
    axis=0,
    random_state=None,
):
    """Perform permutation testing."""
    args = _permutation_test_iv(
        data,
        statistic,
        permutation_type,
        vectorized,
        n_resamples,
        batch,
        alternative,
        axis,
        random_state,
    )
    (
        data,
        statistic,
        permutation_type,
        vectorized,
        n_resamples,
        batch,
        alternative,
        axis,
        random_state,
    ) = args

    observed = statistic(*data, axis=-1)

    null_calculators = {
        "pairings": _calculate_null_pairings,
        "samples": _calculate_null_samples,
        "independent": _calculate_null_both,
    }
    null_calculator_args = (data, statistic, n_resamples, batch, random_state)
    calculate_null = null_calculators[permutation_type]
    null_distribution, n_resamples, exact_test = calculate_null(*null_calculator_args)

    # See References [2] and [3]
    adjustment = 0 if exact_test else 1

    # relative tolerance for detecting numerically distinct but
    # theoretically equal values in the null distribution
    eps = 1e-14
    gamma = np.maximum(eps, np.abs(eps * observed))

    def less(null_distribution, observed):  # noqa
        """Calculate p-values for the one-sided test."""
        cmps = null_distribution <= observed + gamma
        pvalues = (cmps.sum(axis=0) + adjustment) / (n_resamples + adjustment)  # noqa
        return pvalues

    def greater(null_distribution, observed):  # noqa
        """Calculate p-values for the one-sided test."""
        cmps = null_distribution >= observed - gamma
        pvalues = (cmps.sum(axis=0) + adjustment) / (n_resamples + adjustment)  # noqa
        return pvalues

    def two_sided(null_distribution, observed):  # noqa
        """Calculate p-values for the two-sided test."""
        pvalues_less = less(null_distribution, observed)
        pvalues_greater = greater(null_distribution, observed)
        pvalues = np.minimum(pvalues_less, pvalues_greater) * 2
        return pvalues

    compare = {"less": less, "greater": greater, "two-sided": two_sided}

    p_values = compare[alternative](null_distribution, observed)
    p_values = np.clip(p_values, 0, 1)

    return PermutationTestResult(observed, p_values, null_distribution)
