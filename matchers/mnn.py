from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch as th
from torch import Tensor


def get_margin_and_ratio_from_scores_and_mnn_matrix(
    mnn_matrix: Tensor,
    best_scores0: Tensor,
    second_best_scores0: Tensor,
    second_best_scores1: Tensor,
) -> tuple[Tensor, Tensor]:
    """
    Args:
        mnn_matrix:
            n0,n1 bool
        best_scores0:
            n0
        second_best_scores0:
            n0
        second_best_scores1:
            n1
    Returns:
        margin:
            n0
        ratio:
            n0
    """
    assert mnn_matrix.ndim == 2
    assert (
        best_scores0.ndim == second_best_scores0.ndim == second_best_scores1.ndim == 1
    )
    assert mnn_matrix.shape[0] == best_scores0.shape[0] == second_best_scores0.shape[0]
    assert mnn_matrix.shape[1] == second_best_scores1.shape[0]

    rows_matches_idx, column_matches_idx = th.where(
        mnn_matrix
    )  # (n_matches), (n_matches)
    best_scores0_matches = best_scores0[rows_matches_idx]  # n_matches_proposed
    # ? by definition of mnn, the best_scores0_matches are exactly the same as best_scores1_matches
    # best_scores1_matches = best_scores1[batch_matches, column_matches]  # n_matches_proposed
    second_best_scores0_matches = second_best_scores0[
        rows_matches_idx
    ]  # n_matches_proposed
    second_best_scores1_matches = second_best_scores1[
        column_matches_idx
    ]  # n_matches_proposed
    margin = best_scores0_matches - th.max(
        second_best_scores0_matches, second_best_scores1_matches
    )  # n_matches_proposed
    ratio = (
        th.max(second_best_scores0_matches, second_best_scores1_matches)
        / best_scores0_matches
    )  # n_matches_proposed
    return margin, ratio


@dataclass
class MatchingMatrixExtra:
    """utility class to store the matching matrix and extra information about it"""

    # ? the proposed matching matrix
    proposed: Tensor  # (B),n0,n1 bool
    # ? the matching matrix with the correct matches
    correct: Tensor | None = None  # (B),n0,n1 bool
    # ? the matching matrix with the wrong matches
    wrong: Tensor | None = None  # (B),n0,n1 bool
    # ? the matching matrix with the mismatched matches (there exist a correct matches for that point, but it's wrongly matched)
    mismatched: Tensor | None = None  # (B),n0,n1 bool
    # ? a match is found between two points that have no existing match in the GT_matching_matrix
    inexistent: Tensor | None = None  # (B),n0,n1 bool
    # ? matching_matrix_unsure is true when one of the proposed matches does not correspond either to a match or to an unmatch
    unsure: Tensor | None = None  # (B),n0,n1 bool
    score: Tensor | None = None  # (B),n0,n1 float

    def shape(self) -> th.Size:
        return self.proposed.shape

    def __repr__(self) -> str:
        return f"MatchingMatrixExtra [{tuple(self.shape())}]  device: {self.proposed.device}"

    def __getitem__(self, b: int = 0) -> MatchingMatrixExtra:
        assert len(self.shape()) >= 3, (
            "MatchingMatrix must have at least 3 dimensions to be sliced"
        )
        return MatchingMatrixExtra(
            proposed=self.proposed[b],
            correct=self.correct[b] if self.correct is not None else None,
            wrong=self.wrong[b] if self.wrong is not None else None,
            mismatched=self.mismatched[b] if self.mismatched is not None else None,
            inexistent=self.inexistent[b] if self.inexistent is not None else None,
            unsure=self.unsure[b] if self.unsure is not None else None,
            score=self.score[b] if self.score is not None else None,
        )

    def to(self, device: str) -> MatchingMatrixExtra:
        self.proposed = self.proposed.to(device)
        self.correct = self.correct.to(device) if self.correct is not None else None
        self.wrong = self.wrong.to(device) if self.wrong is not None else None
        self.mismatched = (
            self.mismatched.to(device) if self.mismatched is not None else None
        )
        self.inexistent = (
            self.inexistent.to(device) if self.inexistent is not None else None
        )
        self.unsure = self.unsure.to(device) if self.unsure is not None else None
        self.score = self.score.to(device) if self.score is not None else None
        return self

    def cpu(self) -> MatchingMatrixExtra:
        return self.to("cpu")


@dataclass
class Matches:
    matches: Tensor  # n_matches,2
    score_matrix: Tensor  # n0,n1
    score_matrix_with_bins: Tensor | None = None  # n0+1,n1+1
    matching_matrix_extra: MatchingMatrixExtra | None = None
    matching_matrix_GT_with_bins: Tensor | None = None  # n0+1,n1+1

    @property
    def matching_matrix(self) -> Tensor:
        output = th.zeros_like(self.score_matrix, dtype=th.bool)
        output[self.matches[:, 0], self.matches[:, 1]] = True
        return output

    def _compute_matching_matrix_extra(
        self, matching_matrix_GT_with_bins: Tensor
    ) -> None:
        self.matching_matrix_GT_with_bins = matching_matrix_GT_with_bins
        self.matching_matrix_extra = (
            compute_correct_wrong_mismatched_inexistent_unsure_matches(
                self.matching_matrix[None], matching_matrix_GT_with_bins[None]
            )[0]
        )

    def compute_scores_stats(
        self, matching_matrix_GT_with_bins: Tensor
    ) -> dict[str, float]:
        """compute different matching statistics that can be useful to investigate the matching performance
        Args:
            matching_matrix_GT_with_bins:
                n0+1,n1+1
        """
        self._assert_gt_with_bins_shape(matching_matrix_GT_with_bins)
        self._compute_matching_matrix_extra(matching_matrix_GT_with_bins)

        score_matrix_with_inf = self.score_matrix.clone()
        score_matrix_with_inf[score_matrix_with_inf.isnan()] = float("-inf")

        stats = self._count_and_score_stats(matching_matrix_GT_with_bins)
        stats.update(self._margin_ratio_stats(score_matrix_with_inf))
        stats["n_masked"] = self._n_masked(score_matrix_with_inf)
        return stats

    def _assert_gt_with_bins_shape(self, gt: Tensor) -> None:
        """Validate that the GT matching matrix has bins matching the score matrix."""
        assert gt.ndim == 2, f"expected 2D tensor, got {gt.ndim}D"
        assert gt.shape[0] == self.score_matrix.shape[0] + 1, (
            f"expected {self.score_matrix.shape[0] + 1} rows, got {gt.shape[0]}"
        )
        assert gt.shape[1] == self.score_matrix.shape[1] + 1, (
            f"expected {self.score_matrix.shape[1] + 1} cols, got {gt.shape[1]}"
        )

    def _count_and_score_stats(self, gt: Tensor) -> dict[str, float]:
        """Match counts, mean scores per category, and precision/recall."""
        extra = self.matching_matrix_extra
        n_gt = gt[:-1, :-1].sum().item()
        n_proposed = extra.proposed.sum().item()
        n_correct = extra.correct.sum().item()
        return {
            "n_matches_GT": n_gt,
            "n_matches_proposed": n_proposed,
            "n_matches_correct": n_correct,
            "n_matches_wrong": extra.wrong.sum().item(),
            "n_matches_mismatched": extra.mismatched.sum().item(),
            "n_matches_inexistent": extra.inexistent.sum().item(),
            "n_matches_unsure": extra.unsure.sum().item(),
            "mean_GT_score": self.score_matrix[gt[:-1, :-1]].mean().item(),
            "mean_proposed_score": self.score_matrix[extra.proposed].mean().item(),
            "mean_correct_score": self.score_matrix[extra.correct].mean().item(),
            "mean_wrong_score": self.score_matrix[extra.wrong].mean().item(),
            "mean_mismatched_score": self.score_matrix[extra.mismatched].mean().item(),
            "mean_inexistent_score": self.score_matrix[extra.inexistent].mean().item(),
            "mean_unsure_score": self.score_matrix[extra.unsure].mean().item(),
            "mean_matching_matrix_score": self.score_matrix.mean().item(),
            "matches_precision": (
                n_correct / n_proposed if n_proposed > 0 else 0.0
            ),
            "matches_recall": n_correct / n_gt if n_gt > 0 else 0.0,
        }

    def _margin_ratio_stats(self, score_matrix_with_inf: Tensor) -> dict[str, float]:
        """Mean margin and ratio per match category."""
        best_two_scores0 = th.topk(score_matrix_with_inf, 2, dim=-1)[0]  # (n0,2)
        best_two_scores1 = th.topk(score_matrix_with_inf, 2, dim=-2)[0].T  # (n1,2)
        best_scores0, second_best_scores0 = best_two_scores0[:, 0], best_two_scores0[:, 1]
        # ? best scores1 is not needed as all the matches are mutual nearest neighbors anyway,
        # ? so the sampled best_scores0 is the same as the sampled best_scores1 by definition
        _, second_best_scores1 = best_two_scores1[:, 0], best_two_scores1[:, 1]

        extra = self.matching_matrix_extra
        categories = {
            "proposed": extra.proposed,
            "correct": extra.correct,
            "wrong": extra.wrong,
            "mismatched": extra.mismatched,
            "inexistent": extra.inexistent,
        }
        margins_ratios = {
            name: get_margin_and_ratio_from_scores_and_mnn_matrix(
                mat, best_scores0, second_best_scores0, second_best_scores1
            )
            for name, mat in categories.items()
        }
        stats = {
            f"mean_margin_{name}": margin.mean().item()
            for name, (margin, _) in margins_ratios.items()
        }
        stats.update(
            {
                f"mean_ratio_{name}": ratio.mean().item()
                for name, (_, ratio) in margins_ratios.items()
            }
        )
        return stats

    def _n_masked(self, score_matrix_with_inf: Tensor) -> float:
        """Count possible matches shielded (masked) by an existing correct match."""
        # ? find out how many possible mismatched have been shielded by a correct match
        # ? counting how many columns have the row-max at a column where there is a correct match
        matches_correct_idx = self.matching_matrix_extra.correct.nonzero()  # (n_corr,2)
        row_max_mask = (
            score_matrix_with_inf == score_matrix_with_inf.max(dim=-1, keepdim=True)[0]
        ) * score_matrix_with_inf.isfinite()  # n0,n1
        masked_columns = row_max_mask[:, matches_correct_idx[:, -1]].T
        n_masked_by_columns = masked_columns.sum() - masked_columns.shape[0]
        # ? do the same by columns
        column_max_mask = (
            score_matrix_with_inf == score_matrix_with_inf.max(dim=-2, keepdim=True)[0]
        ) * score_matrix_with_inf.isfinite()  # (n0,n1)
        masked_rows = column_max_mask[matches_correct_idx[:, -2], :]
        n_masked_by_rows = masked_rows.sum() - masked_rows.shape[0]
        return (n_masked_by_columns + n_masked_by_rows).item()

    def to(self, device: str) -> Matches:
        self.matches = self.matches.to(device)
        self.score_matrix = self.score_matrix.to(device)
        self.score_matrix_with_bins = (
            self.score_matrix_with_bins.to(device)
            if self.score_matrix_with_bins is not None
            else None
        )
        return self

    def cpu(self) -> Matches:
        return self.to("cpu")

    def __repr__(self) -> str:
        return f"Matches [{tuple(self.matches.shape)}]  device: {self.matches.device}"

    @property
    def shape(self) -> tuple[int, ...]:
        return self.score_matrix.shape


class Matcher(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.name = "Matcher"

    @abstractmethod
    def match(self, des0: list[Tensor], des1: list[Tensor]) -> list[Matches]:
        """Match two batches of descriptors and return one ``Matches`` per pair."""
        raise NotImplementedError

    @abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError


class MNN(Matcher):
    def __init__(
        self, min_score: float, ratio_test: float = 1.0, device: str = "cpu"
    ) -> None:
        self.min_score = min_score
        self.ratio_test = ratio_test
        self.device = device

        self.name = "MNN"
        if self.min_score != -1.0:
            self.name = f"{self.name}{self.min_score}"
        if self.ratio_test != 1.0:
            self.name = f"{self.name}-ratiotest{self.ratio_test}"

    def match(self, des0: list[Tensor], des1: list[Tensor]) -> list[Matches]:
        """Mutual-nearest-neighbor match each descriptor pair in the batch.

        Args:
            des0: List (batch) of descriptor tensors ``(n0, dim)`` from image 0.
            des1: List (batch) of descriptor tensors ``(n1, dim)`` from image 1.

        Returns:
            One ``Matches`` per batch element, holding the matches and score matrix.
        """
        matches_list, score_matrix_list = match_descriptors_mnn_scores_ratio_test(
            des0, des1, self.min_score, self.ratio_test
        )
        output = [
            Matches(matches, score_matrix)
            for matches, score_matrix in zip(matches_list, score_matrix_list)
        ]
        return output

    def __repr__(self) -> str:
        return self.name


def _mutual_nn_mask(score_mat: Tensor) -> Tensor:
    """Bool mask of positions that are the argmax along both their row and column.

    ``score_mat`` (Bxn0xn1) must already be NaN-free (NaNs replaced by ``-inf``).
    Rows/columns whose max is ``-inf`` are routed to a throwaway bin so they match
    nothing.
    """
    B, n0, n1 = score_mat.shape
    device = score_mat.device

    # ? each row scores a descriptor from img0 against all of img1 (and vice-versa)
    nn0_value, nn0_idx = score_mat.max(2)  # (B,n0) with values [0, n1[
    nn1_value, nn1_idx = score_mat.max(1)  # (B,n1) with values [0, n0[
    nn0_idx[nn0_value == float("-inf")] = n1  # route empty rows to the bin
    nn1_idx[nn1_value == float("-inf")] = n0

    nn0_matrix = th.zeros((B, n0 + 1, n1 + 1), dtype=th.bool, device=device)
    nn0_matrix[:, :-1, :].scatter_(2, nn0_idx[:, :, None], True)
    nn1_matrix = th.zeros((B, n0 + 1, n1 + 1), dtype=th.bool, device=device)
    nn1_matrix[:, :, :-1].scatter_(1, nn1_idx[:, None, :], True)

    # ? compose the two directions, then drop the bin row/column
    return (nn0_matrix * nn1_matrix)[:, :-1, :-1]


def _ratio_test_mask(score_mat: Tensor, ratio_test: float) -> Tensor:
    """Lowe-style ratio-test mask: best score must beat ``ratio_test`` x second-best."""
    best_scores0 = score_mat.topk(2, dim=-1, largest=True, sorted=True)[0]  # (B,n0,2)
    best_scores1 = score_mat.topk(2, dim=-2, largest=True, sorted=True)[0]  # (B,2,n1)
    valid_mask0 = best_scores0[:, :, 0] * ratio_test > best_scores0[:, :, 1]  # (B,n0)
    valid_mask1 = best_scores1[:, 0, :] * ratio_test > best_scores1[:, 1, :]  # (B,n1)
    return valid_mask0[:, :, None] * valid_mask1[:, None, :]


def mutual_nearest_neighbors_from_score_matrix(
    score_mat: Tensor, min_score: float = -1.0, ratio_test: float = 1.0
) -> Tensor:
    """return a boolean matrix with a True where the position was a maximum in both row and columns (and grater than the min score)
    Args:
        score_mat: score_matrix matrix
            Bxn0xn1
        min_score: minimum score to consider a match
        ratio_test: ratio test to apply to the score matrix
    Returns:
        mnn: mutual nearest neighbors matrix
            Bxn0xn1 th.bool
    """
    assert score_mat.ndim == 3
    B, n0, n1 = score_mat.shape
    if n0 == 0 or n1 == 0:
        return score_mat.new_zeros((B, n0, n1), dtype=th.bool)

    score_mat = score_mat.clone()
    score_mat[score_mat.isnan()] = float("-inf")

    mnn_matrix = _mutual_nn_mask(score_mat)
    mnn_matrix = mnn_matrix * (score_mat > min_score)
    if ratio_test < 1.0:
        mnn_matrix = mnn_matrix * _ratio_test_mask(score_mat, ratio_test)
    return mnn_matrix


def mutual_nearest_neighbors_from_dist_matrix(dist: Tensor) -> Tensor:
    """return a boolean matrix with a True where the position was a minimum in both row and columns
    Args:
        dist: distance matrix
            Bxn0xn1
    Returns:
        mnn: mutual nearest neighbors matrix
            Bxn0xn1 th.bool
    """
    B, n0, n1 = dist.shape
    if n0 == 0 or n1 == 0:
        return dist.new_zeros((B, n0, n1), dtype=th.bool)

    device = dist.device

    # ? get the closest ones for each row and column
    nn0 = th.argmin(dist, dim=2)  # Bxn0 with values [0, n1[
    nn1 = th.argmin(dist, dim=1)  # Bxn1 with values [0, n0[

    # ? build the closest one matrix for each kpts0 (every row is dist from a kpts0_i and all the others kpts1)
    B0_idxs = th.arange(B).repeat_interleave(n0).to(device)  # B*n0
    nn0_matrix = th.zeros_like(dist, dtype=th.bool)  # Bxn0xn1
    nn0_matrix[B0_idxs, th.arange(n0).repeat(B, 1).reshape(-1), nn0.reshape(-1)] = (
        True  # Bxn0xn1
    )

    # ? build the closest one matrix for each kpts1 (every row is dist from a kpts1_i and all the others kpts0)
    B1_idxs = th.arange(B).repeat_interleave(n1)
    nn1_matrix = th.zeros_like(dist, dtype=th.bool)  # Bxn0xn1
    nn1_matrix[B1_idxs, nn1.reshape(-1), th.arange(n1).repeat(B, 1).reshape(-1)] = True

    # ? by multiplying the two matrices only the mutual-nearest-neighbours are selected
    mnn_matrix = nn0_matrix * nn1_matrix

    return mnn_matrix


def match_descriptors_mnn_scores_ratio_test(
    des0: list[Tensor],
    des1: list[Tensor],
    min_score: float = -1.0,
    ratio_test: float = 1.0,
) -> tuple[list[Tensor], list[Tensor]]:
    """match keypoints looking for mutual nearest neighbor in the descriptors space using the inner product
    Args:
        des0: list of descriptor tensors extracted from img0
            list[B] of Tensor[n_extracted0, des_dim]
        des1: list of descriptor tensors extracted from img1
            list[B] of Tensor[n_extracted1, des_dim]
        min_score: the minimum score of two mnn to be considered a valid match
        ratio_test: if > 0, we require the score of the second-best match to be at least ratio_test times smaller than the
    Returns:
        matches_list: list of matches given with double index notation
            list[B] of Tensor[n_matches, 2]     with order (idx0, idx1)
        score_matrix_list: list of score matrices
            list[B] of Tensor[n_extracted0, n_extracted1]
    """
    B = len(des0)
    device = des0[0].device

    matches_list = []
    score_matrix_list = []
    for b in range(B):
        # ? match keypoints
        if des0[b].shape[0] == 0 or des1[b].shape[0] == 0:
            matches = th.zeros(0, 2, device=device, dtype=th.long)
            score_matrix = th.zeros(des0[b].shape[0], des1[b].shape[0], device=device)
        else:
            score_matrix = des0[b] @ des1[b].permute(1, 0)  # n0 x n1
            # ? set the nan in the score_matrix to -1
            if score_matrix.isnan().any():
                print("WARNING: score matrix have nan values, setting those to -1")
                score_matrix[score_matrix.isnan()] = -1
            matches_mat = mutual_nearest_neighbors_from_score_matrix(
                score_matrix[None], min_score=min_score, ratio_test=ratio_test
            )[0]  # n0 x n1

            matches = th.nonzero(matches_mat)
        matches_list.append(matches)
        score_matrix_list.append(score_matrix)
    return matches_list, score_matrix_list


def _assert_ccw_inputs(
    matching_matrix: Tensor, GT_matching_matrix_with_bins: Tensor
) -> None:
    """Validate the shapes/dtypes for the correct/wrong/... decomposition."""
    assert matching_matrix.shape[0] == GT_matching_matrix_with_bins.shape[0], (
        f"{matching_matrix.shape[0]} != {GT_matching_matrix_with_bins.shape[0]}"
    )
    assert matching_matrix.shape[1] == GT_matching_matrix_with_bins.shape[1] - 1, (
        f"{matching_matrix.shape[1]} != {GT_matching_matrix_with_bins.shape[1] - 1}"
    )
    assert matching_matrix.shape[2] == GT_matching_matrix_with_bins.shape[2] - 1, (
        f"{matching_matrix.shape[2]} != {GT_matching_matrix_with_bins.shape[2] - 1}"
    )
    assert matching_matrix.ndim == 3, f"{matching_matrix.ndim} != 3"
    assert (
        matching_matrix.dtype == th.bool
        and GT_matching_matrix_with_bins.dtype == th.bool
    ), (
        f"{matching_matrix.dtype} != {th.bool} or {GT_matching_matrix_with_bins.dtype} != {th.bool}"
    )


def _known_and_match_masks(
    GT_matching_matrix_with_bins: Tensor, GT_matching_matrix: Tensor
) -> tuple[Tensor, Tensor]:
    """Return ``(known_mask, any_match_mask)`` over the GT matching matrix.

    known_mask: rows/cols that have any GT entry (a match *or* the unmatched bin).
    any_match_mask: rows/cols that have an actual GT match (bins excluded).
    """
    B, H, W = GT_matching_matrix.shape
    known_with_bins = GT_matching_matrix_with_bins.any(1, keepdim=True).repeat(
        1, H + 1, 1
    ) + GT_matching_matrix_with_bins.any(2, keepdim=True).repeat(1, 1, W + 1)
    known_mask = known_with_bins[:, :-1, :-1]
    any_match_mask = GT_matching_matrix.any(1, keepdim=True).repeat(
        1, H, 1
    ) + GT_matching_matrix.any(2, keepdim=True).repeat(1, 1, W)
    return known_mask, any_match_mask


def compute_correct_wrong_mismatched_inexistent_unsure_matches(
    matching_matrix: Tensor, GT_matching_matrix_with_bins: Tensor
) -> MatchingMatrixExtra:
    """
    Args:
        matching_matrix: the matching matrix obtained from descriptors
            B,n0,n1
        GT_matching_matrix_with_bins: the GT matching matrix with one additional bin row and column with the unmatched keypoints
            B,n0+1,n1+1

    Returns:
        MatchingMatrixExtra
    """
    _assert_ccw_inputs(matching_matrix, GT_matching_matrix_with_bins)

    GT_matching_matrix = GT_matching_matrix_with_bins[:, :-1, :-1]
    matching_matrix_correct = matching_matrix * GT_matching_matrix
    known_mask, any_match_mask = _known_and_match_masks(
        GT_matching_matrix_with_bins, GT_matching_matrix
    )

    # unsure: a proposed match that is neither a known match nor a known unmatch
    matching_matrix_unsure = matching_matrix * ~known_mask
    # wrong: a proposed match at a known position that is not the GT match
    matching_matrix_wrong = (
        (matching_matrix ^ GT_matching_matrix) * matching_matrix
    ) * known_mask
    # mismatched: wrong, but the point did have a possible correct match
    matching_matrix_mismatched = matching_matrix_wrong * any_match_mask
    # inexistent: wrong, between two points that had no GT match at all
    matching_matrix_inexistent = matching_matrix_wrong * ~any_match_mask

    return MatchingMatrixExtra(
        matching_matrix,
        matching_matrix_correct,
        matching_matrix_wrong,
        matching_matrix_mismatched,
        matching_matrix_inexistent,
        matching_matrix_unsure,
    )
