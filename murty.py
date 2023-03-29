from itertools import islice
from typing import Iterator

import numpy as np
from scipy.optimize import linear_sum_assignment


def n_best_scores(matrix: np.ndarray, n_best: int):
    """The same result as:
    ```
    from itertools import combinations, permutations
    import numpy as np

    def all_solutions(matrix: np.ndarray, n_best: int):
        rows, cols = matrix.shape
        assigns = min(rows, cols)
        scores = (
            matrix[comb, combination].sum()
            for combination in permutations(range(cols), assigns)
            for comb in combinations(range(rows), assigns)
        )
        return sorted(scores, reverse=True)[:n_best]
    ```
    """
    return list(islice(best_scores(matrix), n_best))


def best_scores(matrix: np.ndarray) -> Iterator[float]:
    """Implementation of https://pubsonline.informs.org/doi/abs/10.1287/opre.16.3.682
    (Murty-Hungarian sorting) inspired by https://github.com/sandipde/Hungarian-Murty/

    Specially implemented for maximizing an integer matrix, arbitrary shape.

    Yields
    ------
    float
        The next highest score assignment.
    """
    rows, cols = linear_sum_assignment(matrix, maximize=True)

    nodes = [matrix]
    scores = [matrix[rows, cols].sum()]
    rsrvs = [0]

    while True:
        max_score_index = np.argmax(scores)
        yield scores[max_score_index]
        node = nodes[max_score_index]
        scores[max_score_index] = np.iinfo(int).min
        rows, cols = linear_sum_assignment(node, maximize=True)

        for m in range(len(rows)):
            rsrv_cost = node[rows[:m], cols[:m]].sum() + rsrvs[max_score_index]
            rsrvs.append(rsrv_cost)

            new_node = node.copy()
            new_node[rows[m], cols[m]] = np.iinfo(int).min
            new_node = np.delete(new_node, rows[:m], axis=0)
            new_node = np.delete(new_node, cols[:m], axis=1)
            nodes.append(new_node)

            node_rows, node_cols = linear_sum_assignment(new_node, maximize=True)
            scores.append(new_node[node_rows, node_cols].sum() + rsrv_cost)
