"""
Patient-similarity graph construction (Section 3.1.2), replacing the previous
tabular-cosine-similarity proxy with the actual rule described in the paper:

  An edge exists between admissions i and j iff:
    (i)  they share >= 1 historical ICD-9/ICD-10 code from a PRIOR admission
         (an ICD-based edge), OR
    (ii) they share >= r CPT/HCPCS codes recorded strictly before the current
         admission's chest X-ray time t0 (a CPT-based edge).

The graph is UNWEIGHTED: edge existence is thresholded, but the mean aggregator
in GraphSAGE (Eq. 6) treats every neighbor equally once an edge exists.

Two entry points are provided:
  - build_patient_graph(...): builds the graph once, from TRAINING admissions only.
  - attach_inductive_nodes(...): attaches new (val/test) admissions to an existing
    training graph at evaluation time, connecting them only to training nodes,
    with no new edges created among the new nodes themselves and no gradient
    flowing back into the frozen training graph. This is what makes the
    evaluation genuinely inductive (Referee 2, point #4).
"""
from typing import Dict, List, Sequence, Tuple
import torch


def _has_overlap(codes_a: Sequence[str], codes_b: Sequence[str], min_shared: int = 1) -> bool:
    if not codes_a or not codes_b:
        return False
    return len(set(codes_a) & set(codes_b)) >= min_shared


def _edges_between(
    icd_histories: List[Sequence[str]],
    cpt_codes: List[Sequence[str]],
    cpt_threshold: int,
    candidate_pairs,
) -> List[Tuple[int, int]]:
    edges = []
    for i, j in candidate_pairs:
        icd_edge = _has_overlap(icd_histories[i], icd_histories[j], min_shared=1)
        cpt_edge = _has_overlap(cpt_codes[i], cpt_codes[j], min_shared=cpt_threshold)
        if icd_edge or cpt_edge:
            edges.append((i, j))
    return edges


def build_patient_graph(
    icd_histories: List[Sequence[str]],
    cpt_codes: List[Sequence[str]],
    cpt_threshold: int = 5,
) -> torch.Tensor:
    """
    Build the training-set graph from scratch. O(N^2) pairwise check, which is
    fine for tens of thousands of admissions; for much larger cohorts, bucket
    admissions by ICD/CPT code first (inverted index) before pairwise checks.

    Returns:
        edge_index: LongTensor [2, 2*E] (undirected, both directions included).
    """
    n = len(icd_histories)
    assert n == len(cpt_codes)

    # Inverted index speeds this up a lot versus brute-force O(n^2) set intersections
    # when code vocabularies are large and per-patient code lists are short.
    icd_to_patients: Dict[str, List[int]] = {}
    for idx, codes in enumerate(icd_histories):
        for c in set(codes):
            icd_to_patients.setdefault(c, []).append(idx)

    cpt_to_patients: Dict[str, List[int]] = {}
    for idx, codes in enumerate(cpt_codes):
        for c in set(codes):
            cpt_to_patients.setdefault(c, []).append(idx)

    candidate_pairs = set()
    for _, plist in icd_to_patients.items():
        if len(plist) < 2:
            continue
        for a in range(len(plist)):
            for b in range(a + 1, len(plist)):
                i, j = plist[a], plist[b]
                candidate_pairs.add((min(i, j), max(i, j)))

    # CPT candidates: any pair co-occurring on >=1 shared code is a candidate;
    # the actual >= r threshold is re-checked exactly in _edges_between.
    for _, plist in cpt_to_patients.items():
        if len(plist) < 2:
            continue
        for a in range(len(plist)):
            for b in range(a + 1, len(plist)):
                i, j = plist[a], plist[b]
                candidate_pairs.add((min(i, j), max(i, j)))

    edges = _edges_between(icd_histories, cpt_codes, cpt_threshold, candidate_pairs)

    if not edges:
        return torch.zeros((2, 0), dtype=torch.long)

    src = [e[0] for e in edges] + [e[1] for e in edges]
    dst = [e[1] for e in edges] + [e[0] for e in edges]
    return torch.tensor([src, dst], dtype=torch.long)


def attach_inductive_nodes(
    train_icd_histories: List[Sequence[str]],
    train_cpt_codes: List[Sequence[str]],
    new_icd_histories: List[Sequence[str]],
    new_cpt_codes: List[Sequence[str]],
    cpt_threshold: int = 5,
) -> torch.Tensor:
    """
    Attach validation/test admissions to the FIXED training graph.

    New nodes are indexed starting at `n_train` (i.e. node id = n_train + local_idx).
    Edges are only created between a new node and a training node -- never between
    two new nodes, and never back-propagated into training-node embeddings during
    evaluation (the caller should run this with torch.no_grad()).

    Returns:
        edge_index: LongTensor [2, 2*E] with training-node ids in [0, n_train) and
                    new-node ids in [n_train, n_train + n_new).
    """
    n_train = len(train_icd_histories)
    edges = []
    for local_j, (icd_j, cpt_j) in enumerate(zip(new_icd_histories, new_cpt_codes)):
        j = n_train + local_j
        for i in range(n_train):
            icd_edge = _has_overlap(train_icd_histories[i], icd_j, min_shared=1)
            cpt_edge = _has_overlap(train_cpt_codes[i], cpt_j, min_shared=cpt_threshold)
            if icd_edge or cpt_edge:
                edges.append((i, j))

    if not edges:
        return torch.zeros((2, 0), dtype=torch.long)

    src = [e[0] for e in edges] + [e[1] for e in edges]
    dst = [e[1] for e in edges] + [e[0] for e in edges]
    return torch.tensor([src, dst], dtype=torch.long)
