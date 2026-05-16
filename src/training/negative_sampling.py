"""Uniform negative edge sampling with rejection.

Rejects edges that are self-loops or appear in the positive set.
"""
import torch


def sample_negative_edges(
    pos_edges: torch.Tensor,
    num_nodes: int,
    num_samples: int,
    seed: int | None = None,
) -> torch.Tensor:
    """Sample `num_samples` negative edges as a [2, num_samples] tensor.

    Args:
        pos_edges: [2, P] positive edges to exclude.
        num_nodes: vocabulary size for sampling.
        num_samples: how many negatives to return.
        seed: per-call RNG seed for reproducibility (None to use ambient).

    Returns:
        neg_edges: [2, num_samples] long tensor.
    """
    if num_samples == 0:
        return torch.empty(2, 0, dtype=torch.long)

    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)

    pos_set: set[tuple[int, int]] = {
        (int(u), int(v)) for u, v in pos_edges.t().tolist()
    }

    sampled: list[tuple[int, int]] = []
    max_attempts = num_samples * 20
    attempts = 0
    while len(sampled) < num_samples and attempts < max_attempts:
        batch = num_samples - len(sampled)
        candidates = torch.randint(0, num_nodes, (2, batch * 2), generator=generator)
        for k in range(candidates.shape[1]):
            u, v = int(candidates[0, k]), int(candidates[1, k])
            if u == v:
                continue
            if (u, v) in pos_set:
                continue
            sampled.append((u, v))
            if len(sampled) == num_samples:
                break
        attempts += 1

    if len(sampled) < num_samples:
        raise RuntimeError(
            f"Could not sample {num_samples} negatives after {max_attempts} attempts "
            f"(got {len(sampled)}). Graph too dense?"
        )

    if not sampled:
        return torch.empty(2, 0, dtype=torch.long)
    return torch.tensor(sampled, dtype=torch.long).t().contiguous()
