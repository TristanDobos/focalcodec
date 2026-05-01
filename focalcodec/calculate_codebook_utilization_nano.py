import argparse
from pathlib import Path
import torch
import math

parser = argparse.ArgumentParser()
parser.add_argument(
    "experiment_name",
    help="Name of the experiment, e.g. NanoCodecWithMetricsWPesqPaper",
)


def compute_entropy_from_saved_tokens(
    tokens_dir,
    codebook_size=4032,
    num_codebooks=4,
):
    tokens_dir = Path(tokens_dir)

    counts = torch.zeros(
        num_codebooks,
        codebook_size,
        dtype=torch.float64,
    )

    pt_files = sorted(tokens_dir.glob("*.pt"))

    if len(pt_files) == 0:
        raise RuntimeError(f"No .pt files found in {tokens_dir}")

    for pt_path in pt_files:
        data = torch.load(pt_path, map_location="cpu")

        tokens = data["tokens"]  # expected shape: [B, C, T]
        tokens_len = data.get("tokens_len", data.get("encoded_len"))

        if tokens_len is None:
            raise ValueError(
                f"{pt_path} does not contain 'tokens_len' or 'encoded_len'"
            )

        if tokens.ndim != 3:
            raise ValueError(
                f"{pt_path} has tokens shape {tokens.shape}, expected [B, C, T]"
            )

        B, C, T = tokens.shape

        if C != num_codebooks:
            raise ValueError(
                f"{pt_path} has {C} codebooks, expected {num_codebooks}. "
                f"Shape: {tokens.shape}"
            )

        for b in range(B):
            length = int(tokens_len[b].item()) if torch.is_tensor(tokens_len) else int(tokens_len[b])
            length = min(length, T)

            for c in range(num_codebooks):
                ids = tokens[b, c, :length].long()

                # safety filter
                ids = ids[(ids >= 0) & (ids < codebook_size)]

                if ids.numel() == 0:
                    continue

                counts[c] += torch.bincount(
                    ids,
                    minlength=codebook_size,
                ).to(torch.float64)

    metrics = {}

    for c in range(num_codebooks):
        cb_counts = counts[c]
        total = cb_counts.sum()

        if total == 0:
            metrics[f"codebook_{c}_used_codes"] = 0
            metrics[f"codebook_{c}_utilization"] = 0.0
            metrics[f"codebook_{c}_entropy_bits"] = 0.0
            metrics[f"codebook_{c}_normalized_entropy"] = 0.0
            metrics[f"codebook_{c}_perplexity"] = 0.0
            metrics[f"codebook_{c}_normalized_perplexity"] = 0.0
            continue

        probs = cb_counts / total
        nonzero_probs = probs[probs > 0]

        entropy_nats = -(nonzero_probs * torch.log(nonzero_probs)).sum()
        entropy_bits = entropy_nats / math.log(2)

        used_codes = (cb_counts > 0).sum().item()
        utilization = used_codes / codebook_size

        max_entropy_bits = math.log2(codebook_size)
        normalized_entropy = entropy_bits.item() / max_entropy_bits

        perplexity = torch.exp(entropy_nats).item()
        normalized_perplexity = perplexity / codebook_size

        metrics[f"codebook_{c}_used_codes"] = used_codes
        metrics[f"codebook_{c}_utilization"] = utilization
        metrics[f"codebook_{c}_entropy_bits"] = entropy_bits.item()
        metrics[f"codebook_{c}_normalized_entropy"] = normalized_entropy
        metrics[f"codebook_{c}_perplexity"] = perplexity
        metrics[f"codebook_{c}_normalized_perplexity"] = normalized_perplexity

    metrics["avg_codebook_utilization"] = sum(
        metrics[f"codebook_{c}_utilization"]
        for c in range(num_codebooks)
    ) / num_codebooks

    metrics["avg_codebook_entropy_bits"] = sum(
        metrics[f"codebook_{c}_entropy_bits"]
        for c in range(num_codebooks)
    ) / num_codebooks

    metrics["avg_normalized_codebook_entropy"] = sum(
        metrics[f"codebook_{c}_normalized_entropy"]
        for c in range(num_codebooks)
    ) / num_codebooks

    metrics["avg_normalized_codebook_perplexity"] = sum(
        metrics[f"codebook_{c}_normalized_perplexity"]
        for c in range(num_codebooks)
    ) / num_codebooks

    return metrics, counts


args = parser.parse_args()

experiment_name = args.experiment_name
tokens_dir = Path("/mnt/scratch/tmp/xdobos00/nemo_tokens_big") / experiment_name

codebook_size = 9 * 8 * 8 * 7
num_codebooks = 4
top_k = 100

metrics, counts = compute_entropy_from_saved_tokens(
    tokens_dir=tokens_dir,
    codebook_size=codebook_size,
    num_codebooks=num_codebooks,
)

for c in range(num_codebooks):
    print(f"Codebook {c}:")
    print(f"  used codes: {metrics[f'codebook_{c}_used_codes']} / {codebook_size}")
    print(f"  utilization: {metrics[f'codebook_{c}_utilization'] * 100:.2f}%")
    print(f"  entropy: {metrics[f'codebook_{c}_entropy_bits']:.3f} bits")
    print(f"  normalized entropy: {metrics[f'codebook_{c}_normalized_entropy'] * 100:.2f}%")
    print(f"  perplexity: {metrics[f'codebook_{c}_perplexity']:.2f}")
    print(f"  normalized perplexity: {metrics[f'codebook_{c}_normalized_perplexity'] * 100:.2f}%")

    cb_counts = counts[c]
    total = cb_counts.sum().item()

    nonzero_ids = torch.nonzero(cb_counts > 0, as_tuple=False).flatten()
    used_count = nonzero_ids.numel()

    k = min(top_k, used_count)

    if k > 0:
        top_counts, top_ids = torch.topk(cb_counts, k=k, largest=True)

        print(f"  most used codes, top {k} of {used_count} used:")
        print(f"    {'rank':>4}  {'code':>6}  {'count':>12}  {'usage':>8}")
        print(f"    {'-' * 4}  {'-' * 6}  {'-' * 12}  {'-' * 8}")

        for rank, (code_id, count) in enumerate(
            zip(top_ids.tolist(), top_counts.tolist()),
            start=1,
        ):
            usage_pct = 100.0 * count / total if total > 0 else 0.0

            print(
                f"    {rank:4d}  "
                f"{code_id:6d}  "
                f"{int(count):12,d}  "
                f"{usage_pct:7.2f}%"
            )
    else:
        print("  most used codes: none")

    print()

print(f"Average utilization: {metrics['avg_codebook_utilization'] * 100:.2f}%")
print(f"Average entropy: {metrics['avg_codebook_entropy_bits']:.3f} bits")
print(f"Average normalized entropy: {metrics['avg_normalized_codebook_entropy'] * 100:.2f}%")
print(f"Average normalized perplexity: {metrics['avg_normalized_codebook_perplexity'] * 100:.2f}%")