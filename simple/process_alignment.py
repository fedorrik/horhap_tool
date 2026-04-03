#!/usr/bin/env python3
import argparse
import numpy as np
from datetime import datetime
from Bio.SeqIO import parse
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage


def read_alignment_fasta(alignment_file):
    """Read FASTA in ORIGINAL input order. No sorting, no header parsing."""
    mfa = [[record.id.replace("(+)", "").replace("(-)", ""), str(record.seq).upper()] for record in parse(alignment_file, "fasta")]
    if not mfa:
        raise ValueError("Empty FASTA")
    return mfa


# Ignore positions where either sequence has '-'
# Normalize by full alignment length

def hamming_no_gap_dist(u, v):
    mismatches = ((u != v) & (u != "-") & (v != "-")).sum()
    return mismatches / len(u)



def main():
    parser = argparse.ArgumentParser(
        description="Compute distance matrix (TSV) + condensed pdist + linkage and save"
    )
    parser.add_argument("-a", "--alignment", required=True, help="Input alignment FASTA")
    parser.add_argument("-o", "--out_prefix", required=True, help="Output prefix")
    parser.add_argument(
        "-l",
        "--linkage_method",
        default="ward",
        help="Linkage method (default=ward)",
    )
    args = parser.parse_args()

    print(datetime.now(), "Reading alignment")
    alignment_list = read_alignment_fasta(args.alignment)

    names = np.array([h for h, _ in alignment_list], dtype=object)
    seqs = [s for _, s in alignment_list]

    lengths = {len(s) for s in seqs}
    if len(lengths) != 1:
        raise ValueError("Alignment is not fixed-length")

    seq_mat = np.array([np.array(list(s), dtype="U1") for s in seqs])

    print(datetime.now(), f"Computing pdist (N={len(seq_mat)}, L={len(seq_mat[0])})")
    y = pdist(seq_mat, metric=hamming_no_gap_dist).astype(np.float32)

    print(datetime.now(), "Converting to square distance matrix")
    dist_matrix = squareform(y)

    print(datetime.now(), "Saving distance matrix as TSV")
    tsv_path = args.out_prefix + "_distance_matrix.tsv"
    with open(tsv_path, "w") as f:
        f.write("\t" + "\t".join(names) + "\n")
        for i, name in enumerate(names):
            row = "\t".join(f"{dist_matrix[i, j]:.6f}" for j in range(len(names)))
            f.write(f"{name}\t{row}\n")

    print(datetime.now(), "Computing linkage:", args.linkage_method)
    Z = linkage(y, method=args.linkage_method)

    print(datetime.now(), "Saving npz")
    np.savez_compressed(
        args.out_prefix + ".npz",
        y=y,
        Z=Z,
        names=names,
        meta=np.array(
            [
                {
                    "created": datetime.now().isoformat(timespec="seconds"),
                    "metric": "hamming_no_gap_dist",
                    "linkage_method": args.linkage_method,
                    "n": len(seq_mat),
                    "L": len(seq_mat[0]),
                    "input_order_preserved": True,
                }
            ],
            dtype=object,
        ),
    )

    print(datetime.now(), "Done")


if __name__ == "__main__":
    main()
