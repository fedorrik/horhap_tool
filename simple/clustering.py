#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sys

from Bio.SeqIO import parse
from collections import Counter
from datetime import datetime
from scipy.cluster.hierarchy import dendrogram, fcluster


def read_alignment_fasta(alignment_file):
    """Read FASTA in ORIGINAL input order. No sorting, no header parsing."""
    mfa = [[record.id.replace("(+)", "").replace("(-)", ""), str(record.seq).upper()] for record in parse(alignment_file, "fasta")]
    if not mfa:
        raise ValueError(f"Empty FASTA: {alignment_file}")
    return mfa


# ------------------------------------------------------------
# Core helpers
# ------------------------------------------------------------
def get_consensus(mfa):
    consensus_seq = ""
    len_seq = len(mfa[0][1])
    for nucl_index in range(len_seq):
        cnt = Counter([seq[nucl_index] for _, seq in mfa])
        consensus_nucleotide = cnt.most_common(1)[0][0]
        if consensus_nucleotide == "-" and len(cnt) == 1:
            continue
        if consensus_nucleotide == "-":
            consensus_nucleotide = cnt.most_common(2)[1][0]
        consensus_seq += consensus_nucleotide
    return consensus_seq


def localization_score(clusters):
    n_clust = len(set(clusters))
    good_contacts = n_clust - 1
    i_prev = None
    for i in clusters:
        if i == i_prev:
            good_contacts += 1
        i_prev = i
    return round(good_contacts / len(clusters), 4)


def horhap_size_ratios(clusters):
    horhaps_cnt = [i[1] for i in Counter(clusters).most_common()]
    ratios = [round(i / len(clusters), 3) for i in horhaps_cnt]
    return ratios


def fcluster_horhaps(linkage_matrix, n_clust):
    distance_threshold = sorted(linkage_matrix[:, 2], reverse=True)[n_clust - 2] - 0.00001
    clusters = fcluster(linkage_matrix, t=distance_threshold, criterion="distance")
    return clusters


def get_horhap_divergence_from_pdist(index_subset, y):
    m = len(index_subset)
    if m <= 1:
        return 0.0

    idxs = sorted(index_subset)
    n = int((1 + np.sqrt(1 + 8 * len(y))) / 2)

    def cidx(i, j):
        return n * i - (i * (i + 1)) // 2 + (j - i - 1)

    vals = []
    for a in range(m - 1):
        i = idxs[a]
        for b in range(a + 1, m):
            j = idxs[b]
            vals.append(y[cidx(i, j)])

    return float(np.mean(vals)) if vals else 0.0


# ------------------------------------------------------------
# Cluster naming / colors
# ------------------------------------------------------------
def get_palette_for_k(n_clust):
    cmap = plt.get_cmap("tab10") if n_clust <= 10 else plt.get_cmap("tab20")
    return [cmap(i) for i in range(n_clust)]


def get_cluster_id_to_color_for_k(n_clust):
    palette = get_palette_for_k(n_clust)
    return {f"C{i + 1}": palette[i] for i in range(n_clust)}


def fcluster_labels_to_cluster_ids(clusters):
    uniq = sorted(set(clusters))
    raw_to_cluster_id = {raw_label: f"C{i + 1}" for i, raw_label in enumerate(uniq)}
    cluster_ids = [raw_to_cluster_id[x] for x in clusters]
    return cluster_ids, raw_to_cluster_id


def build_link_color_func(linkage_matrix, leaf_cluster_ids, cluster_id_to_color):
    n_leaves = linkage_matrix.shape[0] + 1
    node_color = {}

    for leaf_id in range(n_leaves):
        cluster_id = leaf_cluster_ids[leaf_id]
        node_color[leaf_id] = mcolors.to_hex(cluster_id_to_color[cluster_id])

    for i, (left, right, _, _) in enumerate(linkage_matrix):
        left = int(left)
        right = int(right)
        node_id = n_leaves + i

        c1 = node_color[left]
        c2 = node_color[right]
        node_color[node_id] = c1 if c1 == c2 else "black"

    return lambda node_id: node_color[node_id]


# ------------------------------------------------------------
# Mutation panel cache
# ------------------------------------------------------------
def get_leaf_order(linkage_matrix):
    fig, ax = plt.subplots(1, 1, figsize=(1, 1))
    dendr = dendrogram(
        linkage_matrix,
        no_labels=True,
        color_threshold=0,
        above_threshold_color="black",
        orientation="left",
        ax=ax,
    )
    plt.close(fig)
    return dendr["leaves"][::-1]



def build_mutation_image(alignment_list, leaf_order):
    consensus = get_consensus(alignment_list)
    n_seq = len(alignment_list)
    len_seq = len(consensus)

    seqs = [seq for _, seq in alignment_list]
    img = np.ones((n_seq, len_seq, 3), dtype=np.float32)

    nucl_colors = {
        "A": (0, 1, 0),
        "T": (1, 0, 0),
        "G": (0.67, 0, 1),
        "C": (0, 0, 1),
        "N": (0, 0, 0),
        "-": (0.67, 0.67, 0.67),
    }

    for row_idx, input_idx in enumerate(leaf_order):
        seq = seqs[input_idx]
        for col_idx, nucl in enumerate(seq):
            if nucl == "-":
                img[row_idx, col_idx] = nucl_colors["-"]
            elif nucl != consensus[col_idx]:
                img[row_idx, col_idx] = nucl_colors.get(nucl, nucl_colors["N"])

    return consensus, img


# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------
def process_and_plot(alignment_list, linkage_matrix, y_full, mutation_img, n_clust, output_prefix, verbose):
    if verbose:
        print(datetime.now(), "plotting", f"k={n_clust}")

    fig, ax = plt.subplots(1, 2, figsize=(14, 10), gridspec_kw={"width_ratios": [5, 25]})

    sys.setrecursionlimit(max(sys.getrecursionlimit(), len(alignment_list) * 5))

    clusters_input_order = fcluster_horhaps(linkage_matrix, n_clust)
    cluster_ids_input_order, _ = fcluster_labels_to_cluster_ids(clusters_input_order)
    cluster_id_to_color = get_cluster_id_to_color_for_k(n_clust)

    link_color_func = build_link_color_func(
        linkage_matrix=linkage_matrix,
        leaf_cluster_ids=cluster_ids_input_order,
        cluster_id_to_color=cluster_id_to_color,
    )

    dendrogram(
        linkage_matrix,
        truncate_mode="none",
        color_threshold=0,
        above_threshold_color="black",
        link_color_func=link_color_func,
        no_labels=True,
        orientation="left",
        ax=ax[0],
    )

    n_seq, len_seq = mutation_img.shape[:2]
    ax[1].imshow(
        mutation_img,
        interpolation="nearest",
        aspect="auto",
        origin="upper",
        extent=[0, len_seq, 0, n_seq],
    )

    ax[1].set_xlim([0, len_seq])
    ax[1].set_ylim([0, n_seq])
    ax[1].get_yaxis().set_visible(False)
#    ax[1].set_xlabel("Alignment position")
#    ax[1].set_title("Mutations / gaps vs consensus")
#    ax[0].set_title("Dendrogram")
    ax[0].set_yticks([])

    plt.savefig(output_prefix + "_clade_plot.png", bbox_inches="tight", dpi=700)
    plt.close()

    if verbose:
        print(datetime.now(), "writing cluster outputs")

    horhap_consensuses = {}
    horhap_divergence_values = {}

    horhaps_uniq = [f"C{i}" for i in range(1, n_clust + 1)]
    for horhap in horhaps_uniq:
        horhap_alignment = []
        horhap_indices = []
        for idx, (name, seq) in enumerate(alignment_list):
            if cluster_ids_input_order[idx] == horhap:
                horhap_alignment.append([name, seq])
                horhap_indices.append(idx)

        horhap_consensuses[horhap] = get_consensus(horhap_alignment)
        horhap_divergence_values[horhap] = get_horhap_divergence_from_pdist(horhap_indices, y_full)

        with open(f"{output_prefix}_{horhap}.fa", "w") as f:
            for name, seq in horhap_alignment:
                f.write(f">{name}\n{seq}\n")

    with open(f"{output_prefix}_cons.fa", "w") as f:
        for name in horhap_consensuses:
            f.write(f">{name}\n{horhap_consensuses[name]}\n")

    with open(f"{output_prefix}_divergence.tsv", "w") as f:
        for name in horhap_divergence_values:
            f.write(f"{name}\t{horhap_divergence_values[name]}\n")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Load saved distance/linkage .npz, cluster, and make simplified plots/outputs"
    )

    parser.add_argument(
        "--alignment",
        "-a",
        type=str,
        required=True,
        help="Path to the input alignment fasta file",
    )

    parser.add_argument(
        "--dist_npz",
        "-d",
        type=str,
        required=True,
        help="Path to .npz produced by process_alignment_simple.py (must contain y, Z, names)",
    )

    parser.add_argument("--output_prefix", "-o", type=str, required=True, help="Prefix for output files")

    parser.add_argument(
        "--number_of_horhaps",
        "-k",
        type=int,
        default=None,
        help="Number of horhaps. Otherwise selected by max_horhap_size or all k=2..20",
    )

    parser.add_argument(
        "--max_horhap_size",
        "-m",
        type=float,
        default=None,
        help="If k not set, choose smallest k where largest horhap < m (fraction). Else do all k=2..20",
    )

    parser.add_argument("-q", "--quiet", action="store_true", help="Turn off logs (verbosity ON by default)")

    args = parser.parse_args()
    verbose = not args.quiet

    if verbose:
        print(datetime.now(), "started")

    z = np.load(args.dist_npz, allow_pickle=True)
    if "Z" not in z or "y" not in z or "names" not in z:
        raise ValueError("dist_npz must contain arrays: y, Z, names")

    y_full = z["y"]
    linkage_matrix = z["Z"]
    names_saved = list(z["names"])

    alignment_list = read_alignment_fasta(args.alignment)

    if len(alignment_list) != len(names_saved):
        raise ValueError(
            f"FASTA and dist_npz contain different number of sequences: {len(alignment_list)} vs {len(names_saved)}"
        )

    leaf_order = get_leaf_order(linkage_matrix)
    _, mutation_img = build_mutation_image(alignment_list, leaf_order)

    if args.number_of_horhaps:
        n_clust = args.number_of_horhaps
        process_and_plot(alignment_list, linkage_matrix, y_full, mutation_img, n_clust, args.output_prefix, verbose)

    elif args.max_horhap_size:
        n_clust = None
        if verbose:
            print("k", "loc_score", "horhap_sizes", sep="\t")

        for k in range(2, 21):
            clusters = fcluster_horhaps(linkage_matrix, k)
            loc_score = localization_score(clusters)
            sizes = horhap_size_ratios(clusters)

            if verbose:
                print(k, loc_score, sizes, sep="\t")

            if sizes[0] < args.max_horhap_size:
                n_clust = k
                break

        if n_clust is None:
            n_clust = 20

        process_and_plot(alignment_list, linkage_matrix, y_full, mutation_img, n_clust, args.output_prefix, verbose)

    else:
        for k in range(2, 21):
            if verbose:
                print(datetime.now(), "k =", k)
            process_and_plot(
                alignment_list,
                linkage_matrix,
                y_full,
                mutation_img,
                k,
                f"k{k}_{args.output_prefix}",
                verbose,
            )


if __name__ == "__main__":
    main()
