#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sys
from pathlib import Path

from Bio.SeqIO import parse
from collections import Counter
from datetime import datetime
from matplotlib.patches import Rectangle
from scipy.cluster.hierarchy import dendrogram, fcluster
from tqdm import tqdm


# ------------------------------------------------------------
# IO / parsing
# ------------------------------------------------------------
# Required fasta header format: stv::chr:start-end
# Example: S2C8H1L.1-11::chr8:44244904-44246772
def read_alignment_fasta(alignment_file):
    mfa = [[record.id, str(record.seq).upper()] for record in parse(alignment_file, "fasta")]
    if not mfa:
        raise ValueError(f"Empty FASTA: {alignment_file}")

    if "::" not in mfa[0][0]:
        mfa = [["stv::" + i[0], i[1]] for i in mfa]

    # remove (+) and (-) from names
    for i in mfa:
        i[0] = i[0].replace("(+)", "").replace("(-)", "")

    mfa = sorted(mfa, key=lambda x: (x[0].split("::")[1].split(":")[0], int(x[0].split("-")[-1])))
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


def find_dash_ranges(s):
    positions = np.where(np.array(list(s)) == "-")[0]
    if len(positions) == 0:
        return []
    breaks = np.diff(positions) > 1
    range_starts = np.insert(positions[1:][breaks], 0, positions[0])
    range_ends = np.append(positions[:-1][breaks], positions[-1])
    return list(zip(range_starts, range_ends))


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
    """
    Return flat clusters targeting n_clust groups.

    Using criterion="distance" with an epsilon below a merge height can overshoot when
    several merges share the same height. criterion="maxclust" is stable for this use
    and avoids accidental C19/C20 labels when k=18 was requested.
    """
    clusters = fcluster(linkage_matrix, t=n_clust, criterion="maxclust")
    return clusters


def prepare_output_paths(output_prefix):
    output_prefix = Path(output_prefix)
    parent_dir = output_prefix.parent if str(output_prefix.parent) != '' else Path('.')
    base_name = output_prefix.name

    out_dirs = {
        "beds": parent_dir / "horhap_beds",
        "pngs": parent_dir / "pngs",
        "fastas": parent_dir / "fastas",
        "cons": parent_dir / "cons_fa",
        "divergence": parent_dir / "divergence_tsvs",
    }

    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    return base_name, out_dirs


def get_hohrhap_divergence_from_pdist(names_subset, name_to_idx, y, verbose):
    """
    Compute mean pairwise distance within a horhap using the GLOBAL pdist condensed vector y.
    """
    m = len(names_subset)
    if m <= 1:
        return 0.0

    idxs = sorted(name_to_idx[n] for n in names_subset)
    n = len(name_to_idx)

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
    """
    Map raw fcluster labels to C1..Ck preserving raw fcluster label order:
    smallest raw label -> C1, next raw label -> C2, ...
    """
    uniq = sorted(set(clusters))
    raw_to_cluster_id = {raw_label: f"C{i + 1}" for i, raw_label in enumerate(uniq)}
    cluster_ids = [raw_to_cluster_id[x] for x in clusters]
    return cluster_ids, raw_to_cluster_id


def build_link_color_func(linkage_matrix, leaf_cluster_ids, cluster_id_to_color):
    """
    leaf_cluster_ids must be in the original observation order used to build linkage_matrix.

    Rule:
    - leaf gets color of its cluster
    - internal node gets that same color if both children have same color
    - otherwise internal node is black
    """
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
# Plotting
# ------------------------------------------------------------
def plot_maps_of_few_k(linkage_matrix, output_prefix, verbose):
    output_base, out_dirs = prepare_output_paths(output_prefix)
    ks = list(range(2, 21))
    fig, ax = plt.subplots(1, len(ks), figsize=(2 * len(ks), 10))

    if len(ks) == 1:
        ax = [ax]

    for n_clust in tqdm(ks, disable=not verbose):
        ax_id = n_clust - 2
        clusters = fcluster_horhaps(linkage_matrix, n_clust)
        cluster_ids, _ = fcluster_labels_to_cluster_ids(clusters)
        cluster_id_to_color = get_cluster_id_to_color_for_k(n_clust)
        n_seq = len(clusters)

        for i in range(n_seq):
            cluster_id = cluster_ids[i]
            color = cluster_id_to_color[cluster_id]
            y_position = n_seq - (i + 1)
            ax[ax_id].add_patch(Rectangle(xy=(0, y_position), width=1, height=1, facecolor=color))

        ax[ax_id].set_ylim([0, n_seq])
        ax[ax_id].set_xlim([0, 1])
        ax[ax_id].get_xaxis().set_visible(False)
        ax[ax_id].get_yaxis().set_visible(False)
        ax[ax_id].text(0.5, -1, f"k={n_clust}", va="top", ha="center")

    plt.savefig(out_dirs["pngs"] / f"{output_base}_choose_k.png", bbox_inches="tight", dpi=200)
    plt.close()


def set_y_tick_labels(ax, alignment_list, plot_seq_names=False):
    chroms_cnt = Counter([i[0].split("::")[1].split(":")[0] for i in alignment_list])
    chroms_sorted = sorted(chroms_cnt.keys())
    tick_label = []
    for chrom in chroms_sorted:
        chrom_hor_cnt = chroms_cnt[chrom]
        tick_label += [i for i in range(chrom_hor_cnt)]

    tick_label = np.array(tick_label[::-1])
    start_indices = np.where(tick_label == 0)[0]

    tick_number = []
    start_prev = 0
    for i in range(len(start_indices)):
        n = start_indices[i]
        while n > start_prev:
            tick_number.append(n)
            n -= 100
        start_prev = start_indices[i]

    tick_label = tick_label[tick_number]
    ax.set_yticks(tick_number)
    ax.set_yticklabels(tick_label)

    if plot_seq_names:
        for chrom, start in zip(chroms_sorted, start_indices[::-1]):
            ax.text(
                1.02, start, chrom,
                rotation=0,
                va="top",
                ha="left",
                transform=ax.get_yaxis_transform()
            )


def get_dendrogram_leaf_order(linkage_matrix):
    """
    Get stable leaf order once. Coloring changes with k, but leaf order does not.
    """
    dendr = dendrogram(
        linkage_matrix,
        truncate_mode="none",
        color_threshold=0,
        above_threshold_color="black",
        no_labels=True,
        orientation="left",
        no_plot=True,
    )
    return dendr["leaves"][::-1]


def build_mutation_image(alignment_list, hor_names_sorted, verbose):
    """
    Render mutation/gap matrix once as an RGB image in fixed dendrogram order.
    Reuse this image for all k.
    """
    consensus = get_consensus(alignment_list)
    alignment_dict = {i[0]: i[1] for i in alignment_list}

    n_seq = len(hor_names_sorted)
    len_seq = len(consensus)
    img = np.ones((n_seq, len_seq, 3), dtype=np.float32)

    nucl_colors_dict_bright = {
        "A": (0, 1, 0),
        "T": (1, 0, 0),
        "G": (0.67, 0, 1),
        "C": (0, 0, 1),
        "N": (0, 0, 0),
        "-": (0.67, 0.67, 0.67),
    }

    for seq_index in tqdm(range(n_seq), disable=not verbose):
        hor_name = hor_names_sorted[seq_index]
        hor_seq = alignment_dict[hor_name]

        for nucl_index in range(len_seq):
            nucl = hor_seq[nucl_index]
            if nucl == "-":
                continue
            if nucl != consensus[nucl_index]:
                img[seq_index, nucl_index, :] = nucl_colors_dict_bright.get(nucl, (0, 0, 0))

        for start, end in find_dash_ranges(hor_seq):
            img[seq_index, start:end + 1, :] = nucl_colors_dict_bright["-"]

    return img, consensus


def draw_mutation_panel(ax, mutation_img):
    n_seq, len_seq = mutation_img.shape[:2]
    ax.imshow(
        mutation_img,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        extent=(0, len_seq, 0, n_seq),
    )
    ax.set_xlim([0, len_seq])
    ax.set_ylim([0, n_seq])
    ax.get_yaxis().set_visible(False)


def draw_track_panel(ax, hor_names, names_and_group, cluster_id_to_color):
    n_seq = len(hor_names)
    one_nucl_height = 1
    one_nucl_width = 1

    for i, name in enumerate(hor_names):
        group = names_and_group[name]
        color = cluster_id_to_color[group]
        y_position = n_seq - one_nucl_height * (i + 1)
        ax.add_patch(
            Rectangle(
                xy=(0, y_position),
                width=one_nucl_width,
                height=one_nucl_height,
                facecolor=color,
            )
        )

    ax.set_ylim([0, n_seq])
    ax.get_xaxis().set_visible(False)


def process_and_plot(alignment_list, linkage_matrix, y_full, name_to_idx, n_clust, output_prefix, verbose,
                     mutation_img=None, leaf_order_top_to_bottom=None):
    output_base, out_dirs = prepare_output_paths(output_prefix)
    if verbose:
        print(datetime.now(), "plotting1")

    fig, ax = plt.subplots(1, 3, figsize=(15, 10), gridspec_kw={"width_ratios": [5, 25, 1]})

    sys.setrecursionlimit(max(sys.getrecursionlimit(), len(alignment_list) * 5))

    alignment_dict = {i[0]: i[1] for i in alignment_list}
    hor_names = [i[0] for i in alignment_list]

    clusters_input_order = fcluster_horhaps(linkage_matrix, n_clust)
    cluster_ids_input_order, _ = fcluster_labels_to_cluster_ids(clusters_input_order)
    actual_n_clust = len(set(cluster_ids_input_order))

    names_and_group = {name: cluster_id for name, cluster_id in zip(hor_names, cluster_ids_input_order)}
    cluster_id_to_color = get_cluster_id_to_color_for_k(actual_n_clust)

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

    if mutation_img is None or leaf_order_top_to_bottom is None:
        leaf_order_top_to_bottom = get_dendrogram_leaf_order(linkage_matrix)
        hor_names_sorted = [hor_names[i] for i in leaf_order_top_to_bottom]
        mutation_img, _ = build_mutation_image(alignment_list, hor_names_sorted, verbose)

    draw_mutation_panel(ax[1], mutation_img)
    draw_track_panel(ax[2], hor_names, names_and_group, cluster_id_to_color)
    set_y_tick_labels(ax[2], alignment_list, plot_seq_names=True)

    png_path = out_dirs["pngs"] / f"{output_base}_clade_plot.png"
    plt.savefig(png_path, bbox_inches="tight", dpi=700)
    plt.close()

    # write bed
    bed_path = out_dirs["beds"] / f"{output_base}_horhap.bed"
    with open(bed_path, "w") as f:
        for name in hor_names:
            horhap = names_and_group[name]
            color = cluster_id_to_color[horhap]
            color_rgb = ",".join([str(int(c * 255)) for c in color[:3]])

            stv, rest = name.split("::", 1)
            chrom, coords = rest.split(":", 1)
            start, end = coords.split("-", 1)

            out_name = f"{stv}::{horhap}"
            line = "\t".join([chrom, start, end, out_name, "100", "+", start, end, color_rgb])
            f.write(line + "\n")

    if verbose:
        print(datetime.now(), "writing horhap fastas/cons/divergence")

    # per horhap: consensus + divergence + write by-stv fastas
    horhap_alignments = {}
    horhap_consensuses = {}
    horhap_divergence_values = {}

    horhaps_uniq = [f"C{i}" for i in range(1, actual_n_clust + 1)]
    for horhap in horhaps_uniq:
        horhap_alignment = []
        for i_hor in hor_names:
            if names_and_group[i_hor] == horhap:
                horhap_alignment.append([i_hor, alignment_dict[i_hor]])

        horhap_alignments[horhap] = horhap_alignment
        horhap_consensuses[horhap] = get_consensus(horhap_alignment)

        names_subset = [n for n, _ in horhap_alignment]
        horhap_divergence_values[horhap] = get_hohrhap_divergence_from_pdist(
            names_subset, name_to_idx, y_full, verbose
        )

        by_stv = {}
        for name, seq in horhap_alignment:
            stv = name.split("::")[0]
            by_stv.setdefault(stv, []).append([name, seq])

        for stv in by_stv:
            if len(by_stv[stv]) == 1:
                by_stv[stv].append(by_stv[stv][0])

            fasta_path = out_dirs["fastas"] / f"{output_base}_{stv.replace('/', 'h')}::{horhap}.fa"
            with open(fasta_path, "w") as f:
                for name, seq in by_stv[stv]:
                    f.write(f">{name}\n{seq}\n")

    cons_path = out_dirs["cons"] / f"{output_base}_cons.fa"
    with open(cons_path, "w") as f:
        for name in horhap_consensuses:
            f.write(f">{name}\n{horhap_consensuses[name]}\n")

    divergence_path = out_dirs["divergence"] / f"{output_base}_divergence.tsv"
    with open(divergence_path, "w") as f:
        for name in horhap_divergence_values:
            f.write(f"{name}\t{horhap_divergence_values[name]}\n")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Parse saved distance/linkage .npz, cluster, plot, and write horhap outputs"
    )

    parser.add_argument(
        "--alignment",
        "-a",
        type=str,
        required=True,
        help="Path to the input alignment fasta file. Required header format stv::chr:start-end",
    )

    parser.add_argument(
        "--dist_npz",
        "-d",
        type=str,
        required=True,
        help="Path to .npz produced by 01_make_dist.py (must contain y, Z, names)",
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

    # load npz with pdist + linkage + names order
    z = np.load(args.dist_npz, allow_pickle=True)
    if "Z" not in z or "y" not in z or "names" not in z:
        raise ValueError("dist_npz must contain arrays: y, Z, names")

    y_full = z["y"]
    linkage_matrix = z["Z"]
    names_saved = list(z["names"])

    # read alignment and reorder to match names_saved
    alignment_list_raw = read_alignment_fasta(args.alignment)
    seq_map = {h: s for h, s in alignment_list_raw}

    missing = [h for h in names_saved if h not in seq_map]
    if missing:
        raise ValueError(f"These names exist in dist_npz but not in FASTA: {missing[:5]} ... total={len(missing)}")

    alignment_list = [[h, seq_map[h]] for h in names_saved]

    # mapping for divergence-from-pdist
    name_to_idx = {name: i for i, name in enumerate(names_saved)}

    # plot mutation matrix once in fixed dendrogram order; reuse for all k
    if verbose:
        print(datetime.now(), "precomputing mutation matrix")
    leaf_order_top_to_bottom = get_dendrogram_leaf_order(linkage_matrix)
    hor_names = [i[0] for i in alignment_list]
    hor_names_sorted = [hor_names[i] for i in leaf_order_top_to_bottom]
    mutation_img, _ = build_mutation_image(alignment_list, hor_names_sorted, verbose)

    # plot maps of different k (2..20)
    #plot_maps_of_few_k(linkage_matrix, args.output_prefix, verbose)

    # run chosen k mode
    if args.number_of_horhaps:
        n_clust = args.number_of_horhaps
        process_and_plot(alignment_list, linkage_matrix, y_full, name_to_idx, n_clust, args.output_prefix, verbose, mutation_img=mutation_img, leaf_order_top_to_bottom=leaf_order_top_to_bottom)

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

        process_and_plot(alignment_list, linkage_matrix, y_full, name_to_idx, n_clust, args.output_prefix, verbose, mutation_img=mutation_img, leaf_order_top_to_bottom=leaf_order_top_to_bottom)

    else:
        for k in range(2, 21):
            if verbose:
                print(datetime.now(), "k =", k)
            process_and_plot(
                alignment_list,
                linkage_matrix,
                y_full,
                name_to_idx,
                k,
                f"k{k}_{args.output_prefix}",
                verbose,
                mutation_img=mutation_img,
                leaf_order_top_to_bottom=leaf_order_top_to_bottom,
            )


if __name__ == "__main__":
    main()