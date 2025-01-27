import argparse
import numpy as np
import matplotlib.pyplot as plt

from Bio.SeqIO import parse
from collections import Counter
from datetime import datetime
from matplotlib.patches import Rectangle
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from tqdm import tqdm

# Calculates distance matrix and performs clustering and makes dendrogram with split on a certain depth to form given number of horhaps
# Requierd fasta header format stv::chr:start-end (e.g. S2C8H1L.1-11::chr8:44244904-44246772)


# S2C8H1L.1-11::chr8:44244904-44246772
def read_alignment_fasta(alignment_file):
    mfa = [[record.id, str(record.seq).upper()] for record in parse(alignment_file, 'fasta')]
    # make correct header format
    if '::' not in mfa[0][0]:
        mfa = [['stv::'+i[0], i[1]] for i in mfa]
    # sort by coord
    mfa = sorted(mfa, key=lambda x: (x[0].split('::')[1].split(':')[0], int(x[0].split('-')[-1])))
    return mfa


def get_consensus(mfa):
    consensus_seq = ''
    len_seq = len(mfa[0][1])
    for nucl_index in range(len_seq):
        cnt = Counter([seq[nucl_index] for header, seq in mfa])
        consensus_nucleotide = cnt.most_common(1)[0][0]
        if consensus_nucleotide == '-' and len(cnt) == 1:
            continue
        if consensus_nucleotide == '-':
            consensus_nucleotide = cnt.most_common(2)[1][0]
        consensus_seq += consensus_nucleotide
    return consensus_seq


# Calculate Hamming distance only at positions where neither sequence has a gap
# But use all the length for validating
def hamming_no_gap_dist(u, v):
    #valid_positions = ((u != '-') & (v != '-')).sum()
    mismatches      = ((u != v) & (u != '-') & (v != '-')).sum()
    return mismatches / len(u)


# Get ranges of '-' in sequence to plot gaps faster
def find_dash_ranges(s):
    # find indexes of '-'
    positions = np.where(np.array(list(s)) == '-')[0]
    if len(positions) == 0:
        return []
    # get gaps
    breaks = np.diff(positions) > 1
    range_starts = np.insert(positions[1:][breaks], 0, positions[0])
    range_ends = np.append(positions[:-1][breaks], positions[-1])
    # merge gaps
    return list(zip(range_starts, range_ends))


# Proportion of contats btw one horhap
def localiztion_score(clusters):
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

# Convert sequences to numerical vectors
def seq_to_numeric_vector(seq):
    return np.array([ord(nucl) for nucl in seq])

def alignment_to_linkage(alignment_list, pdist_metric, linkage_method, verbosity):
    # Alignment to matrix
    sequence_vectors = np.array([np.array(list(i[1])) for i in alignment_list])
    # Calculate pairwise Euclidean distances
    if verbosity:
        print(datetime.now(), 'calculating distance')
    euclidean_distances = pdist(sequence_vectors, metric=pdist_metric)
    # Perform hierarchical clustering
    if verbosity:
        print(datetime.now(), 'clustering')
    linkage_matrix = linkage(euclidean_distances, method=linkage_method)
    return linkage_matrix


def get_hohrhap_divergence(alignment_list, pdist_metric, verbosity):
    # Alignment to matrix
    sequence_vectors = np.array([np.array(list(i[1])) for i in alignment_list])
    # Calculate pairwise Euclidean distances
    if verbosity:
        print(datetime.now(), 'calculating distance for horhap')
    euclidean_distances = pdist(sequence_vectors, metric=pdist_metric)
    return np.mean(euclidean_distances)
    
    


def fcluster_horhaps(linkage_matrix, n_clust):
    # pick split depth based on given number of clusters
    distance_threshold = sorted(linkage_matrix[:, 2], reverse=True)[n_clust - 2] - 0.00001
    # get clusters
    clusters = fcluster(linkage_matrix, t=distance_threshold, criterion='distance')
    return clusters


def plot_maps_of_few_k(linkage_matrix, output_prefix, verbosity):
    
    fig, ax = plt.subplots(1,8, figsize=(15, 10))
    
    # set up color pallete same as in scipy dendrogram
    color_pallete = [plt.get_cmap("tab10")(i) for i in range(10)]
    colors = {i: color_pallete[i] for i in range(len(color_pallete))}
    for n_clust in tqdm(range(2, 10), disable=not verbosity):
        ax_id = n_clust - 2
        # pick split depth based on given number of clusters
        distance_threshold = sorted(linkage_matrix[:, 2], reverse=True)[n_clust - 2] - 0.00001
        # get clusters
        clusters = fcluster(linkage_matrix, t=distance_threshold, criterion='distance')
        n_seq = len(clusters)
        # plot map of one k
        for i in range(n_seq):
            horhap = clusters[i]
            y_position = n_seq - 1 * (i+1)
            color = colors[horhap]
            ax[ax_id].add_patch(Rectangle(xy=(0, y_position), width=1, height=1, facecolor=color))
            ax[ax_id].set_ylim([0, n_seq])
            ax[ax_id].get_xaxis().set_visible(False)
            ax[ax_id].text(0.5, -1, 'k={}'.format(n_clust), va='top', ha='right')
    
    plt.savefig(output_prefix + '_choose_k.png', bbox_inches='tight', dpi=200)
    plt.close()


    ### SET YTICK LABELS ###
def set_y_tick_labels(ax, alignment_list, plot_seq_names=False):
    chroms_cnt = Counter([i[0].split('::')[1].split(':')[0] for i in alignment_list])
    chroms_sorted = sorted(chroms_cnt.keys())
    tick_label = []
    for chrom in chroms_sorted:
        chrom_hor_cnt = chroms_cnt[chrom]
        tick_label += [i for i in range(chrom_hor_cnt)]
    
    tick_number = np.array([i for i in range(len(alignment_list))])
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

    # add chrom names
    if plot_seq_names:
        for chrom, start in zip(chroms_sorted, start_indices[::-1]):
            ax.text(1.5, start, chrom, rotation=90, va='top', ha='right')
    #######


def process_and_plot(alignment_list, linkage_matrix, n_clust, output_prefix, verbosity):

    consensus = get_consensus(alignment_list)
    
    # Pick split depth based on given number of clusters
    distance_threshold = sorted(linkage_matrix[:, 2], reverse=True)[n_clust - 2]
        
    if verbosity:
        print(datetime.now(), 'plotting1')
    fig, ax = plt.subplots(1,3, figsize=(15, 10), gridspec_kw={'width_ratios': [5, 25, 1]})
    
    # plot dendrogram
    dendr = dendrogram(
        linkage_matrix,
        truncate_mode = 'none',
        color_threshold = distance_threshold,
        no_labels = True,
        orientation = 'left',
        ax=ax[0],
    )
    start, end = ax[0].get_xlim()

    # sort alignment by dendrogram
    alignment_dict = {i[0]: i[1] for i in alignment_list}
    hor_names = [i[0] for i in alignment_list]
    hor_names_sorted = [hor_names[int(i)] for i in dendr['ivl']][::-1]
    dendr_colors = dendr['leaves_color_list'][::-1]
    # dimensions
    n_seq = len(alignment_dict)
    len_seq = len(consensus)
    one_nucl_height = 1# / n_seq
    one_nucl_width  = 1# / len_seq
    
    # plot alignment
    # setup IVG nucleotide colors
    nucl_colors_dict_bright = {
        'A': (0, 1, 0),
        'T': (1, 0, 0),
        'G': (0.67, 0, 1),
        'C': (0, 0, 1),
        '-': (.67, .67, .67)
    }


    # plot mutations
    for seq_index in tqdm(range(len(hor_names_sorted)), disable=not verbosity):
        hor_name = hor_names_sorted[seq_index]
        hor_seq = alignment_dict[hor_name]
        y_position = n_seq - one_nucl_height * (seq_index+1)
        for nucl_index in range(len_seq):
            nucl = hor_seq[nucl_index] 
            if nucl != consensus[nucl_index] and nucl != '-':
                # determine X position
                x_position = one_nucl_width * nucl_index
                color = nucl_colors_dict_bright[nucl]
                # plot nucleotide
                ax[1].add_patch(Rectangle(xy=(x_position, y_position), width=one_nucl_width, height=one_nucl_height, facecolor=color))
                #ax[1].add_patch(Rectangle(xy=(x_position-one_nucl_width*2, y_position), width=one_nucl_width*4, height=one_nucl_height, facecolor=color))
        # plot gaps
        color = nucl_colors_dict_bright['-']
        gap_ranges = find_dash_ranges(hor_seq)
        for start, end in gap_ranges:
            x_position = start
            gap_width = end - start
            ax[1].add_patch(Rectangle(xy=(x_position, y_position), width=gap_width, height=one_nucl_height, facecolor=color)) 

    ax[1].set_xlim([0, len_seq])
    ax[1].set_ylim([0, n_seq])
    ax[1].get_yaxis().set_visible(False)

    # plot map
    names_and_group = {hor_names_sorted[i]: dendr_colors[i] for i in range(len(hor_names_sorted))}
    color_pallete = [plt.get_cmap("tab10")(i) for i in range(10)]
    colors = {'C'+str(i): color_pallete[i] for i in range(len(color_pallete))}
    for i in range(len(hor_names)):
        name = hor_names[i]
        y_position = n_seq - one_nucl_height * (i+1)
        color = colors[names_and_group[name]]
        ax[2].add_patch(Rectangle(xy=(0, y_position), width=one_nucl_width, height=one_nucl_height, facecolor=color))
    ax[2].set_ylim([0, n_seq])
    ax[2].get_xaxis().set_visible(False)
    
    set_y_tick_labels(ax[2], alignment_list, plot_seq_names=True)

    plt.savefig(output_prefix + '_clade_plot.png', bbox_inches='tight', dpi=700)
    plt.close()
    
    # write bed
    with open(output_prefix + '_horhap.bed', 'w') as f:
        for i in range(len(hor_names)):
            name = hor_names[i]
            horhap = names_and_group[name]
            color = colors[horhap]
            color = ','.join([str(int(i*255)) for i in color[:3]])
            stv, name = name.split('::')
            chrom, coords = name.split(':')
            start, end = coords.split('-')
            name = '{}::{}'.format(stv, horhap)
            line = '\t'.join([chrom, start, end, name, '100', '+', start, end, color])
            f.write(line + '\n')

    
    if verbosity:
        print(datetime.now(), 'plotting2')
    fig, ax = plt.subplots(1,2, figsize=(15, 10), gridspec_kw={'width_ratios': [1, 10]})

    # plot map
    names_and_group = {hor_names_sorted[i]: dendr_colors[i] for i in range(len(hor_names_sorted))}
    color_pallete = [plt.get_cmap("tab10")(i) for i in range(10)]
    colors = {'C'+str(i): color_pallete[i] for i in range(len(color_pallete))}
    for i in range(len(hor_names)):
        name = hor_names[i]
        y_position = n_seq - one_nucl_height * (i+1)
        color = colors[names_and_group[name]]
        #color = 'blue'
        ax[0].add_patch(Rectangle(xy=(0, y_position), width=1, height=one_nucl_height, facecolor=color))
    ax[0].set_ylim([0, n_seq])
    ax[0].get_xaxis().set_visible(False)

    set_y_tick_labels(ax[0], alignment_list, plot_seq_names=True)

    # plot mutations
    for seq_index in tqdm(range(n_seq), disable=not verbosity):
        hor_name = hor_names[seq_index]
        hor_seq = alignment_dict[hor_name]
        y_position = n_seq - one_nucl_height * (seq_index+1)
        for nucl_index in range(len_seq):
            nucl = hor_seq[nucl_index] 
            if nucl != consensus[nucl_index] and nucl != '-':
                # determine X position
                x_position = one_nucl_width * nucl_index
                color = nucl_colors_dict_bright[nucl]
                # plot nucleotide
                ax[1].add_patch(Rectangle(xy=(x_position, y_position), width=one_nucl_width, height=one_nucl_height, facecolor=color))
        # plot gaps
        color = nucl_colors_dict_bright['-']
        gap_ranges = find_dash_ranges(hor_seq)
        for start, end in gap_ranges:
            x_position = start
            gap_width = end - start
            ax[1].add_patch(Rectangle(xy=(x_position, y_position), width=gap_width, height=one_nucl_height, facecolor=color))

    ax[1].set_xlim([0, len_seq])
    ax[1].set_ylim([0, n_seq])
    
    set_y_tick_labels(ax[1], alignment_list, plot_seq_names=False)

    plt.savefig(output_prefix + '_coord_plot.png', bbox_inches='tight', dpi=700)
    plt.close()



    # get horhap alignment and consensuses
    horhaps = [names_and_group[i] for i in hor_names]
    horhap_alignments = {}
    horhap_consensuses = {}
    horhap_divergence_values = {} 
    horhaps_uniq = list(set(horhaps))
    for horhap in horhaps_uniq:
        horhap_alignment = []
        for i_hor, i_horhap in zip(hor_names, horhaps):
            if i_horhap == horhap:
                horhap_alignment.append([i_hor, alignment_dict[i_hor]])
        horhap_consensuses[horhap] = get_consensus(horhap_alignment)
        horhap_alignments[horhap] = [horhap_alignment][0]
        # horhap divergence
        horhap_divergence = get_hohrhap_divergence(horhap_alignment, hamming_no_gap_dist, verbosity)
        horhap_divergence_values[horhap] = horhap_divergence
        # write alignment of horhap (all stv in one fasta)
        #with open('{}_{}.fa'.format(output_prefix, horhap), 'w') as f:
        #    for name, seq in horhap_alignment:
        #        f.write('>{}\n{}\n'.format(name, seq))
        # write by-stv alignment of horhap
        by_stv_horhap_alignment = {}
        for name, seq in horhap_alignment:
            stv = name.split('::')[0]
            if stv not in by_stv_horhap_alignment:
                by_stv_horhap_alignment[stv] = [[name, seq]]
            else:
                by_stv_horhap_alignment[stv].append([name, seq])
        for stv in by_stv_horhap_alignment:
            # duplicate seq if it's the only seq in the alignments (for hmmbuild)
            if len(by_stv_horhap_alignment[stv]) == 1:
                by_stv_horhap_alignment[stv].append(by_stv_horhap_alignment[stv][0])
            with open('{}_{}::{}.fa'.format(output_prefix, stv.replace('/', 'h'), horhap), 'w') as f:
                for name, seq in by_stv_horhap_alignment[stv]:
                    f.write('>{}\n{}\n'.format(name, seq))
    # write horhap cons file
    with open('{}_cons.fa'.format(output_prefix), 'w') as f:
        for name in horhap_consensuses:
            f.write('>{}\n{}\n'.format(name, horhap_consensuses[name]))
    # write horhap divergence file
    with open('{}_divergence.tsv'.format(output_prefix), 'w') as f:
        for name in horhap_divergence_values:
            f.write('{}\t{}\n'.format(name, horhap_divergence_values[name]))



def main():
    parser = argparse.ArgumentParser(description='Calculates distance matrix and performs clustering and makes dendrogram with line-split to form given number of horhaps')

    parser.add_argument('--alignment', '-a', type=str, action='store', required=True,
                        help='Path to the input alignment fasta file. Requierd fasta header format stv::chr:start-end (e.g. S2C8H1L.1-11::chr8:44244904-44246772)')

    parser.add_argument('--output_prefix', '-o', type=str, action='store', required=True,
                        help='Prefix for output files')

    #parser.add_argument('--pdist_metric', '-p', type=str, action='store',
    #                    help='[Optional, default=hamming_no_gap_dist] metric to calculate distance matrix',
    #                    default=hamming_no_gap_dist)
    
    parser.add_argument('--linkage_method', '-l', type=str, action='store',
                        help='[Optional, default=ward] method to perform hierarchical/agglomerative clustering. The script is optimized for ward',
                        default='ward')
    
    parser.add_argument('--number_of_horhaps', '-k', type=int, action='store',
                        help='[Optional] Number of horhaps. Otherwise will be selected automatically based on max_horhap_size or all k options from 2 to 9 will be used',
                        default=None) 
    
    parser.add_argument('--max_horhap_size', '-m', type=float, action='store',
                        help='[Optional] If number_of_horhaps is not set, k will be selected when biggest horhap forms less than max_horhap_size of the array. If not set as well, all k from 2 to 9 will be used',
                        default=None) 

    parser.add_argument('--verbosity', '-v', action='store_true', default=False,
                        help='[Optinal, default=False] Show log') 
        
    args = parser.parse_args()
    
    # read alignment
    alignment_list = read_alignment_fasta(args.alignment)
    
    # pick pdist_metric
    pdist_metric = hamming_no_gap_dist
    
    # calculate linkage matrix
    linkage_matrix = alignment_to_linkage(alignment_list, pdist_metric, args.linkage_method, args.verbosity)

    # plot maps of different k
    plot_maps_of_few_k(linkage_matrix, args.output_prefix, args.verbosity)

    # plot if k passed
    if args.number_of_horhaps:
        n_clust = args.number_of_horhaps
        process_and_plot(alignment_list, linkage_matrix, n_clust, args.output_prefix, args.verbosity)
    # pick k by max_horhap_size
    elif args.max_horhap_size:
        if args.verbosity:
            print('k', 'loc_score', 'horhap_sizes', sep='\t')
        for k in range(2,10):
            clusters = fcluster_horhaps(linkage_matrix, k)
            loc_score = localiztion_score(clusters)
            sizes = horhap_size_ratios(clusters)
            if args.verbosity:
                print(k, loc_score, sizes, sep='\t')
            if sizes[0] < args.max_horhap_size:
                n_clust = k
                break
        process_and_plot(alignment_list, linkage_matrix, n_clust, args.output_prefix, args.verbosity)
    # plot all the k from 1 to 9
    else:
        for k in range(2,10):
            if args.verbosity:
                print('k =', k)
            process_and_plot(alignment_list, linkage_matrix, k, 'k{}_{}'.format(str(k), args.output_prefix), args.verbosity)
    

if __name__ == '__main__':
    main()
