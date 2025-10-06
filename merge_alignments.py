import argparse
from Bio.SeqIO import parse
from os import listdir
from re import finditer


def read_alignment(fasta):
    fl_cons = ''
    aln = {}
    for record in parse(fasta, 'fasta'):
        if 'CONS' in record.id or 'cons' in record.id:
            fl_cons = str(record.seq)
        else:
            aln[record.id] = record.seq
    return fl_cons, aln


def get_gap_positions(string):
    gaps_position_and_length = []
    for i in finditer(r'-+', string):
        start, end = i.span()
        gaps_position_and_length.append([start, end-start])
    return gaps_position_and_length


def insert_gap_to_alignments(alignments, gap_data):
    gap_start, gap_length, stv_of_gap = gap_data.pop(0)
    # insert gaps
    for stv in alignments:
        if stv != stv_of_gap:
            for hor_name in alignments[stv]:
                seq = alignments[stv][hor_name]
                alignments[stv][hor_name] = seq[:gap_start] + '-'*gap_length + seq[gap_start:]
    # edit gap list
    new_gap_data = []
    for gap in gap_data:
        if gap[2] == stv_of_gap:
            new_gap_data.append(gap)
        else:
            new_gap_start = gap[0] + gap_length
            gap = [new_gap_start] + gap[1:]
            new_gap_data.append(gap)
    return(alignments, new_gap_data)
        


def merge_alignments(datadir, output_alignment):
    # get parallel lists with fl hor and alignment from each file
    fl_hors = []
    alignments = {}
    gap_data = []
    for fasta in [i for i in listdir(datadir) if 'fa' in i]:
        fl_hor, alignment = read_alignment(datadir + '/' +fasta)
        fl_hors.append(fl_hor)
        alignments[fasta] = alignment
        gaps = get_gap_positions(fl_hor)
        gaps = [i + [fasta] for i in gaps]
        gap_data += gaps
    gap_data = sorted(gap_data, key=lambda x: x[0])
    
    # insert gaps loop
    for i in range(len(gap_data)):
        alignments, gap_data = insert_gap_to_alignments(alignments, gap_data)
    
    # write_alignment
    with open(output_alignment, 'w') as f:
        for stv in alignments:
            for hor_name in alignments[stv]:
                seq = alignments[stv][hor_name]
                f.write('>{}\n{}\n'.format(hor_name, seq))


def main():
    parser = argparse.ArgumentParser(description='Merge alignments of different StVs (with one full-length HOR each)')
    parser.add_argument('--alignment_dir', '-d', type=str, action='store', help='Path to dir with alignments (HORs of ONE StV with ONE full-length HOR in file)')
    parser.add_argument('--output_fasta', '-o', type=str, action='store', help='Path to output fasta file with merged alignment of all the StVs')
    args = parser.parse_args()

    merge_alignments(args.alignment_dir, args.output_fasta)  


if __name__ == '__main__':
    main()
