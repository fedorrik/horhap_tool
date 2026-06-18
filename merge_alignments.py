import argparse
from Bio.SeqIO import parse
from os import listdir


def read_alignment(fasta):
    fl_cons = ''
    aln = {}
    for record in parse(fasta, 'fasta'):
        if 'CONS' in record.id or 'cons' in record.id:
            fl_cons = str(record.seq)
        else:
            aln[record.id] = record.seq
    return fl_cons, aln


def get_gap_counts_by_ref_pos(fl_cons):
    gap_counts = {0: 0}
    ref_pos = 0
    for nt in fl_cons:
        if nt == '-':
            gap_counts[ref_pos] = gap_counts.get(ref_pos, 0) + 1
        else:
            ref_pos += 1
            gap_counts.setdefault(ref_pos, 0)
    return gap_counts


def project_sequence_to_gap_plan(seq, fl_cons, merged_gap_counts):
    seq = str(seq)
    if len(seq) != len(fl_cons):
        raise ValueError('Sequence and CONS alignment lengths do not match')

    seq_out = []
    ref_pos = 0
    local_gap_count = 0

    for seq_nt, cons_nt in zip(seq, fl_cons):
        if cons_nt == '-':
            seq_out.append(seq_nt)
            local_gap_count += 1
        else:
            seq_out.append('-' * (merged_gap_counts.get(ref_pos, 0) - local_gap_count))
            seq_out.append(seq_nt)
            ref_pos += 1
            local_gap_count = 0

    seq_out.append('-' * (merged_gap_counts.get(ref_pos, 0) - local_gap_count))
    return ''.join(seq_out)


def merge_alignments(datadir, output_alignment):
    # get full-length consensus and alignment from each file
    fl_hors = {}
    alignments = {}
    for fasta in [i for i in listdir(datadir) if 'fa' in i]:
        fl_hor, alignment = read_alignment(datadir + '/' +fasta)
        fl_hors[fasta] = fl_hor
        alignments[fasta] = alignment

    ungapped_fl_hors = {i.replace('-', '') for i in fl_hors.values()}
    if len(ungapped_fl_hors) != 1:
        raise ValueError('All CONS sequences must be identical after removing gaps')

    merged_gap_counts = {}
    for fl_hor in fl_hors.values():
        for ref_pos, gap_count in get_gap_counts_by_ref_pos(fl_hor).items():
            merged_gap_counts[ref_pos] = max(merged_gap_counts.get(ref_pos, 0), gap_count)

    for fasta, alignment in alignments.items():
        fl_hor = fl_hors[fasta]
        for hor_name, seq in alignment.items():
            alignment[hor_name] = project_sequence_to_gap_plan(seq, fl_hor, merged_gap_counts)
    
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
