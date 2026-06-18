# HOR haplotype test workflow

This directory contains a small example run for `horhap_tool`.

## Requirements

External tools used while preparing the example:

- `bedtools`
- `muscle3`

Python packages used by the scripts:

- `biopython`
- `matplotlib`
- `numpy`
- `scipy`
- `tqdm`

## Inputs

The main input is an aligned FASTA file of AS-HOR sequences:

- `asat_hors.mfa`

The tool expects FASTA headers with genomic coordinates in this format:

```text
stv::chr:start-end
```

If the `stv::` prefix is missing, the scripts add it automatically. Strand labels such as `(+)` and `(-)` are removed from sequence names.

## Preparing `asat_hors.fa`

To create `asat_hors.fa`, first annotate the assembly with an AS-HOR monomer annotation tool, for example:

```text
https://github.com/fedorrik/HumAS-HMMER_for_AnVIL
```

Then merge monomers from the BED file into HORs, for example with:

```text
https://github.com/fedorrik/stv
```

Extract HOR sequences from the assembly:

```bash
bedtools getfasta -s -name -bed stv_raw.bed -fi assembly.fa > asat_hors.fa
```

This creates FASTA headers with the coordinate format needed by this tool.

Align the HOR sequences:

```bash
muscle3 -in asat_hors.fa -out asat_hors.mfa
```

## Run the test

From this directory:

```bash
python3 ../process_alignment.py -a asat_hors.mfa -o test
```

This reads the alignment, computes pairwise distances, and writes:

- `test.npz`: compressed distance/linkage data for clustering
- `test_distance_matrix.tsv`: square distance matrix

Then run clustering:

```bash
python3 ../clustering.py -a asat_hors.mfa -o test -d test.npz
```

With no `-k/--number_of_horhaps` or `-m/--max_horhap_size`, clustering writes outputs for all `k=2..20`.

## Outputs

`clustering.py` creates these output directories next to the output prefix:

- `pngs/`: dendrogram and mutation plots
- `horhap_beds/`: BED tracks with HOR haplotype assignments
- `fastas/`: FASTA files for each HOR haplotype
- `cons_fa/`: consensus FASTA files
- `divergence_tsvs/`: average within-haplotype divergence tables
