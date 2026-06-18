# horhap_tool

Tools that take alignments of AS-HORs, compute pairwise Hamming distance matrices, and perform Ward's hierarchical clustering.

### Current Version

The current version includes the following scripts:

- `process_alignment.py`
- `clustering.py`

### Simple Version
- `simple/` - Simple version which has no requirements to sequence headers. It can make clustering for any alignment but it doesn't make bed file and track plots.

### Test

- `test/` - Small example run with the input files and commands needed to process an alignment and generate HOR haplotype outputs.

### Older Version

- `wdl/` - Older version (deprecated)

### Citation

If you use this tool, please cite these preprints, where it was first described and used:

- https://doi.org/10.64898/2025.12.14.693655
- https://doi.org/10.64898/2026.03.27.714900
