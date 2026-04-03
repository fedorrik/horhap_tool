FROM python:3.9-slim

MAINTAINER Fedor Ryabov, fryabov@ucsc.edu

RUN apt-get update && apt-get install -y \
    wget \
    hmmer \
    seqtk \

    && rm -rf /var/lib/apt/lists/*

RUN pip install biopython scipy numpy pandas scikit-learn hdbscan matplotlib tqdm

WORKDIR /opt/

RUN wget \
    https://public.gi.ucsc.edu/~fryabov/horhap/hmmertblout2bed.awk \
    https://public.gi.ucsc.edu/~fryabov/horhap/overlap_filter.py \
    https://public.gi.ucsc.edu/~fryabov/horhap/bedmap && chmod +x bedmap

WORKDIR /data
