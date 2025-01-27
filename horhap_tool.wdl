version 1.0 


workflow horhaps {
    input {
        File fasta 
        File stv_bed
        String chromosome
        Float stv_perc_cutoff = 1  # 1 percent

        String file_name    = basename(sub(sub(sub(fasta, "\\.gz$", ""), "\\.fasta$", ""), "\\.fa$", ""))
    }

    ## for a chromosome, pull top two StVs over stv_perc_cutoff
    call align_hors {
        input:
            fasta           = fasta,
            stv_bed         = stv_bed,
            stv_perc_cutoff = stv_perc_cutoff,
            chromosome      = chromosome,
            file_name       = file_name
    }

    call get_horhaps {
        input:
            alignment     = align_hors.aligned_fasta_out
    }

    output {
        File selected_hors_bed   = align_hors.selected_hors_bed
        File selected_hors_fasta = align_hors.selected_hors_fasta
        File aligned_fasta       = align_hors.aligned_fasta_out
        File output_dir          = get_horhaps.output_dir
    }
}



task align_hors {
    input {
        File fasta
        File stv_bed
        Float stv_perc_cutoff
        String chromosome
        String file_name 

        ## runtime parameters 
        Int threadCount    = 2
        Int memSizeGB      = 8        
        Int diskSize       = 32
        Int preemptible    = 1                
    }

    String output_fasta  = "~{file_name}_~{chromosome}_selected.fa"
    String output_bed    = "~{file_name}_~{chromosome}_selected.bed"
    String aligned_fasta = "~{file_name}_~{chromosome}.fa"

    command <<<

        ## read in the bed file, pull rows with the hor name we are targeting
        ## find start and end, break the array into tiles
        ## pull seqs_per_tile and write selected StVs to a bed file.
        ## also write tiles to bed file.

        # First filter for just the requested chromosome and count occurrences of each name
        grep "~{chromosome}_" ~{stv_bed} | \
            cut -f4 | sort | uniq -c > name_counts.txt
        
        # Calculate total number of StVs in chromosome for percentage calculation
        total_entries=$(awk '{sum += $1} END {print sum}' name_counts.txt)
        
        # Calculate cutoff count based on percentage
        cutoff_count=$(echo "scale=0; (~{stv_perc_cutoff} * $total_entries / 100)" | bc)
        if [ "$cutoff_count" -eq 0 ]; then
            cutoff_count=1
        fi

        # Get only the top 2 names that are above the cutoff, sorted by frequency (highest first)
        awk -v cutoff="$cutoff_count" '$1 >= cutoff {print $1, $2}' name_counts.txt | \
            sort -nr | \
            head -n 2 | \
            awk '{print $2}' > frequent_names.txt

        # Filter the original bed file to only keep rows with frequent names
        grep "~{chromosome}_" ~{stv_bed} | \
            awk 'BEGIN{while(getline < "frequent_names.txt"){names[$1]=1}} $4 in names' > ~{output_bed}

        ## Get the fasta (each region in the bed is one entry in the fasta)
        bedtools getfasta \
            -name \
            -bed ~{output_bed} \
            -fi ~{fasta} \
            > ~{output_fasta}
        
        ## Align
        /muscle3 \
            -in ~{output_fasta} \
            -out ~{aligned_fasta} \
            -quiet 

    >>>
    output {
        File selected_hors_bed   = "~{output_bed}"
        File selected_hors_fasta = "~{output_fasta}"
        File aligned_fasta_out   = "~{aligned_fasta}"
    }
    runtime {
        cpu: threadCount
        memory: memSizeGB + " GB"
        preemptible : preemptible
        disks: "local-disk " + diskSize + " SSD"
        docker: "fedorrik/horhap_align@sha256:2e4439d723f659e7cf5ca792146842c16b7c44eccc899f7802de52d318e8e9fd"
    }
}


task get_horhaps {
    input {
        File alignment
        
        ## runtime parameters 
        Int threadCount    = 2
        Int memSizeGB      = 8        
        Int diskSize       = 32
        Int preemptible    = 1   
    }

    String output_prefix = basename(alignment, ".fa")
    
    command <<< 

        # download script
        wget -q https://raw.githubusercontent.com/fedorrik/horhap_tool/refs/heads/main/cluster_alignment_ward.py

        # run script
        python cluster_alignment_ward.py \
            --verbosity \
            --alignment ~{alignment} \
            --output_prefix ~{output_prefix}
        
        # by-stv alignments to hmm
        for k in {2..9}; do
          touch "k${k}_~{output_prefix}_horhaps.hmm"
          for fasta in k${k}_*.fa; do
            hmmbuild hmm ${fasta} > /dev/null
            cat hmm >> "k${k}_~{output_prefix}_horhaps.hmm"
            rm hmm
          done
        done


        # put outpluts to dir and compress
        mkdir "output_~{output_prefix}"
        mv "~{output_prefix}_choose_k.png" k* "output_~{output_prefix}"
        tar czf "output_~{output_prefix}.tar.gz" "output_~{output_prefix}"


    >>>

    output {
        File output_dir = "output_~{output_prefix}.tar.gz"
    }

    runtime {
        cpu: threadCount
        memory: memSizeGB + " GB"
        preemptible : preemptible
        disks: "local-disk " + diskSize + " SSD"
        docker: "fedorrik/get_horhaps@sha256:d495be6e32c00aac0975e17fedf4bdba15117875017cf30e9329490cd8c3e91a"
    }
}

