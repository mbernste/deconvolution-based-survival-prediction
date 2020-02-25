import sys
from optparse import OptionParser
import pandas as pd
import numpy as np
import os
from os.path import join
import json

sys.path.append('/Users/matthewbernstein/Development/single-cell-hackathon/src/common')

import load_GSE103224_TCGA_GBM
import run_DeconRNASeq

TOTAL_GENES = 1000

def main():
    usage = "" # TODO 
    parser = OptionParser(usage=usage)
    parser.add_option("-o", "--out_dir", help="Directory to write output")
    (options, args) = parser.parse_args()

    out_dir = options.out_dir

    filter_by = 'de'

    if filter_by == 'variance':
        # Filter genes
        top_expr = int(len(load_GSE103224_TCGA_GBM.GENE_NAMES)*0.2)
        gene_stat_df = pd.read_csv('gene_statistics.tsv', sep='\t', index_col=0)
        gene_stat_df = gene_stat_df.loc[[
            x.split('.')[0]
            for x in load_GSE103224_TCGA_GBM.GENE_IDS
        ]]
        gene_stat_df = gene_stat_df.iloc[:top_expr]
        gene_stat_df = gene_stat_df.sort_values(by='variance', ascending=False)
        gene_stat_df = gene_stat_df.iloc[:TOTAL_GENES]
        filt_gene_ids = gene_stat_df.index
        gene_id_to_index = {
            gene_id: index
            for index, gene_id in enumerate(load_GSE103224_TCGA_GBM.GENE_IDS)
        }
        keep_gene_inds = [
            gene_id_to_index[gene_id]
            for gene_id in filt_gene_ids
        ]
    elif filter_by == 'de':
        de_f = '../single-cell-hackathon/tmp/all_tumors_cluster_top_de_genes_aligned.json'
        with open(de_f, 'r') as f:
            clust_to_de = json.load(f)
        all_de_genes = set()
        for clust, de_genes in clust_to_de.items():
            all_de_genes.update(de_genes)
        keep_genes = sorted(all_de_genes)
        gene_name_to_index = {
            gene: index
            for index, gene in enumerate(load_GSE103224_TCGA_GBM.GENE_NAMES)
        }
        keep_gene_inds = [
            gene_name_to_index[gene]
            for gene in keep_genes
        ]

    cluster_f = '../single-cell-hackathon/tmp/all_tumors_aligned_clusters.tsv'
    cluster_df = pd.read_csv(cluster_f, sep='\t')

    clusts = sorted(set(cluster_df['louvain']))

    ref_mat = []
    for clust in clusts:
        cells = cluster_df.loc[cluster_df['louvain'] == clust]['cell']
        print('Loading {} cells for cluster {}...'.format(len(cells), clust))
        counts = load_GSE103224_TCGA_GBM.counts_matrix_for_sample_ids(cells) # Load from combined data
        print('done.')
        cpm = _compute_clust_cpm(counts)
        ref_mat.append(cpm)
    ref_mat = np.array(ref_mat)
    ref_mat = ref_mat[:,keep_gene_inds]
    print('Shape of reference matrix: ', ref_mat.shape)
    
    # Load and normalize the bulk data
    print('Loading bulk TCGA samples...')
    tcga_counts, tcga_samples = load_GSE103224_TCGA_GBM.counts_matrix_for_dataset('TCGA_GBM')
    print('done.')
    tcga_cpms = np.array([
        x/sum(x)
        for x in tcga_counts
    ])
    tcga_cpms *= 10e6
    tcga_cpms = tcga_cpms[:,keep_gene_inds]
    print('Shape of TCGA matrix: ', tcga_cpms.shape)



    # Format the data to feed to DeconRNASeq
    ref_df = pd.DataFrame(
        data=ref_mat.T,
        index=keep_genes,
        columns=clusts
    )
    query_df = pd.DataFrame(
        data=tcga_cpms.T,
        index=keep_genes,
        columns=tcga_samples
    )
    decon_res = run_DeconRNASeq.run_DeconRNASeq(ref_df, query_df)
    print(decon_res)
    decon_res.to_csv('deconvolved_TCGA_GBM.tsv', sep='\t')

def _compute_clust_cpm(counts):
    clust_counts = np.sum(counts, axis=0)
    depth = np.sum(clust_counts)
    cpm = np.array([
        x/float(depth)
        for x in clust_counts
    ])
    cpm *= 10e6
    return cpm
    
    

if __name__ == "__main__":
    main()
