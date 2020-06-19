from natsort import natsorted

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.stats import spearmanr


def load_TargetScan(gene):
    # Load TargetScan predictions for gene
    return set(open("data/TargetScan_{}.txt".format(gene)).read().strip().split("\n"))


def load_paired_tables(path):
    # Open isomiR / mRNA TCGA tables and select paired samples
    df_isomiR = pd.read_csv("{}/isomiR_normal.tsv".format(path), sep="\t", index_col=0)
    df_mRNA = pd.read_csv("{}/mRNA_normal.tsv".format(path), sep="\t", index_col=0)
    paired_samples = natsorted(list(set(df_isomiR.columns).intersection(df_mRNA.columns)))
    df_isomiR = df_isomiR[paired_samples]
    df_mRNA = df_mRNA[paired_samples]

    return df_isomiR, df_mRNA


def load_tables_for_organ(organ):
    # Merge all TCGA samples for specified organ
    global organs_projects, isomiR_thr_quantile

    dfs_isomiR, dfs_mRNA = [], []
    for project in organs_projects[organ]:
        df_isomiR, df_mRNA = load_paired_tables("data/TCGA/{}".format(project))

        # Select highly expressed isomiR's
        medians = df_isomiR.median(axis=1)
        isomiR_thr = np.quantile(medians[medians > 0], isomiR_thr_quantile)
        df_isomiR = df_isomiR.loc[ df_isomiR.median(axis=1) >= isomiR_thr ]

        dfs_isomiR.append(df_isomiR)
        dfs_mRNA.append(df_mRNA)

    common_isomiRs = set.intersection(*[set(df_isomiR.index) for df_isomiR in dfs_isomiR])
    dfs_isomiR = [df_isomiR.loc[common_isomiRs] for df_isomiR in dfs_isomiR]

    df_isomiR = pd.concat(dfs_isomiR, axis=1)
    df_mRNA = pd.concat(dfs_mRNA, axis=1)
    return df_isomiR, df_mRNA


def show_gene_in_organs(gene, ax, lab):
    # Draw boxplot and export statistical summary to the table (S1 Table)
    matrix = []
    for organ in organs_projects:
        df_isomiR, df_mRNA = load_tables_for_organ(organ)

        for e in df_mRNA.loc[gene].tolist():
            matrix.append([organ, e])

    coln = gene + ", log(FPKM-UQ)"
    df = pd.DataFrame(matrix, columns=["Organ", coln])

    medians = df.groupby("Organ").median()
    order = medians.sort_values(coln, ascending=False).index

    cmap = sns.color_palette("Blues")

    table = [["Organ", "Minimum", "Q1", "Q2", "Q3", "Maximum", "Mean", "Standard deviation"]]
    for o in order:
        x = df.loc[df.Organ == o][coln]
        table.append([o, np.min(x), np.quantile(x, 0.25), np.quantile(x, 0.5), np.quantile(x, 0.75), np.max(x), np.mean(x), np.std(x)])

    print("\n".join(["\t".join(list(map(str, row))) for row in table]), file=open("tables/S1_{}.tsv".format(gene), "w"))

    #mpl.rcParams["figure.figsize"] = 4, 5
    p = sns.boxplot(x="Organ", y=coln, data=df, order=order, color=cmap[3], saturation=1, ax=ax)
    p.set_xticklabels(p.get_xticklabels(), rotation=45, ha="right")
    p.set_xlabel("")
    ax.set_title(lab, loc="left", fontdict={"fontsize": "xx-large", "fontweight": "bold"})


def get_regulators_of_genes(genes):
    # Correlation analysis on miRNA -> gene interactions
    global R_thr

    # Load TargetScan and the list of intronic sense miRNAs
    db_regulators = {gene: load_TargetScan(gene) for gene in genes}
    df = pd.read_csv("data/intronic_sense_miRNA.tsv", sep="\t", header=None)
    host_genes = {miRNA: gene for gene, miRNA in zip(df[0], df[1])}

    matrix = []
    for organ in organs_projects:
        df_isomiR, df_mRNA = load_tables_for_organ(organ)

        for gene in genes:
            isomiRs = [isomiR for isomiR in df_isomiR.index if isomiR.split("|")[0] in db_regulators[gene]]

            for isomiR in isomiRs:
                x = df_isomiR.loc[isomiR]
                y = df_mRNA.loc[gene]
                R, pvalue = spearmanr(x, y)

                host_gene = host_genes.get(isomiR.split("|")[0])

                if R <= R_thr and pvalue < 0.05:
                    matrix.append([organ, isomiR, gene, np.median(x), np.median(y), R, pvalue, host_gene])

    return pd.DataFrame(
        matrix,
        columns=["Organ", "isomiR", "Gene", "isomiR median expression", "Gene median expression", "R", "p-value", "Host gene"]
    )


def jaccard(u, v):
    return 1 - np.sum((u > 0) & (v > 0)) / len(u)


def plot_clustermap(gene, lab):
    global df

    cmap = sns.color_palette("Blues")

    df1 = df.loc[df["Gene"] == gene][["Organ", "isomiR", "isomiR median expression"]]
    df1 = df1.pivot(index="Organ", columns="isomiR", values="isomiR median expression")

    p = sns.clustermap(
            df1.T, metric=jaccard, dendrogram_ratio=(.05, .2),
            cmap=cmap, figsize=(7.5, 8), vmin=0, vmax=10, linecolor="lightgrey", linewidth=0.1,
            cbar_kws={"label": "log(R-UQ)"},
    )
    plt.setp(p.ax_heatmap.get_xticklabels(), rotation=45, ha="right")
    plt.setp(p.ax_heatmap.get_yticklabels(), rotation=0)
    p.ax_heatmap.set_xlabel("")
    p.ax_heatmap.set_ylabel("")
    #p.ax_heatmap.set_yticklabels("")
    plt.title(lab, loc="left", fontdict={"fontsize": "xx-large", "fontweight": "bold"})

    plt.tight_layout()
    plt.savefig("figures/Fig2_{}.tif".format(gene), format="tiff", pil_kwargs={"compression": "tiff_lzw"})
    plt.close()


organs_projects = {
    "Bladder": ["TCGA-BLCA"],
    "Breast": ["TCGA-BRCA"],
    "Colon": ["TCGA-COAD"],
    "Uterine corpus": ["TCGA-UCEC"],
    "Esophagus": ["TCGA-ESCA"],
    "Kidney": ["TCGA-KICH", "TCGA-KIRC", "TCGA-KIRP"],
    "Liver": ["TCGA-LIHC"],
    "Lung": ["TCGA-LUAD", "TCGA-LUSC"],
    "Prostate": ["TCGA-PRAD"],
    "Stomach": ["TCGA-STAD"],
    "Thyroid": ["TCGA-THCA"]
}

# Take top-20% isomiRs
isomiR_thr_quantile = 0.8
# Cutoff on Spearman correlation
R_thr = -0.3


if __name__ == "__main__":
    mpl.rcParams["font.sans-serif"] = "Arial"
    mpl.rcParams["figure.dpi"] = 300
    mpl.rcParams['axes.titlepad'] = 20

    # Draw figure 1 (ACE2 and TMPRRS2 expression in organgs)
    mpl.rcParams["font.sans-serif"] = "Arial"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    show_gene_in_organs("ACE2", ax1, "A")
    show_gene_in_organs("TMPRSS2", ax2, "B")

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2)
    plt.savefig("figures/Fig1.tif", format="tiff", pil_kwargs={"compression": "tiff_lzw"})
    plt.close()

    # Get miRNA regulators of ACE2 / TMPRSS2 and plot clustermaps
    df = get_regulators_of_genes(["ACE2", "TMPRSS2"])
    df.to_csv("tables/S2_Table.tsv", sep="\t", index=None)

    plot_clustermap("ACE2", "A")
    plot_clustermap("TMPRSS2", "B")

    # Draw figure 3 (scatterplots of interesting interactions)
    to_draw = [
        ("Esophagus", "hsa-miR-125a-5p|0", "ACE2"),
        ("Kidney", "hsa-miR-125a-5p|0", "ACE2"),
        ("Lung", "hsa-miR-125a-5p|0", "ACE2"),
        ("Liver", "hsa-miR-199a-5p|0", "TMPRSS2"),
        ("Stomach", "hsa-miR-199a-5p|0", "TMPRSS2"),
        ("Uterine corpus", "hsa-miR-199a-5p|0", "TMPRSS2"),
    ]

    fig, axs = plt.subplots(2, 3, figsize=(10, 6))
    axs = [ax for axx in axs for ax in axx]
    for (organ, isomiR, gene), ax in zip(to_draw, axs):
        df_isomiR, df_mRNA = load_tables_for_organ(organ)
        x = df_isomiR.loc[isomiR]
        y = df_mRNA.loc[gene]
        sns.regplot(x, y, ax=ax)
        ax.set_xlabel("{}, log(R-UQ)".format(isomiR.split("|")[0]))
        ax.set_ylabel("{}, log(FPKM-UQ)".format(gene))
        ax.set_title(organ, fontdict={"fontweight": "bold"})

    plt.tight_layout()
    plt.savefig("figures/Fig3.tif", format="tiff", pil_kwargs={"compression": "tiff_lzw"})
    plt.close()
