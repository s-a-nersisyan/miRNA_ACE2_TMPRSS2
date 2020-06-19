import scanpy as sc
import numpy as np


data = {
	"smillie19_epi.processed.h5ad": "Colon, epithelial",
	"james20.processed.h5ad": "Colon, immune cells",
	"wang20_rectum.processed.h5ad": "Rectum",

	"vento18_10x.processed.h5ad": "Placenta/Decidua",

	"martin19.processed.h5ad": "Ileum",

	"voigt19.processed.h5ad": "Retina",

	"vieira19_Alveoli_and_parenchyma_anonymised.processed.h5ad": "Lung parenchyma",
	"vieira19_Bronchi_anonymised.processed.h5ad": "Bronchi",
	"vieira19_Nasal_anonymised.processed.h5ad": "Nasal",

	"baron16.processed.h5ad": "Pancreas",
	"guo18_donor.processed.h5ad": "Testis",
	"henry18_0.processed.h5ad": "Prostate",
	"madissoon20_oesophagus.processed.h5ad": "Oesophagus",
	"madissoon20_spleen.processed.h5ad": "Spleen",
	"stewart19_adult.processed.h5ad": "Kidney"
}

g0 = "KDM5B"
g1 = "ACE2"
g2 = "TMPRSS2"

out_exprs = open("tables/scRNA-seq_expressions.tsv", "w")
out_dropouts = open("tables/scRNA-seq_dropouts.tsv", "w")

# Headers
out = ["Filename", "Organ", "Cell type", "Number of cells", g0, g1, g2]
print("\t".join(out), file=out_exprs)
print("\t".join(out), file=out_dropouts)

# Cell type columns
ct_columns = ["BroadCellType", "broad_celltype", "CellType", "cell_type", "Celltypes", "celltype"]


for fn, organ in data.items():
    adata = sc.read("data/COVID-19CellAtlas/{}".format(fn))

    # Select first matching cell type column
    for ct in ct_columns:
        if ct in adata.obs:
            ct_col = ct
            break

    if fn == "lukowski19.processed.h5ad":
        # This file was scaled using 10**3 factor
        adata.X = (adata.X.expm1() / 10).log1p()
    elif fn == "madissoon19_lung.processed.h5ad":
        # This file was not transformed at all
        library_sizes = np.sum(adata.X, axis=1)
        adata.X = (adata.X.multiply(1 / library_sizes).tocsc() * 10**4).log1p()
    elif fn.startswith("madissoon20"):
        # This file was not library size normalized
        library_sizes = adata.obs.n_counts.to_numpy()[:, None]
        adata.X = (adata.X.expm1().multiply(1 / library_sizes).tocsc() * 10**4).log1p()

    # Natural log -> binary log
    adata.X /= np.log(2)

    cell_types = np.unique(adata.obs[ct_col])
    for ct in cell_types:
        adata_ct = adata[adata.obs[ct_col] == ct, [g0, g1, g2]]
        X = adata_ct.X.toarray()

        x0 = X[:, 0]
        x1 = X[:, 1]
        x2 = X[:, 2]

        mean_cpm0 = np.mean(x0)
        mean_cpm1 = np.mean(x1)
        mean_cpm2 = np.mean(x2)

        out = [fn, organ, ct, len(x0), mean_cpm0, mean_cpm1, mean_cpm2]
        print("\t".join(map(str, out)), file=out_exprs)

        dropout_rate0 = len(x0[x0 == 0]) / len(x0)
        dropout_rate1 = len(x1[x1 == 0]) / len(x1)
        dropout_rate2 = len(x2[x2 == 0]) / len(x2)

        out = [fn, organ, ct, len(x0), dropout_rate0, dropout_rate1, dropout_rate2]
        print("\t".join(map(str, out)), file=out_dropouts)
