import scanpy as sc
import squidpy as sq
import os

def download_human_data():
    os.makedirs("data", exist_ok=True)
    
    print("Downloading Human Visium Breast Cancer dataset...")
    adata_visium = sc.datasets.visium_sge('V1_Breast_Cancer_Block_A_Section_1')
    adata_visium.write("data/human_breast_cancer_visium.h5ad")
    print(f"Downloaded Visium: {adata_visium.n_obs} spots, {adata_visium.n_vars} genes")

    # For the single-cell reference, we will download a standard human PBMC or breast cancer dataset
    # Since scanpy doesn't have a built-in breast cancer scRNA-seq, let's download the PBMC dataset for now 
    # to serve as a placeholder for the pipeline construction.
    print("Downloading Human PBMC single-cell reference...")
    adata_sc = sc.datasets.pbmc3k()
    adata_sc.write("data/human_pbmc_sc.h5ad")
    print(f"Downloaded scRNA-seq: {adata_sc.n_obs} cells, {adata_sc.n_vars} genes")

if __name__ == "__main__":
    download_human_data()
