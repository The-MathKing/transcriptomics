import squidpy as sq
import scanpy as sc
import os

def main():
    print("Starting data acquisition Phase 2...")
    os.makedirs("data", exist_ok=True)
    
    # 1. Download scRNA-seq Reference: Mouse Cortex
    print("Downloading Single-Cell Reference (Mouse Cortex)...")
    adata_sc = sq.datasets.sc_mouse_cortex()
    print(f"Downloaded scRNA-seq: {adata_sc.n_obs} cells, {adata_sc.n_vars} genes")
    adata_sc.write("data/sc_mouse_cortex.h5ad")
    
    # 2. Download In-Distribution Spatial: 10x Visium Mouse Coronal Brain
    print("Downloading In-Distribution Spatial (10x Visium Mouse Coronal Brain)...")
    adata_visium = sq.datasets.visium_hne_adata()
    print(f"Downloaded Visium: {adata_visium.n_obs} spots, {adata_visium.n_vars} genes")
    adata_visium.write("data/visium_mouse_brain.h5ad")
    
    # 3. Download Shifted Spatial: Slide-seqV2 Mouse Hippocampus
    print("Downloading Shifted Spatial (Slide-seqV2 Mouse Hippocampus)...")
    adata_slideseq = sq.datasets.slide_seqv2()
    print(f"Downloaded Slide-seqV2: {adata_slideseq.n_obs} spots, {adata_slideseq.n_vars} genes")
    adata_slideseq.write("data/slideseqv2_mouse_hippocampus.h5ad")
    
    print("All datasets downloaded and saved to data/ directory successfully!")

if __name__ == "__main__":
    main()
