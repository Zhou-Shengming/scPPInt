# scPPInt

**PPI-guided Integration and Imputation of Single-cell Proteomes with Inconsistent Protein Coverage**

scPPInt is a Python framework for integrating paired single-cell proteomics datasets with incomplete and inconsistent protein coverage. It jointly learns cell representations, protein representations, and a reconstructed expression matrix defined on the union of proteins from both sources. The framework uses a protein–protein interaction (PPI) network as a structural prior and supports downstream cell-level, protein-level, covariation, and spatial proteomics analyses.

## Contents

- [Overview](#overview)
- [Method](#method)
- [Study highlights](#study-highlights)
- [Repository structure](#repository-structure)
- [Environment](#environment)
- [Input format](#input-format)
- [Quick start](#quick-start)
- [Step 1: Prepare integration data](#step-1-prepare-integration-data)
- [Step 2: Retrieve a STRING PPI network](#step-2-retrieve-a-string-ppi-network)
- [Step 3: Run single-cell integration](#step-3-run-single-cell-integration)
- [Step 4: Run spatial integration](#step-4-run-spatial-integration)
- [Command-line parameters](#command-line-parameters)
- [Output files](#output-files)
- [Example datasets](#example-datasets)
- [Data sources](#data-sources)
- [Reproducibility notes](#reproducibility-notes)
- [Limitations](#limitations)
- [Citation](#citation)
- [License and contact](#license-and-contact)

## Overview

Single-cell proteomics is affected by sparse measurements, missing values, batch effects, and differences in protein coverage between experiments. Restricting integration to shared proteins can discard source-specific measurements, whereas treating unobserved positions as true zeros can confound technical non-detection with low abundance.

scPPInt addresses these issues by:

1. Aligning two datasets on the union of their protein identifiers.
2. Using a mask to distinguish observed measurements from zero placeholders introduced during alignment.
3. Learning protein embeddings with a dual-branch graph autoencoder on a PPI network.
4. Learning cell embeddings with a cell autoencoder.
5. Coupling the protein and cell branches through joint cell–protein reconstruction.
6. Applying PPI consistency, within-batch structure preservation, and batch-mixing regularization constraints.
7. Exporting an integrated protein expression matrix for downstream analysis.

## Method

The implementation contains two coupled components:

- **Dual-branch protein graph autoencoder:** learns source-specific protein representations and a fused protein representation over the protein union using graph convolutions and edge-aware Transformer convolutions.
- **Cell autoencoder:** extracts compact cell or spatial-location embeddings using one-dimensional convolution, multi-head self-attention, residual connections, layer normalization, and batch normalization.

The training objective combines the following terms:

- Mask-Aware Reconstruction Loss
- PPI Consistency Loss
- Within-Batch Structure Preservation Loss
- Batch-Mixing Regularization Loss
- Joint cell–protein reconstruction based on the inner product of cell and protein embeddings

The default loss weights are:

```text
rec_loss_w       = 1.0
kmeans_loss_w    = 1.0
kmeans_intra_w   = 1.0
kmeans_inter_w   = 1.0
kmeans_margin    = 1.0
corr_struct_w    = 0.01
ppi_loss_w       = 10.0
```

![scPPInt framework](Figure/scPPInt.jpg)

## Study highlights

The manuscript evaluates scPPInt on several paired proteomics settings:

- **N2 and nanoPOTS:** mouse single-cell proteomics with C10, RAW, and SVEC cell types. The manuscript reports clear cell-type separation, improved batch mixing, bidirectional label transfer, and recovery of selected source-specific marker and differential proteins.
- **Leduc_2024 and Leduc_2025:** mouse airway-related single-cell proteomics used to assess non-shared protein completion, covariation reconstruction, and consistency with PPI-supported protein relationships.
- **plexDIA and pSCoPE pairings:** human single-cell proteomics used to examine robustness across platforms and experimental years, including ablation analyses of the loss terms.
- **Human tonsil spatial proteomics:** 16 ROI regions with 22 antigen channels and 84,085 cells, used to evaluate antigen-channel reconstruction, covariation, UMAP structure, and major spatial enrichment patterns.

The reported results emphasize a balance among cell-type structure preservation, batch mixing, cross-dataset label transfer, protein completion, and protein-level covariation. The exact metrics and evaluation settings are described in the manuscript associated with this repository.

## Repository structure

```text
scPPInt/
├── data/
│   ├── N2+nanoPOTS/
│   ├── Leduc_2024+Leduc_2025/
│   ├── plexDIA_2023+pSCoPE_2022/
│   ├── plexDIA_2023+pSCoPE_2023/
│   ├── pSCoPE_2022+pSCoPE_2023/
│   ├── ROI/
│   ├── prepare_integration_data.py
│   ├── retrieve_ppi_network.py
├── model/
│   ├── protein_model.py
│   ├── cell_model_1.py
│   ├── single_cell_run.py
│   ├── spatial_run.py
├── Figure/
│   └── scPPInt.jpg
├── environment.yml
└── README.md
```

`protein_model.py` and `cell_model_1.py` must remain in the same directory as the two model-running scripts because both scripts import these modules locally.

## Environment

The repository provides an exported Conda environment in [`environment.yml`](environment.yml). It was exported from a Linux CUDA environment. The main versions used in the development environment include:

| Package | Version |
|---|---:|
| Python | 3.8.20 |
| PyTorch | 2.4.1+cu121 |
| PyTorch Geometric | 2.6.1 |
| NumPy | 1.24.4 |
| pandas | 2.0.3 |
| scikit-learn | 1.3.2 |
| openpyxl | 3.1.5 |
| requests | 2.32.4 |

The exported environment also contains CUDA 12.1 libraries and PyTorch Geometric extension packages. The exact installation command for PyTorch and PyTorch Geometric may depend on the CUDA driver and operating system of the target machine.

### Create the Conda environment

```bash
conda env create -f environment.yml
conda activate integration
```

The `prefix` field in the exported file records the original environment location and may not be valid on another machine. If Conda reports a path-related issue, remove the `prefix` line from a local copy of `environment.yml` before creating the environment.

At minimum, the model scripts require:

```text
numpy
pandas
scikit-learn
openpyxl
torch
torch-geometric
```

The PPI retrieval script additionally requires:

```text
requests
```

## Input format

### Expression matrices

Expression CSV files should use the following layout:

```text
              cell_or_location_1  cell_or_location_2  ...
protein_1     value               value
protein_2     value               value
...           ...                 ...
```

That is:

- rows are proteins or antigen channels;
- columns are cells or spatial locations;
- the first column is used as the row index;
- identifiers must be consistent across `merged_data`, the source matrices, and the PPI file.

The data-preparation script can detect whether raw matrices have cells in rows or columns by matching identifiers against the annotation-file index, then writes matrices in the protein-by-cell format expected by the model.

### Cell or spatial-location annotations

Each annotation file uses the cell or spatial-location identifier as its index. It must contain:

```text
cell_type
cell_batch
```

Additional columns, such as spatial coordinates, may be retained for downstream analysis. For labels written as `C10(N2)`, the preparation script extracts:

```text
cell_type = C10
cell_batch = N2
```

### PPI edge table

The PPI CSV must contain:

```text
protein1,protein2,score
```

`score` is expected to be a numeric STRING confidence score normalized to the interval `[0, 1]` by the retrieval script. Only PPI edges whose endpoints are present in `merged_data` are used by the model.

## Quick start

Run commands from the repository root:

```bash
cd scPPInt
```

For the included N2+nanoPOTS example, the complete workflow is:

```bash
python "data/prepare_integration_data.py" \
  --x1_data "data/N2+nanoPOTS/N2_data.csv" \
  --x1_cell "data/N2+nanoPOTS/N2_cell.csv" \
  --x2_data "data/N2+nanoPOTS/nanoPOTS_data.csv" \
  --x2_cell "data/N2+nanoPOTS/nanoPOTS_cell.csv"

python "data/retrieve_ppi_network.py" \
  --merged_data "outputs/merged_data.csv" \
  --species 10090

python "model/single_cell_run.py" \
  --merged_data "data/N2+nanoPOTS/merged_data.csv" \
  --x1_data "data/N2+nanoPOTS/N2_data.csv" \
  --x2_data "data/N2+nanoPOTS/nanoPOTS_data.csv" \
  --x1_cell "data/N2+nanoPOTS/N2_cell.csv" \
  --x2_cell "data/N2+nanoPOTS/nanoPOTS_cell.csv" \
  --ppi "data/N2+nanoPOTS/ppi_edges_protein_ids.csv"
```

The example data are already prepared in `data/N2+nanoPOTS`. Therefore, if the six example files and PPI network are already present, the preparation and retrieval steps can be skipped and the model can be run directly.

## Step 1: Prepare integration data

Use [`data/prepare_integration_data.py`](data/prepare_integration_data.py) to prepare two raw datasets.

The script has exactly four required input files:

```text
--x1_data
--x1_cell
--x2_data
--x2_cell
```

Example:

```bash
python "data/prepare_integration_data.py" \
  --x1_data "data/N2+nanoPOTS/N2_data.csv" \
  --x1_cell "data/N2+nanoPOTS/N2_cell.csv" \
  --x2_data "data/N2+nanoPOTS/nanoPOTS_data.csv" \
  --x2_cell "data/N2+nanoPOTS/nanoPOTS_cell.csv"
```

Optional parameters:

| Parameter | Default | Description |
|---|---:|---|
| `--x1_name` | `N2` | Output prefix for dataset 1 |
| `--x2_name` | `nanoPOTS` | Output prefix for dataset 2 |
| `--output_dir` | `./outputs` | Output directory; created automatically |

The default output files are:

```text
outputs/N2_data.csv
outputs/N2_cell.csv
outputs/nanoPOTS_data.csv
outputs/nanoPOTS_cell.csv
outputs/merged_data.csv
```

## Step 2: Retrieve a STRING PPI network

Use [`data/retrieve_ppi_network.py`](data/retrieve_ppi_network.py) to obtain a PPI edge table for the proteins in a merged expression matrix.

The script has exactly two required parameters:

```text
--merged_data
--species
```

For mouse data, use taxonomy ID `10090`:

```bash
python "data/retrieve_ppi_network.py" \
  --merged_data "outputs/merged_data.csv" \
  --species 10090
```

For human data, use taxonomy ID `9606`:

```bash
python "data/retrieve_ppi_network.py" \
  --merged_data "outputs/merged_data.csv" \
  --species 9606
```

The script maps input identifiers to STRING IDs, retrieves network edges in batches, filters interactions using the STRING confidence threshold, canonicalizes undirected edges, and retains the maximum score for duplicated endpoint pairs.

Optional parameters:

| Parameter | Default | Description |
|---|---:|---|
| `--output_dir` | `./outputs` | Output directory; created automatically |
| `--output_name` | `ppi_edges_protein_ids.csv` | Output edge-table filename |
| `--score_threshold` | `700` | STRING score threshold, from 0 to 1000 |
| `--batch_size_map` | `100` | Batch size for identifier mapping |
| `--batch_size_net` | `200` | Batch size for network requests |
| `--connect_timeout` | `10` | Connection timeout in seconds |
| `--read_timeout` | `60` | Read timeout in seconds |
| `--max_retries` | `5` | Maximum retries per request |

This step requires network access to the STRING database API.

## Step 3: Run single-cell integration

Use [`model/single_cell_run.py`](model/single_cell_run.py) for a single-cell proteomics dataset pair.

The six required input files are:

```text
--merged_data
--x1_data
--x2_data
--x1_cell
--x2_cell
--ppi
```

N2+nanoPOTS example:

```bash
python "model/single_cell_run.py" \
  --merged_data "data/N2+nanoPOTS/merged_data.csv" \
  --x1_data "data/N2+nanoPOTS/N2_data.csv" \
  --x2_data "data/N2+nanoPOTS/nanoPOTS_data.csv" \
  --x1_cell "data/N2+nanoPOTS/N2_cell.csv" \
  --x2_cell "data/N2+nanoPOTS/nanoPOTS_cell.csv" \
  --ppi "data/N2+nanoPOTS/ppi_edges_protein_ids.csv"
```

The output directory is created automatically. If `--output_dir` is omitted, outputs are written to:

```text
./outputs
```

## Step 4: Run spatial integration

Use [`model/spatial_run.py`](model/spatial_run.py) for a spatial proteomics or ROI data pair.

The command-line format and the six required inputs are the same as for the single-cell model. The only command-level difference is the script name:

```text
model/single_cell_run.py  -> single-cell proteomics
model/spatial_run.py      -> spatial proteomics
```

For spatial data, replace the script and the six paths in the single-cell command:

```bash
python "model/spatial_run.py" \
  --merged_data "path/to/spatial/merged_data.csv" \
  --x1_data "path/to/spatial/x1_data.csv" \
  --x2_data "path/to/spatial/x2_data.csv" \
  --x1_cell "path/to/spatial/x1_cell.csv" \
  --x2_cell "path/to/spatial/x2_cell.csv" \
  --ppi "path/to/spatial/ppi_edges_protein_ids.csv"
```

The current terminal spatial script processes only the supplied `x1_data` and `x2_data` pair in one run; it does not automatically loop over multiple ROI combinations. For large spatial batches, the within-batch structure loss uses random subsampling when a batch exceeds the cap:

```text
--max_structure_samples 10000
```

The spatial script does not save `final_pmodel.pth` or `final_cmodel.pth`; it saves the embeddings, loss log, and integrated expression matrix listed below.

## Command-line parameters

Both model scripts share the following important defaults:

| Parameter | Default | Description |
|---|---:|---|
| `--output_dir` | `./outputs` | Output directory; created automatically |
| `--epochs` | `100` | Number of training epochs |
| `--inter_dim1` | `128` | Protein branch intermediate dimension |
| `--inter_dim2` | `512` | Cell branch interface dimension |
| `--embd_dim1` | `128` | Protein embedding dimension |
| `--embd_dim2` | `128` | Cell embedding dimension |
| `--proj_out_dim` | `128` | Fused protein projection dimension |
| `--seed` | `42` | Random seed |
| `--lr` | `1e-3` | AdamW learning rate |
| `--wd` | `1e-4` | AdamW weight decay |
| `--drop_p` | `0.2` | Dropout probability |
| `--kernel_size` | `3` | Cell convolution kernel size |
| `--rec_loss_w` | `1.0` | Reconstruction loss weight |
| `--kmeans_loss_w` | `1.0` | Batch-mixing loss weight |
| `--kmeans_intra_w` | `1.0` | Intra-batch term weight |
| `--kmeans_inter_w` | `1.0` | Inter-batch term weight |
| `--kmeans_margin` | `1.0` | Batch-centroid margin |
| `--corr_struct_w` | `0.01` | Within-batch structure loss weight |
| `--corr_eps` | `1e-8` | Numerical stability constant |
| `--ppi_loss_w` | `10.0` | PPI consistency loss weight |
| `--merge_mode` | `concat` | Protein representation merge mode |
| `--shared_decoder_input` | `shared` | Decoder input mode |
| `--ppi_sim` | `dot_sigmoid` | PPI similarity function |
| `--device` | auto | `cuda` when available, otherwise `cpu` |
| `--cuda_visible_devices` | `0` in the terminal scripts | CUDA-visible device setting; can be overridden, for example with `--cuda_visible_devices 1` |

Boolean options:

```text
--normalize_emb
--reverse_batch
```

These flags are disabled unless explicitly provided.

Spatial-only parameter:

| Parameter | Default | Description |
|---|---:|---|
| `--max_structure_samples` | `10000` | Maximum cells used per batch for spatial structure-loss distance computation |

To view every supported argument:

```bash
python "model/single_cell_run.py" --help
python "model/spatial_run.py" --help
```

To view data-preparation and PPI arguments:

```bash
python "data/prepare_integration_data.py" --help
python "data/retrieve_ppi_network.py" --help
```

## Output files

### Data preparation

```text
N2_data.csv
N2_cell.csv
nanoPOTS_data.csv
nanoPOTS_cell.csv
merged_data.csv
```

### PPI retrieval

```text
ppi_edges_protein_ids.csv
```

The edge table contains:

```text
protein1,protein2,score
```

### Single-cell model

```text
final_pmodel.pth
final_cmodel.pth
final_protein_emb.npy
final_cell_emb.npy
training_losses.xlsx
integrated_protein_expression.csv
```

### Spatial model

```text
final_protein_emb.npy
final_cell_emb.npy
training_losses.xlsx
integrated_protein_expression.csv
```

`integrated_protein_expression.csv` has the same orientation as `merged_data` used by the model:

- proteins are rows;
- cells or spatial locations are columns;
- the protein index and cell/location column names are retained.

`training_losses.xlsx` contains the epoch-wise training loss log. The NumPy files contain the final protein and cell embeddings. The single-cell script additionally saves the two PyTorch model state dictionaries.

## Example datasets

The repository includes prepared example directories for several paired datasets:

```text
data/N2+nanoPOTS/
data/Leduc_2024+Leduc_2025/
data/plexDIA_2023+pSCoPE_2022/
data/plexDIA_2023+pSCoPE_2023/
data/pSCoPE_2022+pSCoPE_2023/
data/ROI/
```

The N2+nanoPOTS example contains:

```text
data/N2+nanoPOTS/N2_data.csv
data/N2+nanoPOTS/N2_cell.csv
data/N2+nanoPOTS/nanoPOTS_data.csv
data/N2+nanoPOTS/nanoPOTS_cell.csv
data/N2+nanoPOTS/merged_data.csv
data/N2+nanoPOTS/ppi_edges_protein_ids.csv
```

## Data sources

The manuscript reports the following data sources:

- **N27:** MassIVE accession `MSV000086809`
- **nanoPOTS6:** MassIVE accession `MSV000084110`
- **Leduc_2024, Leduc_2025, plexDIA_2023, pSCoPE_2022, pSCoPE_2023:** [scp.slavovlab.net](https://scp.slavovlab.net/)
- **Human tonsil spatial proteomics dataset:** [Zenodo DOI 10.5281/zenodo.10982119](https://doi.org/10.5281/zenodo.10982119)

The manuscript describes N27, nanoPOTS6, Leduc, plexDIA, and pSCoPE as mouse or human datasets according to the source study. Choose the STRING taxonomy ID corresponding to the species of the input data:

```text
Mouse: 10090
Human: 9606
```

## Reproducibility notes

- The default random seed is `42`.
- Python, NumPy, PyTorch, and CUDA seeds are initialized when the model starts.
- Deterministic CuDNN settings are enabled where supported.
- The default number of epochs is `100`.
- The N2+nanoPOTS analysis described in the manuscript was extended to `150` epochs because the four reconstruction losses did not converge below `1.0` within the default number of epochs. To reproduce that manuscript-specific setting, add:

```bash
--epochs 150
```

- GPU visibility is configured before importing PyTorch. GPU `0` is the default setting in the terminal scripts; use `--cuda_visible_devices` to select another GPU or `--device cpu` to force CPU execution.
- PPI retrieval depends on the external STRING API and may be affected by network availability, API response changes, identifier mapping success, and species selection.

## Limitations

The manuscript highlights several limitations:

- The current mask does not distinguish non-detection, low abundance, and other data-processing causes of missingness.
- Source-specific proteins generally lack complete ground truth in the other dataset, so reconstruction is evaluated indirectly through masking experiments, marker patterns, covariation, and PPI context.
- Reconstruction depends on the cell structure available from the other source and on the coverage and accuracy of the PPI network.
- The current training setup is primarily designed for paired datasets.
- The terminal spatial extension processes one supplied pair per run and does not directly model spatial coordinates, cell neighborhoods, or tissue images.
- Covariation can be affected by cell number, cellular composition, and low-abundance measurements; it does not by itself establish direct molecular interactions.

## Citation

The repository is associated with the scPPInt manuscript. Please cite the official preprint or journal article after it becomes available. The working manuscript title is:

```text
scPPInt: PPI-guided Integration and Imputation of Single-cell Proteomes with Inconsistent Protein Coverage
```

No verified author list, DOI, preprint identifier, or journal citation was available in the supplied manuscript snapshot. A complete citation should therefore be added here after publication or preprint posting. The manuscript identifies the code repository as:

```text
https://github.com/Zhou-Shengming/scPPInt
```

## License and contact

No `LICENSE` file or explicit license statement was found in the current repository snapshot. Please add a license before redistributing the code or accepting external contributions.

No maintainer email or separate contact file was found in the current repository snapshot. For project updates, use the GitHub repository:

```text
https://github.com/Zhou-Shengming/scPPInt
```
