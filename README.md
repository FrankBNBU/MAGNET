

## Installation

### Conda
```bash
conda env create -f environment.yml
conda activate spatial-gae
```

### pip
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Data Preprocessing
```bash
python main.py --mode preprocess \
    --inputdirpath /path/to/data.csv \
    --outputdirpath /path/to/output \
    --studyname study_name
```

Input format: CSV with columns `Cell_ID`, `X`, `Y`, `Cell_Type`, and gene expression columns.

### Model Training
```bash
python main.py --mode train \
    --inputdirpath /path/to/data.csv \
    --outputdirpath /path/to/output \
    --studyname study_name \
    --split 0.7
```

## Architecture

Multi-view graph autoencoder with:
- Cell-level encoder for spatial relationships
- Gene-level encoder for regulatory networks
- Cross-attention mechanism for multi-modal fusion
- Contrastive learning for feature discrimination
- AlignmentLoss for interpretability

## Output Files

- `*_trained_gae_model_*.pth` - Model weights
- `*_metrics_*.csv` - Training metrics
- `reconstructed_*_adjacency.npy` - Reconstructed adjacency matrices
