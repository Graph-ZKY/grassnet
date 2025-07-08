# GrassNet: A PyTorch Implementation

> A PyTorch-based implementation of the GrassNet Algorithm for graph-based node classification tasks. This repository supports graph datasets and utilizes state-space models for graph representation learning.

---

## 🔧 Dependencies

Ensure the following packages are installed:

- Python ≥ 3.7
- PyTorch ≥ 1.10
- tqdm
- networkx
- [DGL](https://github.com/dmlc/dgl)
- [mamba-ssm](https://github.com/state-spaces/mamba)

```bash
pip install torch torchvision torchaudio
pip install dgl
pip install tqdm
pip install networkx
pip install mamba-ssm
```

## 📂 Project Structure
```bash
grassnet/
├── impl/             # Hyperparameter searching
├── model/            # Model definitions including GrassNet & Mamba variants
├── utils/            # Utility scripts (e.g., metrics, data loading)
├── main.py           # Main training/testing pipeline
├── README.md         # Project overview and usage
```

---

## 📊 Supported Datasets

- Cora
- Citeseer
- Pubmed

> Note: Additional datasets can be manually downloaded and used by modifying the data loading scripts. All standard datasets can be automatically downloaded via DGL.
---


## 🚀 Getting Started

### Node Classification

To run experiments on node classification datasets:

```bash
python main.py --dataset cora
python main.py --dataset citeseer
```

### Additional Options

| Argument         | Description                                 | Default       |
|------------------|---------------------------------------------|---------------|
| `--dataset`      | Dataset to use (e.g. `cora`, `pubmed`)       | `cora`        |
| `--gpu`          | GPU index to use (-1 for CPU)                | `0`           |
| `--epochs`       | Number of training epochs                    | `1000`         |
| `--lr`           | Learning rate                                | `1e-1`        |
| `--weight-decay` | Weight decay (L2 regularization)             | `5e-4`        |
| `--dropout`      | Dropout rate                                 | `0.5`         |

---

## 🧠 Model Architecture

GrassNet integrates the structured state space model (SSM, via Mamba) with graph Laplacian spectral filtering to enhance spatial reasoning. It includes:

- Eigen decomposition of graph Laplacian
- Learnable frequency filter via Mamba
- Integration with GCN/MLP heads

For architecture details, see `model/myMamba_model.py`.

---
