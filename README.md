# CGR-HGR
# Conflict-guided Heterogeneous Knowledge Graph for Multimodal ERC

This repository contains the source code of our paper.

## Datasets

- [IEMOCAP](https://drive.google.com/file/d/1y4xLSKicRHZCd7dsG5y75TFecvHHyY3Q/view?usp=drive_link)  
  Download the preprocessed dataset and put it into `data/`.

- [MELD](https://drive.google.com/file/d/1H8NAwW_ogxsFOQvCuesxiVqBqa8X19J-/view?usp=drive_link)  
  Download the preprocessed dataset and put it into `data/`.

## Installation

```bash
pip install -r requirements.txt
```

## Data Preparation

Download the preprocessed datasets and place them in the `data/` directory.

## Training

We provide training commands aligned with the paper settings.

```bash
# IEMOCAP
bash exec_iemocap.sh

# Or run directly
python -u train.py --Dataset IEMOCAP --use-knowledge --use-graph \
  --iemocap-path data/iemocap_multimodal_features.pkl \
  --knowledge-path data/iemocap_features_comet.pkl

# MELD
bash exec_meld.sh
```
