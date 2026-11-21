# Conflict-guided Heterogeneous Knowledge Graph for Multimodal ERC

This repository contains the source code of our paper, using the following datasets:

- [IEMOCAP](https://drive.google.com/file/d/1y4xLSKicRHZCd7dsG5y75TFecvHHyY3Q/view?usp=drive_link) — Download the preprocessed dataset and put it into `data/`.
- [MELD](https://drive.google.com/file/d/1H8NAwW_ogxsFOQvCuesxiVqBqa8X19J-/view?usp=drive_link) — Download the preprocessed dataset and put it into `data/`.

## 1. Install requirements

```bash
pip install -r requirements.txt
```

## 2. Prepare datasets

Download the preprocessed datasets and put them into `data/`.

## Train

We provide training commands aligned with the paper settings.

```bash
# IEMOCAP

python -u train.py --Dataset IEMOCAP --use-knowledge --use-graph 

# MELD
python -u train.py --Dataset MELD --use-knowledge --use-graph 
```
