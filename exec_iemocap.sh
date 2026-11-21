python -u train.py --lr 0.0001 --batch-size 16 --epochs 150 --temp 1 --Dataset IEMOCAP \
  --use-knowledge --use-graph \
  --iemocap-path data/iemocap_multimodal_features.pkl \
  --knowledge-path data/iemocap_features_comet.pkl
