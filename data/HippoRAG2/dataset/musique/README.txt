---
configs:
- config_name: default
  data_files:
  - split: train
    path:
    - "musique_full_v1.0_train.jsonl"
  - split: validation
    path: 
    - "musique_full_v1.0_dev.jsonl"
- config_name: answerable
  data_files:
  - split: train
    path:
    - "musique_ans_v1.0_train.jsonl"
  - split: validation
    path: 
    - "musique_ans_v1.0_dev.jsonl"
---