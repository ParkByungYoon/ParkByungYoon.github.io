---
layout: single
title: '이미지 분류대회 실험일지 - 07'
categories:
  - Experiment Report
---
## 2022-03-02 실험 일지
- Data Augmentation
  - Over sampling + Under sampling
    - Imbalanced Dataset Sampler를 통한 batch 별 class balance
  - Transform
    - CutMix, Mixup 랜덤하게 적용
    - 중복되는 데이터가 많아졌을 수 있기 때문에 overfitting을 줄이기 위해 비율을 높임
- Loss function
  - label smoothing으로 고정
    - Class imbalance 문제 해결을 이루기도 했고 실험을 통해 확인 했을 때도 Data Noise 때문인지 label smoothing이 성능이 더 높게 나옴