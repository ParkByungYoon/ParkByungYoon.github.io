---
layout: single
title: '이미지 분류대회 실험일지 - 03'
categories:
  - Experiment Report
---
# 2022-02-25 실험일지
## Model Selection
- CoatNet (Covolution + Transformer)
  - 최근에 나온 모델이라 pretrained model을 찾을 수 없었다
  - 학습 시키기엔 너무 큰 모델
- ViT (Transformer)
  - Transformer를 사용한 모델
- ImageNet (Covolution)
  - Convolution Network 중 현재 가장 좋은 성능을 보이는 모델
- FaceNet
  - 얼굴을 학습한 모델
  
## Experiements
- k-fold는 시간이 너무 많이 걸리는 관계로 접어두는 게 좋을듯
- Efficient Network 성능 확인
- Multi Sample Dropout 성능 확인
- 