---
layout: single
title: '이미지 분류대회 실험일지 - 05'
categories:
  - boostcamp AI tech
---
## 2022-02-27 실험 일지
- Efficient Network
  - Data imbalance 문제 때문인지 Overfitting이 심하다
  - Data augmentation 작업 후 다시 학습이 필요할 듯하다
- Vision Transformer
  - 다양하게 기술들을 접목 중이다
  - Multi Sample Dropout
    - 큰 성능 향상은 이끌지 못했다
  - CutMix/Mixup
    - 실험 진행 중
    - RMSE Monitoring이 가능하도록 만듦
    - 각 epoch마다 다른 데이터가 생성되어 Data Augmentation 효과를 크게 일으킬 수 있을 것으로 예상됨
  -  MTCNN을 통한 Face Crop
     -  큰 성능 향상은 이끌지 못함
- 주가 되는 문제는 Data Imbalance 때문인 듯
  - 이러한 이유로 CutMix/Mixup 실험 결과가 기대됨