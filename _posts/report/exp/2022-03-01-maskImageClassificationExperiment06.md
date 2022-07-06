---
layout: single
title: '2022-03-01 이미지 분류대회 실험일지'
categories:
	- Experiment Report
---
## 2022-03-01 실험 일지
- Data Augmentation
  - Over sampling
    - 60세 이상 두배로 늘임 (flip,blur,rgbshift,brightness)
  - Transform
    - 매 배치 random하게 transform (flip, brightness)
    - CutMix/Mixup
      - 마지막 5~10 epoch CutMix/Mixup 적용 x
      - Mixup이 우리 데이터에 맞는 듯 (Global하게 적용)
- Loss function
  - focal loss: Hard Negative Example 학습에 초점
    - Class imbalance 문제 해결을 위해 
  - Label smoothing : Mislabeled Example를 고려한 학습
    - label 을 0또는 1이 아니라 smooth 하게 부여하는 것
- optimizer SGD 고정