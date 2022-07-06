---
layout: single
title: 'Precision & Recall'
categories:
  - boostcamp AI tech
---
## Precision
- TP / TP+FP
- 예측중 한개만 Positive이어도 맞아다면 1.0
- 예측중 Positive가 적을수록 점수는 오른다

## Recall
- TP / TP+FN
- 예측이 모두 Positive이라면 1.0
- 예측중 Positive가 많을수록 점수는 오른다

## F1-score
- 2 * Recall * Precision / (Recall + Precision)
- 둘을 조화롭게 보는 Metric

## AUROC
- 위 Precision, Recall은 모델의 아웃풋을 threshold(일반적으로 0.5)를 통해 Positive와 Negative로 나눈다 
- 그렇기 때문에 threshold의 영향을 안받는 metric을 만듦
- But Data가 Imbalance할수록 높게 나오는 경향이 있음
- True Positive Rate와 False Positive Rate가 이루는 곡선 형태 그레프의 밑 면적 수치
- 분포 Metric이며 0과 1 분포 사이 차이가 클 수록 더 높게 나타나는 Metric

![jpg](/assets/images/2022-04-20/AUROC.jpg)