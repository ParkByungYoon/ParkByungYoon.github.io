---
layout: single
title: '이미지 분류대회 실험일지 - 02'
categories:
  - boostcamp AI tech
---
# 2022-02-24 실험일지
## Model Output 확인
- train dataset에 대한 결과값을 통해 진행
  - Validation dataset만을 비교할 필요가 있어보임
  - Age가 가장 error rate가 높았음
    - but 60세 이상 데이터가 가장 적어 문제가 될 줄 알았으나 30~60세 데이터를 더 못맞추는것으로 확인됨
  - Mask Incorrect 데이터도 잘 못 맞추는 경향이 있음
## Ensemble
- k-fold 코드 완성
  - 실험에 많은 시간이 소요됨
  - 현재로서 많은 실험을 돌려보는게 좋을 듯해 epoch 수를 10으로 고정하는게 좋을 듯함