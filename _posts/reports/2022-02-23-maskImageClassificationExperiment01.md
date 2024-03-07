---
layout: single
title: '이미지 분류대회 실험일지 - 01'
categories:
  - boostcamp AI tech
---
# 2022-02-23 실험일지
## CoAtNet
  - CUDA out of Memory
  - Batch size를 줄여야함
## ViT
  - timm 에서 pretrained model load
  - model input에 맞추기 위해 Resize 필요
    - but Resize로 인한 정보 손실 존재 
      - CenterCrop (224, 224)
    - 모델을 고칠 방법을 찾아봐야할듯
## Class imbalance 문제  
  - Label Smoothing Loss function 사용
  - Data Augmentation 사용해보기
  - 어느 클래스가 젤 많이 틀리는지 확인해보자