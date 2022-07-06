---
layout: single
title: '2022-02-26 이미지 분류대회 실험일지'
---
## 2022-02-26 실험 일지
- CUDA GPU Utilization
  - Dataloader
    - 서버에서는 num_workers = 4가 가장 효율적
    - batch_size는 실험마다 달라져야함 GPU Utilization을 가장 극대화할 수 있는 방향으로 모델과 Data의 크기에 따라 유동적으로 정해야함
  - Optimizer
    - accumulation_step (batch iterator가 step만큼 반복될 동안 gradient update를 시켜주지 않음)을 통해 큰 batch size를 사용하는 효과
    - Optimizer에 따라 gpu utilization이 바뀐다? (SGD vs Adam)
    - https://discuss.pytorch.org/t/cuda-out-of-memory-when-optimizer-step/55942/2

- Efficient Network
  - baseline model과 scaled model로 나뉜다
  - baseline model
    - 가볍다는 장점 
    - 성능은 확인해봐야할 듯
  - scaled model
    - 무겁다 (batch_size는 32로 줄여돌려야함) 
    - 성능도 확인해야함
  - Convolution 기반이라 image size는 고려 x
    - but 큰 이미지 사이즈는 돌리기 어려움
   
- Vision transformer
  - 224\*224 나 384\*384 size만을 사용할 수 있다
  