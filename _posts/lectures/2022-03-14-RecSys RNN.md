---
layout: single
title: 'Recommendation System with RNN'
categories:
  - boostcamp AI tech
---
## RNN Families

1. Recurrent Neural Network (RNN)

    시퀀스 데이터의 처리와 이해에 좋은 성능을 보이는 신경망 구조

2. Long-Short Term Memory (LSTM)

    시퀀스가 길어지는 경우 학습 능력이 저하되는 RNN의 한계를 극복한 모델로 장기 의존성 해결을 위해 Cell state라는 구조를 고안해냈다

3. Gated Recurrent Unit(GRU)

    LSTM의 변형 중 하나로 출력 게이트가 따로 없어 파라미터와 연산량을 더 줄인 모델

    reset gate : 바로 전 cell에서 들어온 정보를 얼마만큼 쓸 것인지 결정

    update gate : 얼마만큼 과거의 정보와 새로 들어온 정보를 생각해 업데이트 할 것인지 결정

## GRU4Rec

고객의 선호는 고정된 것이 아니라 시간에 따라 변하는 것이다 즉 '지금' 고객이 좋아하는 것을 추천해야한다

유저가 서비스를 이용하는 동안의 행동을 묶어 기록한 것을 **Session**이라고 한다

GRU4Rec은 '지금' 고객이 원하는 상품을 추천하는 것을 목표로 추천 시스템에 RNN을 적용한 논문

1. GRU4Rec 아이디어
   
    Session이라는 시퀀스를 GRU 레이어에 입력하여 바로 다음에 올 확률이 가장 높은 아이템을 추천한다

2. 모델 구조
   
    입력으로 one-hot encoding된 session이 들어가고

    GRU 레이어는 시퀀스상 모든 아이템들에 대한 맥락적 관게를 학습하며

    출력으로는 다음에 골라질 아이템에 대한 선호도 스코어가 나온다

3. 학습
    
    Session Parallel Mini Batches

    대부분의 세션은 매우 짧지만 긴 것도 존재하여 길이가 짧은 세션들이 단독 사용되면 학습이 비효율적으로 일어나게 된다

    이러한 이유로 하나의 세션만을 그대로 사용하는 것이 아니라 짧은 세션의 경우 다른 세션을 붙여 병렬적으로 구성해 미니 배치 학습이 가능케했다


    Sampling on the output

    현실에서는 아이템의 수가 너무 많아 모든 후보 아이템의 확률을 계산하기 어렵다는 문제가 존재하기 때문에 
    
    아이템을 negative sampling하여 subset만으로 loss를 구했다

    사용자가 상호작용 하지 않은 아이템은 사용자가 관심이 없는 아이템이라고 가정하였을 때 아이템의 존재 자체를 몰랐거나 관심이 없는지 구분하기 어렵다

    이러한 이유로 아이템의 인기가 높은데도 상호작요이 없었다면 사용자가 관심이 없는 아이템이라고 가정하였다

