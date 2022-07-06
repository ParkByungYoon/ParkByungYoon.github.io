---
layout: single
title: 'Factorization Machines'
categories:
  - boostcamp AI tech
---
## Factorization Machines

SVM과 Factorization Model의 장점을 결합한 FM을 처음 소개한 논문

1. FM 등장 배경
    
    딥러닝 등장 이전에는 커널 공간을 활용하여 비선형 데이터셋에 높은 성능을 보이는 SVM이 가장 많이 사용되는 모델이었다 하지만 CF 환경에서는 SVM보다 MF 계열의 모델이 더 높은 성능을 내왔고 MF는 특별한 환경 혹은 데이터에만 적용할 수 있었기 때문에 둘의 장점을 결합하고자 하였다

2. FM 공식
   
   두 피쳐의 상호작용을 K 차원의 Factorization Parameter로 표현

3. FM 활용
   
   유저의 영화에 대한 평점 데이터는 대표적인 High Sparsity data이다. 평점 데이터를 일반적인 입력 데이터로 바꾸면(One-Hot encoding) 입력 차원이 전체 유저와 아이템 수만큼 증가하기 때문이다

   유저 A의 ST에 대한 평점 예측은 V_A와 V_ST를 FM 모델을 통해 학습하여 상호작용을 반영한다 V_ST는 유저 B,C의 영화 ST에 대한 평점 데이터를 통해(**ST 영화**에 대한 특징을 ST영화를 본 B,C의 평점을 통해) 학습되며 V_A는 유저 B,C가 유저 A와 공유하는 영화 SW의 평점 데이터를 통해(**A 유저**에 대한 특징을 ST영화를 본 B,C의 동시에 본 데이터를 통해) 학습된다
   
   FM은 SVM에 비해 sparse한 데이터에 대해서 높은 예측 성능을 보이며 선형 복잡도를 가져 수십억개의 학습 데이터에 대해서도 빠르게 학습한다 
   
   MF과 비교하면 CF 뿐만 아니라 다양한 예측 모델에 사용가능하다는 장점과 일반적인 feature를 모델의 입력값(유저, 아이템 ID 외 다른 부가 정보들)으로 활용이 가능하다는 장점을 가진다

## Field-aware Factorization Machine (FFM)

FM의 변형된 모델인 FFM을 제안하여 더 높은 성능을 보인 논문

1. FFM의 등장배경

   FM은 예측 문제에 두루 적용 가능한 모델로 특히 sparse 데이터에 대해 강한 성능을 보이는데 FFM은 이러한 FM을 발전시킨 모델로서 PITF 모델에서 영감을 얻었다

   여기서 PITF란 MF를 발전시킨 모델로, 유저-아이템 2차원 데이터를 사용하는 MF와 달리 유저-아이템-태그 3차원 텐서를 사용하여 MF를 확장시켰다 하나의 임베딩만을 사용한다면 서로 다른 field간의 상호작용을 표현하기에는 어려움이 있기 때문에 (유저, 아이템), (아이템, 태그), (유저, 태그) 각각에 대해 서로 다른 latent factor를 정의하여 구했다

   이를 일반화하여 여러 필드에 대해 latent factor를 정의한 것이 바로 FFM 모델이다


2. FFM의 특징
   
   입력 변수를 field로 나누어 field별로 서로 다른 latent factor를 가지도록 factorize했다 (기존의 FM은 하나의 변수에 대해 k개로 factorize but FFM은 f개의 필드에 대해 각각 k개로 factorize)

   field는 모델 설계 시 함께 정의, 같은 의미를 갖는 변수들의 집합을 설정한다 (유저: 성별, 디바이스, 운영체재 / 아이템 : 광고, 카테고리 / 컨텍스트: 어플리케이션, 배너)

   CTR 예측에 사용되는 피쳐는 이보다 훨씬 다양해 피쳐 개수만큼 필드를 정의하여 사용한다
   
3. FM vs FFM
   
   FM은 필드가 존재하지 않고 하나의 변수에 대해 factorizaiton k차원 만큼 파라미터를 학습하는 반면 FFM은 필드의 개수 f 와 factorization 차원 k의 곱의 개수만큼 파라미터를 학습한다

   