---
layout: single
title: 'Context-aware Recommendation'
categories:
  - boostcamp AI tech
---
## Context-aware Recommendation

CF의 경우 user나 item의 meta-data를 사용하기에 어려움이 있으며 MF 기법을 활용한 CF는 상호작용 정보가 부족할 경우(cold start)에 대한 대처가 어렵다

 R: User X item X Context Feature --> Rating

 위와 같이 X를 통해 Y값을 추론하는 일반적인 예측 문제에 두루 사용 가능한 General Predictor 모델 구조이다

1. Click-Through Rate Prediction (CTR) 예측

    유저가 주어진 아이템을 클릭할 확률을 예측하는 문제로 Context-aware Recommendation이 사용되는 대표적인 활용 예시이다

    CTR 예측은 광고에서 주로 사용된다 광고가 노출된 상황의 다양한 유저, 광고, 컨택스트 피쳐를 모델의 입력 변수로 사용하며 유저 ID가 존재하지 않는 데이터도 다른 유저 피쳐나 컨텍스트 피쳐를 사용해 예측이 가능하다

2. Logistic Regression (0~1 사이 값 예측하는 선형모델)

    로지스틱 회귀 모델은 변수간의 상호작용을 전혀 모델링할 수 없기 때문에 유저 정보와 아이템 간의 상호작용이 모델에 반영되지 않는다 

    이러한 이유로 강제로 두개의 변수 간 상호작용, Catersian Product을 만들어 학습하게 한다(Polynomial Model)

    하지만 이 모델 또한 파라미터 수가 급격히 증가한다는 한계가 존재한다

3. 사용 데이터

    dense feature: 벡터로 표현했을 때 비교적 작은 공간에 밀집되어 분포하는 변수 (ex. 유저-아이템 평점, 기온, 시간 등)

    sparse feature: 벡터로 표현했을 때 비교적 넓은 공간에 분포하는 변수 (ex. 유저-아이템 ID, 요일, 분류, 키워드, 태그 등)

    CTR 예측 문제에 사용되는 대부분의 데이터는 sparse feature이다

    dense feature 모두를 One-hot Encoding으로 처리하기엔 파라미터 수가 너무 많아질 수 있으며 학습 데이터에 등장하는 빈도에 따라 특정 카테고리가 overfitting/underfitting을 낳을 수 있다

    이러한 이유로 feature embedding(Item2Vec, Latent Dirichlet Allocation, BERT)을 한 이후에 예측
