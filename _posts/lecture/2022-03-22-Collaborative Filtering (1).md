---
layout: single
title: 'Collaborative Filtering - 1'
categories:
  - boostcamp AI tech
---
## Memory-based CF for Rating Prediction

사용자가 아이템에 부여할 평점은 다른 유사한 사용자가 부여한 또는 유사한 아이템에 부여된 평점을 기반으로 추정할 수 있을 것이고 이때 유사한 사용자 또는 아이템일수록 더 높은 중요도를 가진다

### Item-to-Item Similarity

r(u,i)를 예측하고자 할 때, 아이템 i와 가장 유사한 아이템들(j)에 사용자 u가 부여한 평점 R(u,j)를 주목한다

### User-to-User Similarity

r(u,i)를 예측하고자 할 때, 사용자 u와 가장 유사한 사용자들(v)에 아이템 i에 부여한 평점 R(v,i)를 주목한다

## Model-based Collaborative Filtering

사용자와 아이템의 저차원 표현을 학습하는 것으로 볼 수 있고 명시적인 feature를 사용하지 않고도 잠재적인 의미를 가진 representation들을 학습하기 때문에 latent factor model이라고도 한다

Sparse한 user와 item 간 상호작용 matrix를 분해하여 각 행이나 열이 user의 preference, item properties를 담은 matrix를 만든다 학습된 User Latent Facor, Item Latent Factor를 같은 공간에 도식화 할 경우 별도의 정보(meta-data)를 제공하지 않아도 상호작용 패턴만으로 잠재정보를 발견할 수 있다

### Implicit Feedback for Recommendation

Implicit feedback은 Explicit feedback과 달리 사용자의 선호에 대한 암시적인 정보만을 제공하지만 수집 효율성, 일반화 성능 덕에 사용자 실제 선호를 추정하는데 더욱 효율적일 수 있다 

Observed 평점 분포와 Unseen 평점 분포의 불일치가 존재할 수 있으며 추천 모델은 Observed 평점으로부터 Unseen 평점 분포를 추정하기 때문에 일반화 성능이 떨어질 수 있어 Implicit feedback이 더 좋게 작용할 수도 있다

하지만 Explicit feedback 또한 명확하고 세분화된 사용자 선호를 반영해 정보 제공 측면에서 유리하다는 장점을 가지고 있다

## Properties of Collaborative Filtering

1) Label과 feture의 구분이 모호하다 feature과 label이 컬럼별로 고정된 clf, reg와는 달리 모든 column이 각 example에서 feature이자 label로 기능할 수 있다
2) Traning/testing 구분이 row-wise가 아닌 entry-wise로 이루어져있다

CF에서 cold-start problem이 존재하는 이유는 새로운 행이나 열이 추가되었을 때 추가된 행이나 열은 기존의 학습된 모델에서 사용하던 feature에 대응하지 않기 때문이다 이러한 이유로 Content-based 방법이 cold-start problem에 robust하기도 하다


## User-free Model-based Approaches

User-free 모델의 장점은 다음과 같다 (user-preference matrix, \gamma_u를 사용하지 않을 때 장점)
1) \gamma_u는 새로운 사용자가 발생할 때마다 재학습을 필요로 하지만 없을 경우 새로운 사용자에 대해 inference가 가능하다
2) 이력이 거의 없는 사용자에 대한 대응이 가능하다
3) CF 모델에서 종종 무시되는 sequential 시나리오에 대해 대응이 가능하다

그럼 어떻게 \gamma_u를 제거하고 personalized recommendation이 가능할까? Item의 vector(interaction matrix R에서 사용자 u에 대응하는 하나의 row)를 입력으로 받아 추천 결과를 생성하는 형태의 모델을 사용한다 형태가 Memory-based CF와 유사하지만 heuristics(cosine, jaccard ...)를 사용하는 대신 parameter W를 학습한다

SLIM, FSIM, AutoRec, Item2vec, Sequential Recommendation Models ...