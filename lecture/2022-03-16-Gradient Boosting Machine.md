## Gradient Boosting Machine (GBM)

GBM은 CTR 예측을 통해 개인화된 추천 시스템을 만들 수 있는 또 다른 대표적인 모델이다. 대표적으로 하이퍼 커넥트 하쿠나 라이브의 추천시스템은 서비스 데이터가 축적됨에 따라 초기의 인기도 기반 추천 또는 휴리스틱 기반 추천 시스템에서 탈피했으며, 하이퍼파라미터에 비교적으로 민감하지 않은 (robust) GBM 모델을 사용했다

0. Boosting이란?
   
    앙상블 기법의 일종으로 의사결정 나무로 된 weak learner들을 연속적으로 학습해 결합하는 방식이다. 이전 단계의 weak learner가 취약했던 부분들을 위주로 데이터를 샘플링하거나 가중치를 부여해 다음 단계의 learner를 학습한다는 의미이다.

    대표적인 모델로는 AdaBoost, Gradient Boosting Machine, XGBoost 등 이 존재한다

1. GBM이란?

    GBM은 Gradient descent를 사용해 loss가 줄어드는 방향으로 weak learner들을 반복적으로 결합하여 성능을 향상시키는 알고리즘이다

    SGD와 다른점은 파라미터를 통해 loss function을 미분하는 것이 아닌 learner 그 자체로 미분을 진행하여 Gradient를 구한다

    통계학적 관점에서 Gradient Boosting은 실제값과 예측값 간의 차이인 residual을 fitting하는 것으로 이해할 수 있다 이를 기반으로 이전 단계의 weak learner까지의 residual을 계산하여, 이를 예측하는 다음 weak learner를 학습하여 기존 모델에 결합한다.

    residual 값을 구할 때 회귀문제에서는 예측값을 그대로 사용하는 반면 분류 문제에서는 0과 1사이 값을 표현하기 위해 log(odds) 값을 사용한다

     Gradient Boosting은 random forest보다 나은 성능을 보이나 학습 속도가 느리다는 단점과 과적합이 쉽다는 문제를 가지고 있다 이러한 문제를 해결하기 위하여 1. 병렬처리 및 근사 알고리즘을 통해 학습 속도를 개선한 XGBoost 2. Microsoft에서 제안한 병렬 처리 없이도 빠르게 Gradient Boosting을 학습할 수 있도록 하는 LightGBM, 3. 범주형 변수에 효과적인 알고리즘 등을 구현하여 학습 속도를 개선하고 과적합을 방지하고자 한 CatBoost 가 등장하였다
