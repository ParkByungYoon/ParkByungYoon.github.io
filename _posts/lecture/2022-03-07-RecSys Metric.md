---
layout: single
title: 'Recommendation System Metric'
---
# 추천 시스템 평가 지표
## 비즈니스 / 서비스 관점
  - 매출 증가
  - CTR(노출 대비 클릭률) 상승
## 품질 관점
- Offline Test
  - 추천 모델을 검증하는 단계
  - 실제 서비스 상황에서는 다른 양상 (로그 데이터를 기반으로 학습하는 serving bias 존재)
- Precision / Recall @K
  - Precision@K
    - 우리가 추천한 K개 아이템 가운데 실제 유저가 관심있는 아이템의 비율
  - Recall@K
    - 유저가 관심있는 전체 아이템 가운데 우리가 추천한 아이템 비율 
- AP@K
    - Precision@1 부터 Precision@K 까지의 평균값 
- MAP@K
    - 모든 유저에 대한 AP 평균
- Normalized Discounted Cumulative Gain(NDCG)
  - Top K 리스트를 만들고 유저가 선호하는 아이템을 비교
  - 추천 순위가 높을 수록 가중치를 더 많이 두어 성능 평가
    - 1에 가까울수록 좋다
  - MAP와는 달리 연관성을 binary가 아닌 수치로도 사용이 가능하다
  - Cumulative Gain
    - 상위 K개의 아이템에 대해 관련도를 합친것 (선호도)
  - Discounted Cumulative Gain
    - 순위에 따라 Cumulative Gain을 Discount한다
  - Ideal DCG
    - 가능한 DCG 값 중 제일 큰 값으로 이상적인 추천이 일어났을 때 GCD값을 말한다
  - Normalized DCG
    - 추천 결과에 따라 구해진 DCG를 IDCG로 나눈값
- Online Test
  - Online A/B Test
    - 동시에 대조군과 실험군의 성능을 평가
    - 대부분 현업에서는 매출과 CTR이 성능의 지표

# 추천 시스템 인기도 기반 추천
- 인기도 기반 추천이란
  - 말 그대로 가장 인기있는 아이템을 추천
- 조회수 Most Popular 스코어링 
  - Hacker News Formula
    - 시간에 따라 줄어드는 score를 조정하기 위해 gravity 상수 사용 
  - Reddit Formula
    - 나중에 게시된 포스팅일수록 높은 점수
    - log 함수를 통해 초반 vote에 더 높은 가치 부여, vote가 늘어날수록 증가 폭이 줄어든다
- 평점 Highly Rated 스코어링
  - 신뢰 할 수 있는 평점인가?
  - Steam Rating Formula 
    - Review 개수가 너무 적을 경우 보정 (0.5보다 score가 낮을/높을 경우 조금 높게/낮게 보정)
    - Review의 개수가 많아진 경우에는 평균 rating과 비슷해진다