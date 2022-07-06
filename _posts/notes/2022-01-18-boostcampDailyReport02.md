---
layout: single
title: 'boostcamp Daily Report - 02'
categories:
	- boostcamp Daily Report
---

## 1. 새로 알게된 내용
- **역행렬을 이용한 선형 회귀 분석**
	- 데이터 행렬 X에 $\beta$벡터를 곱해주면 선형모델식을 구할 수 있다. 이때 어떤 $\beta$를 쓰느냐에 따라 데이터들을 잘 표현할 수 있다.
	- 그럼 $\beta$를 어떻게 찾을까?
	- 선형회귀 분석은 연립방정식과 달리 행이 더 많기 때문에 방정식을 푸는것은 불가능하다 (y에 대한 데이터가 없는 새로운 데이터 X')
	- Moore-Penrose 역행렬을 이용하면 L2-norm을 최소화하는 beta를 찾을 수 있다.
	- https://ko.wikipedia.org/wiki/%EC%84%A0%ED%98%95_%ED%9A%8C%EA%B7%80#%EC%86%90%EC%8B%A4_%ED%95%A8%EC%88%98
- **경사하강법으로 선형회귀 계수 구하기 (Objective function 미분과정 수식 전개)**

![jpg](/assets/images/2022-01-18/20220118_234040196.jpg)
![jpg](/assets/images/2022-01-18/20220118_234214559.jpg)

  - 각 element가 곱해지는 모습을 보면 matrix 연산과 같아 위와 같이 수식 전개가 가능함을 할 수 있다 (Vectorization)
  - http://taewan.kim/post/cost_function_derivation/


## 2. 내일 해야할 공부
- AI math 딥러닝 학습방법 + 확률론
- Python Basics 강의 + Text Processing 1 과제

## 3. 회고
- 수식 이해 및 증명 과정이 다소 오래 걸렸다 그래도 끈질기게 하자 언젠간 된다는 걸 느꼈다
- 공부하는 시간이 짧지 않다보니 체력 분배가 쉽지 않다 오늘은 초반에 체력을 너무 많이 쓴 탓에 저녁에는 힘을 못 쓴듯 하다
- 주간 계획을 좀 더 영리하게 세울 필요가 있는 듯 하다 너무 많지도 않고 적지도 않은 나에게 딱 맞는 분량을 먼저 알아내는게 우선임을 느꼈다
