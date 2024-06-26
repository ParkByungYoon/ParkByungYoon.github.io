---
layout: single
title: 'Gradient Descent - 2'
categories:
  - boostcamp AI tech
---

## 경사하강법을 통한 선형회귀
- 경사하강법으로 선형회귀 계수 구하기
	- Moore-Penrose 역행렬을 이용해 구할 수도 있으나 경사하강법을 통해서도 구할 수 있다.
	- 주어진 데이터의 정답에 해당하는 y
	- 선형 모델 X * beta
	- 두 벡터 차이 L2 norm을 최소화하는 beta를 찾는 것이 목적
	- 찾기 위해선 위 목적식을 beta로 미분하고 주어진 벡터에 빼기를 반복해 목적식의 최소화한다
	- 제곱근을 사용하는것이 아닌 L2 norm의 제곱을 사용해도 목적식을 최소화 하는 방향은 같다
- 경사하강법 기반 선형회귀 알고리즘
	- error term : $Y - X \beta$
	- gradient vector : - $X^t$ * $error$
	- $\beta$ : $\beta$- lr * gradient vector
	- 일반적으로 종료조건을 학습 횟수(t)로 설정
	- 학습률(lr)과 학습 횟수(t)는 중요한 hyperparameter로 작용한다

- 경사하강법은 만능일까?
	- 수학적으로는 convex 함수에 대해 적절한 학습률(lr)과 학습횟수(t)가 주어진다면 수렴된다는 것은 증명이 되어있다
	- 특히 선형회귀의 경우 위 목적식은 회귀계수 beta에 대해 볼록함수이기 때문에 수렴이 보장된다
	- but non-linear 문제의 경우 목적식이 볼록하지 않을 수 있다 (수렴이 항상 보장되지는 않는다)

## 확률적 경사하강법
- 앞서 설명한 경사하강법은 non-convex 목적식에 적용하기 힘들기 때문에 확률적 경사하강법(stochastic gradient descent, SGD)이 필요하다
- SGD는 모든 데이터를 사용하는 것이 아닌 한개 또는 일부(mini-batch SGD)만을 활용해 업데이트하는 방식
- SGD 특징
	- 볼록이 아닌 목적식에서 최적화가 가능하다
	- 모든 데이터를 쓰는 것이 아니기 때문에 모든 데이터를 사용한 gradient descent 와는 다를 수 있으나 기댓값은 유사하다는 것은 확률적으로 보장된다
	- SGD 또한 만능은 아니지만 딥러닝의 경우 SGD가 실증적으로 낫다고 검증이 된 상태
	- 데이터 중 일부만을 사용하기 때문에 연산 자원을 효율적으로 사용이 가능 (O(d^2^n) > O(d^2^b))
	- 미니배치 또한 hyperparameter로 작용한다 (미니배치에 의해 학습속도가 결정되기도 한다)
	- 일반적인 경사하강법처럼 모든 데이터를 업로드 하면 메모리가 부족하다
	- 그러나 SGD는 병렬적으로 일처리가 가능하다
- SGD 원리 
	- $D = (X,y)$를 가지고 목적식의 그레디언트 벡터 $\nabla_\theta L(D,\theta)$를 구하는 것이 아닌 미니배치 $D_{(b)} = (X_{(b)}, y_{(b)})$를 가진다
	- 목적식 자체는 다를 수 있으나 방향성은 비슷할 것이라 기대할 수 있다
	- step별로 다른 미니 배치를 사용해 경사하강법을 적용 (매번 목적식의 모양은 바뀌게 된다)
		- 이로 인해 local point(극소점/극대점)에 도달해도 확률적으로 모양이 바뀌어 더 이상 극소점이 아니게 될 확률이 발생하게 된다
		- 즉 gradient vector가 0에 도달하더라도 모양이 바뀐 뒤에는 gradient vector가 0이 아닐 수 있기 때문에 local point에서 탈출이 가능하다