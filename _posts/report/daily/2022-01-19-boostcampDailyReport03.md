---
layout: single
title: 'Daily Report - 03'
---
## 1. 새로 알게된 내용
- **softmax 함수가 e^x^를 사용하는 이유**
	- softmax 함수는 시그모이드 함수로부터 유도되었기 때문
	- https://gooopy.tistory.com/53
- **softmax 함수 구현**
	- output vector로 너무 큰 값이 들어올 경우 overflow가 일어날 수 있기 때문에 이를 방지하기 위해 분자 exponential 부분에 max값을 빼준다
- **추론시에는 max 값만 사용하면 되기 때문에 softmax가 굳이 필요없다**
- **$tanh(x)$ 함수**
	- 미분 시 $1−tanh^2(x)$ 가 된다
- **Backpropagation through time (BPTT)**
	- loss function의 state에 대한 gradient 구하는 과정
![jpg](/assets/images/2022-01-19/20220120_012955422_02.jpg)
	- loss function의 학습 가능한 parameter에 대한  gradient 구하는 과정 ($W_rec$에 대해서만 진행)
![jpg](/assets/images/2022-01-19/20220120_012955422_03.jpg)
		- $W_{rec}$ 이나 $W_x$ 에 대한 gradient를 구할 때 summation이 필요한 이유는 각 state가 이전 time step에 대해 고려해야해서 이에 의해 이전 time step에 대한 영향성을 더할 필요가 있기 때문
		- https://m.blog.naver.com/infoefficien/221210061511
## 2. 내일 해야할 공부
- BPTT Loss function의 W_x에 대한 gradient 구해보기
- 통계학 + 베이즈 통계학
- Maximum Likelihood
- Text Processing 1, 2

## 3. 회고
- 안 풀리던 문제를 계속 붙잡고 있다 풀어서인지 꽤 뿌듯한 하루였다
- 문제를 푸는 것도 중요하겠지만 계속 고민하는 연습이 더 중요한 듯 하다 문제를 풀었어도 어렵거나 헷갈리는게 있다면 계속 고민하자
- 이번주는 모든 강의를 노트 남기는데는 어려움이 있을 것 같다 중요하다고 생각되는 것만 남기도록 하자
