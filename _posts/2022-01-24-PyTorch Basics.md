---
layout: single
title: 'Pytorch Operations'
categories:
  - boostcamp AI tech
---
# Pytorch Operations
- numpy operation와 비슷
- Tensor
	- 다차원 array를 표현하는 pytorch 클래스
	- Tensor 생성은 list나 ndarray를 사용 가능하다
	- numpy와 같이 c에서 사용할 수 있는 대부분의 data type을 가질 수 있다
	- numpy와의 차이점은 GPU를 사용할 수 있느냐 없느냐
	- 기본적으로 numpy의 사용법이 그대로 적용
		- **ones_like**는 행렬 계산에 자주 사용된다
	- tensor에는 gpu에 올려져있는지, 메모리에 올려있는지 확인시켜주는 device property가 존재
	- reshape 대신 view를 사용해야한다
		- reshape은 deep copy를 통해 새로운 tensor 생성
		- view는 shape만을 바꾸고 메모리를 보장해준다
	- squeeze(), unsqueeze(dim)

- Tensor operations
	- 덧셈, 뺄셈 지원
	- 행렬 곱셈 연산 (dot, mm, matmul)
		- dot 벡터 내적
		- mm vs matmul
			- matmul은 broadcasting 지원 --> 이로 인해 결과를 헷갈리게 할 수 있는 요소 존재

- Tensor operations for ML/DL formula
	- nn.functional
		- softmax, argmax 함수 ...
		- cartesian_prod 