---
layout: single
title: '딥러닝 프레임워크 Pytorch 알아보기'
---
## PyTorch란
- 딥러닝 Framework
- TensorFlow vs Pytorch vs Keras
	- Keras & TensorFlow
		- wrapper (TF/Pytorch) 
		- High-level API
		- ***Static Graph***
	- Pytorch
		-  ***Dynamic computation graphs***
	- ***Static Graph*** vs ***Dynamic computation graphs***
		- ***Static Graph*** 는 Define & Run (그래프를 정의하는 코드 작성해두고 실행 시점에 데이터 feed)
		- ***Dynamic computation graphs*** 는 그래프를 BP를 쓸때, 즉 자동미분을 할 때 실행시점에서 정의
	- Tensorflow의 장점 
		- Production에 강점 (Multi-GPU, Cloud)
		- Scalability
	- Pytorch
		- Define by Run 장점
		- pythonic code
		- Numpy + AutoGrad + 다양한 딥러닝의 함수(Dataset , Multi-GPU, Data Augmentation) 지원
		- Numpy 구조를 가지는 Tensor 객체로 array 표현