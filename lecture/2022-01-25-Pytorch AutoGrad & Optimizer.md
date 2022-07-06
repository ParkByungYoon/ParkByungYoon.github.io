---
layout: single
title: 'Pytorch 구성 단위'
---
## torch.nn.Module
- 딥러닝을 구성하는 Layer의 base class
- Input, Output, Forward, Backward 정의
- 학습 대상이 되는 parameter(tensor) 정의
## nn.Parameter
- Tensor 객체의 상속 객체
- nn.Module 내에 attribute가 될 때는 required_grad=True로 지정되어 학습 대상이 되는 Tensor
- 우리가 직접 지정할 일은 거의 없음  
	- 대부분의 layer 객체에는 weight 값들이 지정되어 있기 때문
- Tensor만을 사용해도 똑같은 결과값 but model.parameters()를 통해 parameter를 확인하기는 불가능

## Backward
- optimizer.zero_grad()
	- epoch 단위로 gradient buffer들을 clear
- loss.backward()
	- output과 label 간 loss 미분
- optimizer.step()
	- optimizer에 의한 weight 값들의 업데이트