---
title: Daily Report - 06
layout: single
---

## 1. 새로 알게된 내용
- **view vs reshape**
	- tensor의 shape를 바꾸고 싶을 때 사용하는 함수들
	- view의 경우 별도의 메모리 공간을 소모할 필요 없으나 할당된 메모리 공간이 contiguous한 경우에만 사용 가능하다 (**contiguous 하지 않은 경우 Runtime Error**)
	- 반면 reshape은 copy를 통해 새로운 메모리 공간에 할당해 contiguous 하지 않아도 문제가 되지 않는다 하지만 메모리 소모가 커질 수 있다는 단점이 존재한다
	- https://inmoonlight.github.io/2021/03/03/PyTorch-view-transpose-reshape/

- **임의의 크기의 3D tensor에서 대각선 요소 모으기**
	- [gather](https://pytorch.org/docs/stable/generated/torch.gather.html)를 사용해 대각선 요소 추출 
	- gather에 대한 설명은 https://velog.io/@nawnoes/torch.gather%EB%9E%80) 참고
	- 대각선 요소는 위치는 (0,0) (1,1) (2,2) ... 와 같다
	- 위와 같이 row, col index 각 1씩 증가하는 규칙을 gather에 반영하기 위하여 
		1. **gather dimesion의 경우 1을 선택**하였고 (tensor[i][j][k], j=index[i][j][k])
		2. H와 W 중 작은 값까지 **index들을 차례로 증가**시키고 ([0,1,2, ...]) 
		3. 해당 인덱스들을 **각 채널마다** 넣어 구했다
![jpg](/assets/images/2022-01-24/20220124_235647882.jpg)

## 2. 내일 해야할 공부
- Custom model 과제 끝내기
- Pytorch 4~5강
- Custom dataset 과제

## 3. 회고
- 해야할 과제 양이 꽤 많다 시간을 허투로 쓰지 않는게 가장 중요할 듯 하다
- 과제는 많은데 시간이 많다고 느껴서 그런가 많이 해이해졌다 금요일을 없는 날이라 생각하고 열심히 해야겠다
- 집에 
