---
layout: single
title: Daily Report - 08
---

## 1. 새로 알게된 내용
- DataLoader
	- sampler / batch_sampler
		- index를 컨트롤하는 방법
		- 데이터의 index를 원하는 방식대로 조정 
		- shuffle은 False이어야한다
		- Sampler 종류
			- SequentialSampler
			- RandomSampler
			- SubsetRandomSampler
			- WeigthRandomSampler
			- BatchSampler 
			- DistributedSampler

	- num_workers
		- 데이터를 불러올때 사용하는 서브 프로세스(subprocess) 개수
		- 데이터를 불러 CPU와 GPU 사이에서 많은 교류가 일어나면 오히려 병목이 생길 수 있다
		- https://jybaek.tistory.com/799
	- collate_fn
		- 보통 map-style 데이터셋에서 sample list를 batch 단위로 바꾸기 위해 필요한 기능
		- zero-padding이나 Variable Size 데이터 등 **데이터 사이즈를 맞추기 위해** 많이 사용 

- Transfer Learning
	- 적은 데이터로 좋은 모델을 만드는 방법론이 다양하게 개발되었는데, 그 중 하나가 Transfer Learning이다
	- "Source Tasks"에서 학습된 지식을 "Target Task"로 전이하는 절차 및 방법론

## 2. 내일 해야할 공부
- Pytorch 6~7강
- 과제 복습

## 3. 회고
- 왜? Why? 를 많이 생각하자
- 과제 복습을 꼼꼼히 하자 (복습은 필수다)
- youtube 줄이기 (정해진 시간에만 보자)
