---
layout: single
title: 'Pytorch Dataset과 Dataloader'
---

## Data Flow
- collecting / cleaning / pre-processing
- dataset class  
	- 데이터 정의
- transforms
	- ToTensor(), CenterCrop()
	- Tensor를 만드는 일과 image data를 전처리를 통해 넘겨주는 역할은 다르다!
- DataLoader
	- Model feeding (batch, shuffle ...) 역할

## Dataset Class
- image, text, audio 데이터에 따라 다른 입력 정의
- __init\__() : 초기 데이터 생성
- __len\__() : 데이터의 전체 길이
- __getitem\__() : 하나의 데이터를 부를 때 어떻게 불러올지
- 데이터에 따라 각 함수를 다르게 정의
- 모든 것을 데이터 생성 시점에 처리 x
	- image의 Tensor 변화의 경우 학습에 필요한 시점에 변환 (transform)
 - 데이터 셋에 대한 표준화된 처리방법 제공 필요

## Pytorch Dataset
- 파이토치는 멀티 스레딩을 통한 데이터 병렬화, 데이터 증식 및 배치 처리와 같은 여러 복잡한 작업을 추상화하는 여러 유틸리티 클래스를 제공
- Dataset 관련 모듈
	- [`torch.utils.data`](https://pytorch.org/docs/stable/data.html): 데이터셋의 표준을 정의하고 데이터셋을 불러오고 자르고 섞는데 쓰는 도구들이 들어있는 모듈
	- [`torchvision.dataset`](https://pytorch.org/docs/stable/data.html?highlight=dataset#torch.utils.data.Dataset): `torch.utils.data.Dataset`을 상속하는 이미지 데이터셋의 모음 (ex. MNIST, CIFAR-10)
	- [`torchtext.dataset`](https://pytorch.org/text/stable/datasets.html): `torch.utils.data.Dataset`을 상속하는 텍스트 데이터셋의 모음 (ex. IMDb, AG_NEWS)
	- [`torchvision.transforms`](https://pytorch.org/vision/stable/transforms.html): 이미지 데이터셋에 쓸 수 있는 여러 가지 변환 필터를 담고 있는 모듈
	- [`torchvision.utils`](https://pytorch.org/vision/stable/utils.html): 이미지 데이터를 저장하고 시각화하기 위한 도구가 들어있는 모듈

## DataLoader
- data의 batch를 생성해주는 클래스
- GPU feed 전 데이터 변환
- Tensor 변환 + Batch 처리
- sampler, batch_sampler
	- https://subinium.github.io/pytorch-dataloader/
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
- drop_last
	- batch_size에 따라 마지막 batch의 길이가 달라지면 loss를 구할 때 차질이 생길 수 있다
	- 그럴 때 drop_last 인자를 통해 마지막 batch를 사용하지 않는다


## Transform
- 수집한 모든 데이터의 크기가 동일하지 않은 경우 고정된 입력값이 보장하기 위해 torchvision에서 제공하는 함수
- **torchvision은 항상 PIL 객체로 받아야한다**
	- 그렇지만 항상 PIL로 불러올 필요는 없다 `ToPILImage` 메서드를 이용하면 바로 array style에서 불러올 수 있기 때문
- transforms.Compose
	- Compose 함수를 통해서 여러 transforms들을 하나로 묶어서 처리해줄 수 있습니다. 
- transforms.[Resize](https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.Resize)
	- 이미지의 사이즈를 변환
- transforms.[RandomCrop](https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.RandomCrop)
	- 이미지를 임의의 위치에서 자른다
- transforms.[RandomRotation](https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.RandomRotation)
	- 이미지를 임의의 각도만큼 회전