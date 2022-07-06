---
layout: single
title: 'Variable, List, Basic Operation'
categories:
  - boostcamp AI tech
---

## Variable & List
- Variable & Memory
	- 변수란 데이터를 메모리에 저장하는 장소
	- 변수는 메모리 주소를 가지며 해당 메모리 주소에는 값이 저장된다
	- 폰 노이만 아키텍쳐
		- 입력된 정보를 메모리에 저장
		- CPU가 순차적으로 정보를 처리
	- 값 할당시 DRAM에 저장 

## Basic Operation
- 얼마나 메모리 공간을 차지하느냐에 따라 다른 data type을 쓸 필요가 있다
- Dynamic Typing (대부분의 언어 지원 x)
	- 실행 시 메모리 공간 확보와 같은 이유로 다소 느리다는 단점 존재
	- but 형 변환이 자유롭다
- 연산자와 피연산자
	- *, **, / , %, +=, -=, ...
- 데이터 형 변환 함수 int(), float() ...
	- 실수형에서 정수형으로 변환 시 소수점 이하 내림
- 반올림 오차
	- 실수를 이진수로 변환 시 무한 소수가 될 수 있으나 오차가 충분히 작아 일반적으로 문제가 되지 않는다
	- 왜 이진수를 쓰는가 ?
		- 트랜지스터를 통해 전류가 흐를때 1, 흐르지 않을 때 0으로만 숫자를 표현 가능하기 때문

## List
- 다양한 데이터 타입을 저장할 수 있는 구조
- list에 있는 값들은 주소(offset)을 가진다
- slicing
	- list의 주소값을 기반으로 잘라서 쓰는게 가능하다
- 리스트에도 다양한 연산 존재
	- +, in, append, remove ...
- 리스트 변수에는 리스트 주소값이 저장된다
	- 이러한 이유로 1차원 리스트 copy는a = b[:] 와 같이 이루어져야함
- 패킹과 언패킹 지원
- one-dimensional list는 copy가 가능하지만 two-dimension list는 지원 x 
	- then how ? copy 라이브러리를 사용