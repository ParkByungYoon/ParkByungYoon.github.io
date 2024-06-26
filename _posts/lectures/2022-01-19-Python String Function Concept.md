---
layout: single
title: 'Python String/Function Concept & Coding convention'
categories:
  - boostcamp AI tech
---

## String
- 시퀀스 자료형 data
- 한 글자 당 1 byte
	- 1byte = 8bit = 2^8^ = 256 까지 표현가능
	- 컴퓨터는 문자를 직접적으로 인식 못하기 때문에 2진수를 문자로 변환하는 표준 규칙 존재
	- int : 4 byte , float ; 8 byte, long: 무제한
- 많은 문자열 함수 존재
- 문장 내 " or ' 사용시 \ 추가

## Function Concept
- 함수 호출시 메모리 처리가 어떻게 되는가, 함수에서 parameter를 전달하는 방식
	1. Call by Value
		- 인자를 넘길 때 값만 넘김
		- 함수 내에서 인자 값 변경해도 호출자 영향 x
	2. Call by Reference
		- 메모리 주소를 넘김
		- 인자값 변경 시 호출자의 값 또한 변경
	3. Call by Object Reference
		- 객체 주소가 함수로 전달 
		- 전달된 객체를 참조하여 변경 시 호출자에게 영향을 주지만 새로운 객체가 만들어질 경우 영향을 주지 않는다
- 지역변수(함수 내) / 전역변수(프로그램 전체)
	- 전역변수는 함수에서 사용가능하나 함수내 전역변수와 같은 이름의 변수 선언 시 새로운 지역 변수가 생긴다 (함수 내에서 전역변수 사용 시 global 키워드 사용)
- 재귀함수
- dynamic typing : 사용자가 interface를 알기 어렵다
	- type hints 기능 제공
- docstring : 파이썬 함수에 대한 상세스펙 제공
- 함수 작성 가이드 라인
	- 함수는 가능한 짧게
	- 함수 이름에 함수의 역할, 의도가 분명하게!
	- 하나의 함수에는 유사한 역할을 하는 코드만
	- 인자로 받은 값 자체로 바꾸지 않을 것 (임시 변수 선언)
	- 공통적으로 사용되는 코드
	- 복잡한 수식
	- 복잡한 조건
	
**좋은 프로그래머는 사람이 이해할 수 있는 코드를 짠다**

- 파이썬 코딩 컨벤션
	- 들여쓰기는 Tab or 4 space
	- 한 줄은 최대 79자 까지
	- = 연산자는 1칸 이상 띄우지 않는다
	- PEP8 : 파이썬 코딩 컨벤션 기준
		- flake8 모듈로 체크가 가능하다