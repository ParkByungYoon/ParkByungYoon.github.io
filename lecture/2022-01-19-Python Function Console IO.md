---
layout: single
title: 'Python Function & Console I/O'
---

## 함수
- 어떤 일을 수행하는 코드 덩어리
- 반복적인 코드 단축
- 캡슐화로 인해 타인 코드 사용 가능
- 실행시
	- 함수는 메모리에
	- 함수 호출
	- 함수 수행
- python formatting 함수 간에는 두줄 간격
- parameter : 함수 입력값
- argument : 실제 Parameter에 대입된 값
- parameter, 반환 값 유무에 따라 함수 수행 능력이 달라진다

## Console I/O
- 마우스가 아닌 키보드로 명령
- python 입력 출력
	- 입력 : Input()
	- 출력 : print() 콤마(,) 사용시 연결됨
- format에 맞춰 출력
	- print 문을 활용한 결과 formatting
		1. %string 
		2. format 함수
		3. fstring 
			- PEP498에 근거한 formatting 기법
			- print(f'Hello {name}')
	- {>10s} : 10칸을 비우면서 왼쪽 정렬
- ctrl + shift + - : 코드 셀 나누기