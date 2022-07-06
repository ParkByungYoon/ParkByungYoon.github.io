---
layout: single
title: 'File System & Terminal'
categories:
  - boostcamp AI tech
---

## File System & Terminal Basic

- 컴퓨터 OS
	- 운영체제
	- 우리의 프로그램이 동작할 수 있게하는 환경
	- 소프트웨어와 하드웨어가 연결하는 기반이 되는 시스템
	- app은 운영체제에 dependent
		- ex) exe file(only window)
	- python은 운영체제로부터 독립적

- 파일 시스템
	- 파일을 저장할 수 있는 트리구조의 저장 체계 (Mac : root 디렉토리 시작, Window : C drive를 기준으로 시작)
	- 디렉토리 : 파일(읽기, 쓰기, 실행)과 다른 디렉토리를 담을 수 있는 그릇
	- 절대경로 vs 상대경로
		- 절대경로 : 루트 디렉토리부터 파일 위치
		- 상대경로 : 현재 디렉토리부터의 위치 (. : 현재 위치 , .. : parent)
- 터미널
	- 명령을 Text, 키보드로 입력
	- Command Line Interface (CLI)
	- Windows: CMD, cmder 권장
	- Mac, Linux : Terminal
	- Console = Terminal
	- 요즘은 Windows 환경에서 Ubuntu 다운 받는다면 Linux 사용 가능
	- 프로그램을 작동하는 shell이 존재
		- shell이라는 명령을 통해 안쪽 core를 다룰 수 있다
		- cd, clear, cp, rm, ls, mkdir ...