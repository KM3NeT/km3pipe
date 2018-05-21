PKGNAME=km3pipe
SUFFIX=${DOCKER_NAME}

default: build

all: install

build: 
	@echo "No need to build anymore :)"

install: 
	pip install -U numpy
	pip install .

install-dev:
	pip install -U numpy
	pip install -e .

clean:
	python setup.py clean --all
	rm -f $(PKGNAME)/*.cpp
	rm -f $(PKGNAME)/*.c
	rm -f -r build/
	rm -f $(PKGNAME)/*.so

test: 
	py.test --junitxml=./reports/junit$(SUFFIX).xml km3pipe

test-km3modules: 
	py.test --junitxml=./reports/junit_km3modules$(SUFFIX).xml km3modules

test-cov:
	py.test --cov ./ --cov-report term-missing --cov-report xml:reports/coverage$(SUFFIX).xml --cov-report html:reports/coverage km3pipe km3modules pipeinspector

test-loop: 
	# pip install -U pytest-watch
	py.test
	ptw --ext=.py,.pyx --ignore=doc

flake8: 
	py.test --flake8
	py.test --flake8 km3modules

pep8: flake8

docstyle: 
	py.test --docstyle
	py.test --docstyle km3modules

lint: 
	py.test --pylint
	py.test --pylint km3modules

dependencies:
	pip install -U numpy
	pip install -Ur requirements.txt

.PHONY: all clean build install install-dev test test-km3modules test-nocov flake8 pep8 dependencies docstyle
