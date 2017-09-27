.PHONY: test
test:
	py.test test/unittests.py -vv
	py.test --pep8 src/utils.py
