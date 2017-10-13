.PHONY: test
test:
	py.test test/utils_unittests.py -vv
	py.test --pep8 src/utils.py
