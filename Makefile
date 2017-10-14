.PHONY: test
test:
	py.test test/utils_unittests.py -vv
	py.test test/process_tweets_unittests.py -vv
	py.test --pep8 src/utils.py
	py.test --pep8 src/process_tweets.py

