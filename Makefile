.PHONY: test
test:
	py.test test/utils_unittests.py -vv
	py.test test/process_tweets_unittests.py -vv
	py.test test/twitter_scraper_unittests.py -vv

test-syntax:
	py.test --pep8 src/utils.py
	py.test --pep8 src/process_tweets.py
	py.test --pep8 src/model.py
	py.test --pep8 src/twitter_scraper.py
