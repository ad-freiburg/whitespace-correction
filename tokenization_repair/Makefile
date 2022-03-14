.PHONY: demo
demo:
	streamlit run demo/demo.py -- -e experiments \
	-b benchmark/tokenization_repair_benchmarks/cleaned_benchmarks/ \
	-r benchmark/tokenization_repair_benchmarks/cleaned_results/

.PHONY: download_data
download_data:
	mkdir -p data/downloads

	@echo "Downloading latest Wikidump"
	@echo "(Cannot download Wikidump 2020/10/20 as used in the blog post since it is no longer available, using latest instead)"

	if [ ! -f data/downloads/enwiki-latest-pages-articles-multistream.xml.bz2 ] ; \
		then \
		wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles-multistream.xml.bz2 \
		-P data/downloads ; \
	fi;

	mkdir -p data/wikidump_20201020
	python -m wikiextractor.WikiExtractor data/downloads/enwiki-latest-pages-articles-multistream.xml.bz2 \
		-o data/wikidump_20201020/extracted
	wget https://github.com/ad-freiburg/tokenization-repair/blob/master/data/wikipedia_development_article_ids.txt \
		-P data/wikidump_20201020
	wget https://github.com/ad-freiburg/tokenization-repair/blob/master/data/wikipedia_test_article_ids.txt \
		-P data/wikidump_20201020

	@echo "Downloading Bookcorpus"

	if [ ! -f data/downloads/books1.tar.gz ] ; \
		then \
		wget wget https://battle.shawwn.com/sdb/books1/books1.tar.gz -P data/downloads ; \
	fi;

	mkdir -p data/bookcorpus
	tar -xzvf data/downloads/books1.tar.gz -C data/bookcorpus

.PHONY: help
help:
	@bash help.sh

.PHONY: tests
tests:
	make -C .. tests

.PHONY: checkstyle
checkstyle:
	make -C .. checkstyle
