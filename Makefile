.PHONY: clean data

all: clean data

data:
	rm -rf data/interim/*
	rm -rf data/processed/*
	python -m bowel.data.preprocess

clean:
	rm -rf data/interim/*
	rm -rf data/processed/*
