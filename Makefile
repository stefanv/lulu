.PHONY: all clean test

all:
	python setup.py build_ext --inplace

clean:
	rm lulu/*.so lulu/*.c

test:
	nosetests -v

