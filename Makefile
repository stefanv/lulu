.PHONY: all clean test

all:
	python setup.py build_ext --inplace

clean:
	rm *.so *.c

test:
	nosetests

