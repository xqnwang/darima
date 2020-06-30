#! /usr/bin/sh


PYSPARK_PYTHON=/usr/local/bin/python3.7 spark-submit \
	--master yarn \
        ./test-fli.py

exit 0;
