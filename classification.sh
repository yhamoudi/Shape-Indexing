#!/bin/bash

python3 src/classify.py --classes classes.csv --eigenvalues eigenvalues/ev3.db $1
