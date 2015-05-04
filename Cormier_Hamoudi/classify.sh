#!/bin/bash

python3 src/classify.py --classes classes.csv --eigenvalues eigenvalues/eigenvalues.db $1
