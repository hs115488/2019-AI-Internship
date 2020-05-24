#!/bin/bash


# This script takes an input of repetitions and manually trains
#Once one repition is done, it appends the result into a text file


for i in {0..$1}
do 	
	python neural_network.py >> test.txt
done
