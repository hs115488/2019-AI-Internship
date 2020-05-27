#!/bin/bash


cd Users/njcu/Downloads/AutomaticStockTrading/

file = "/Stocks.csv"
file1 = "/Output.csv"
sed -e "s/ /,/g" $file 

[[ -f $file ]] || echo "File does not exist. Please try again"

date = awk '{print $1}' $file
close = awk {print $4}  $file
test_file = $date$close

awk -F '{
          getline file1 < $file; print file1, $1, $4
        }' OFS=, $file
        
cat $file1

echo "Your file is complete"
echo "It is located at $(pwd)"
echo "Goodbye"

exit
