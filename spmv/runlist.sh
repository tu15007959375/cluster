#！/bin/bash
 while read line
 do
	sh run.sh $line >>res.out 2>&1
done <  ./diatest.txt
