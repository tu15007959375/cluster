#！/bin/bash
 while read line
 do
	scp tujinxing1@111.115.201.25:/gpfs/public/matrixdata_untar/untar/$line.mtx  ~/
	sh run.sh $line >>test.out 2>&1
	rm ~/$line.mtx
done <  ./merge/ufl_matrices.txt
