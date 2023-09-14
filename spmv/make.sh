cd ./csr5
make VALUE_TYPE=double
cd ../LightSpMV-1.0
make
cd ../merge
make gpu_spmv