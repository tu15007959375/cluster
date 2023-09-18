#！/bin/bash
matirxpath="/home/tujinxing/matrixdata"
dev=4
#light spmv
if [ $1 = 'light' ]
then
    echo "\e[36m----------------------------light---------------------------------- \e[0m"
    ./LightSpMV-1.0/lightspmv -g "${dev}" -i "${matirxpath}"/$2.mtx
elif [ $1 = 'merge' ]
#merge spmv
then
    echo "\e[36m----------------------------merge---------------------------------- \e[0m"
    cd merge
    ./gpu_spmv --device="${dev}" --mtx="${matirxpath}"/$2.mtx
elif [ $1 = 'csr5' ]
#csr5 spmv
then
    echo "\e[36m----------------------------csr5----------------------------------- \e[0m"
    ./csr5/spmv "${dev}" "${matirxpath}"/$2.mtx
else
    echo -n "$1 "
    if [ ! -f "${matirxpath}/$1.mtx" ]; then
        echo ""
    else
        # echo "\e[36m----------------------------light---------------------------------- \e[0m"
        ./LightSpMV-1.0/lightspmv -g "${dev}" -i "${matirxpath}"/$1.mtx
        #./lightspmv -g 4 -i /home/tujinxing/af_shell4.mtx
        # echo "\e[36m----------------------------csr5----------------------------------- \e[0m"
        ./csr5/spmv "${dev}" "${matirxpath}"/$1.mtx
        # echo "\e[36m----------------------------merge---------------------------------- \e[0m"
        cd merge
        ./gpu_spmv --device="${dev}" --mtx="${matirxpath}"/$1.mtx
        #./_gpu_spmv_driver --device=3 --mtx=af_shell4.mtx
    fi
fi