struct COOElement {
    int row;
    int col;
    float value;
};
struct DIAFormat {
    int num_diagonals;
    int max_diag_length;
    int* diagonals_offsets;
    float* diagonals_values;
};
struct BRCSD2 {
    int nrows;//每块行数
    int tile_size;//块数
    int ac_size;//合并后的块数
    int *offset_key;//后续每块中包含的块数量,ex,1,2
    int *val_size;//每块的dia对角线数量,,ex:3，2
    int *offset_val;//每块的dia偏移,一维数组,ex:[-1，0，2],[0，1]
    float *data;//数据,一维数组,ex:{[0, 3, 1, 4, 2, 5],[6, 8, 7, 0],[9, 11, 10, 0]}.
};
struct BDIA {
    int tile_size;//块数
    int *row_begin;//每一块开始的行偏移【0，2，5，】
    long *data_begin;//每一块开始的数据偏移【0，5，20，】
    int *dia_begin;//每一块开始的对角线偏移【0,3,5】
    int *row_tile_index;//每一块所在的块号
    int *val_size;//每块的dia对角线数量,,ex:3，2
    int *offset_val;//每块的dia偏移,一维数组,ex:[-1，0，2],[0，1]
    float *data;//数据,一维数组,ex:{[0, 1, 2, 3, 4, 5],[6, 7, 8, 0],[9, 10, 11, 0]}.
};