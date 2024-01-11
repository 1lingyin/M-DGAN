#!/bin/bash
#2023/10/2
#written by ZhangYang
#采样音频48k采样到16k采样
#只能一个文件一个文件夹转换
input_folder="G:\PYPro\dataset\data\test\noisy_testset_wav"
output_folder="G:\PYPro\dataset\data\test\noisy_testset_16k_wav"
let a=0
for file in "$input_folder"/*.wav; do
    filename=$(basename "$file")        # 获取文件名（包含后缀）
    extension="${filename##*.}"         # 获取文件后缀
    filename="${filename%.*}"           # 获取文件名（不含后缀）
    sox "$file" -r 16000 "$output_folder/$filename.$extension"
    let a+=1
    echo "ok $a"
done
echo "完成转换"
