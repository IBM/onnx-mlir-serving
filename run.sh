#!/bin/bash
# Usage: run.sh AccuracyOnly dlc
zdnn="/aivol/DLCpp/third_party/zdnn-lib/libzdnn.so"
onnx_root="/aivol/onnx-zoo-scripts"
input="/aivol/inputs"
mode="PerformanceOnly"
scenario=(SingleStream MultiStream Server Offline)
if [ $mode == "AccuracyOnly" ];then
  scenario=(SingleStream)
fi
s_len=${#scenario[@]}


config="models_input.txt"
IFS=" "

rm mlperf_log_*
result_dir=`date '+%Y%m%d-%H%M%S'`
mkdir $result_dir
for (( i=0; i<$s_len; i++ ))
do
  while read -r model minput moutput
  do
    echo app aiuConfig.cfg $mode ${scenario[$i]} $input/$minput
    # export LD_LIBRARY=./
    # export ${PRELOAD_STR} 
    ./app aiuConfig.cfg $mode ${scenario[$i]} $input/$minput
    tar -zcvf ${scenario[$i]}_$moutput.tar.gz mlperf_log_*
    rm mlperf_log_*
  done < $config
done

mv *.tar.gz $result_dir