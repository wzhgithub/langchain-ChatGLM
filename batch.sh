#!/bin/bash
set -x
rm logs/*.log
i=0
for letter in {a..j}
do
  p="/home/zh.wang/chatglm_llm_fintech_raw_dataset/test_questions_a${letter}_res.json"
  echo $p $i
  nohup bash cli.sh start qa --submit_file $p --index $i --device_num 8 >logs/qa_${i}.log 2>&1 &
  let "i=i+1"
  sleep 10s
done
