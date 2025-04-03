TS=$(date "+%Y%0m%0d_%T")

project_root_path="UCD"
cli_path="${project_root_path}/src/benchmark_evaluation/truthfulqa_eval.py"
data_path="${project_root_path}/data/truthfulqa"

model_name="Your exp model name"
amateur_model_name="Your base model name"

### Baseline
output_path= "Your_path"
mkdir -p $output_path
cp $0 "$(dirname "$output_path")"


generation_args="
    --relative_top 0.0
"

echo "Greedy Decoding"
for i in 0 1 2 3 4 5 6 7 ; do
    echo "devices: ${i}"
    CMD="CUDA_VISIBLE_DEVICES=$i nohup python ${cli_path}
        --model-name ${model_name} \
        --num-gpus 1 \
        --data-path ${data_path} \
        --output-path ${output_path}"/result" \
        --is-chat \
        --mode greedy \
        --parallel \
        --total-shard 8 \
        --shard-id $i \
        ${generation_args} \
        >${output_path}/shard_${i}.log 2>&1 &"
    echo $CMD
    eval $CMD
done
wait

### Our method
output_path= "Your_path"
mkdir -p $output_path
cp $0 "$(dirname "$output_path")"


echo "ICD"
for i in 0 1 2 3 4 5 6 7 ; do
    echo "devices: ${i}"
    CMD="CUDA_VISIBLE_DEVICES=$i nohup python ${cli_path}
        --model-name ${model_name} \
        --amateur-model-name ${amateur_model_name} \
        --num-gpus 1 \
        --amateur-model-nums-gpus 1 \
        --data-path ${data_path} \
        --output-path ${output_path}"/result" \
        --is-chat \
        --mode UCD \
        --parallel \
        --total-shard 8 \
        --shard-id $i \
        ${generation_args} \
        >${output_path}/shard_${i}.log 2>&1 &"
    echo $CMD
    eval $CMD

done
wait
