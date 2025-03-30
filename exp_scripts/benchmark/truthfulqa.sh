TS=$(date "+%Y%0m%0d_%T")

project_root_path="UCD"
cli_path="${project_root_path}/src/benchmark_evaluation/truthfulqa_eval.py"
data_path="${project_root_path}/data/truthfulqa"

### Exps with Llama2-7B
model_name="/data1/hklee/co-llm/weights/Mixtral-8x7B-v0.1"
amateur_model_name="/data1/hklee/co-llm/weights/Mistral-7B-v0.1"

# ### For experiments using Baichuan2
# model_name="baichuan-inc/ Baichuan2-7B-Chat"
# amateur_model_name="HillZhang/untruthful_baichuan2_7b"

# ### For experiments using Mistral
# model_name="mistralai/Mistral-7B-Instruct-v0.1"
# amateur_model_name="HillZhang/untruthful_mistral_7b"

### Baseline
output_path="${project_root_path}/exp_results/truthfulqa/${TS}/beta_실험"
mkdir -p $output_path
cp $0 "$(dirname "$output_path")"

generation_args="
    --relative_top 0.0
"
#CMD + / 누르고 주석 풀넣

# echo "BASE"
# for i in 0 1 2 3 4 5 6 7 ; do
#     echo "devices: ${i}"
#     CMD="CUDA_VISIBLE_DEVICES=$i nohup python ${cli_path}
#         --model-name ${model_name} \
#         --num-gpus 1 \
#         --data-path ${data_path} \
#         --output-path ${output_path}"/result" \
#         --is-chat \
#         --mode greedy \
#         --parallel \
#         --total-shard 8 \
#         --shard-id $i \
#         ${generation_args} \
#         >${output_path}/shard_${i}.log 2>&1 &"
#     echo $CMD
#     eval $CMD
# done
# wait



### Our method
output_path="${project_root_path}/리뷰탈/truthfulqa/모델개선/ucd"
mkdir -p $output_path
cp $0 "$(dirname "$output_path")"

echo "Cumulative_Entropy-contrastive-decoding"
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
        --mode Cumulative-Energy-contrastive-decoding2 \
        --parallel \
        --relative_top 0.0 \
        --total-shard 8 \
        --shard-id $i \
        ${generation_args} \
        >${output_path}/shard_${i}.log 2>&1 &"
    echo $CMD
    eval $CMD

done
wait


# ### Our method
# output_path="${project_root_path}/exp_results/truthfulqa/${TS}/W_Entropy_llama2_7b_chat"
# mkdir -p $output_path
# cp $0 "$(dirname "$output_path")"

# echo "Cumulative_Entropy-contrastive-decoding"
# for i in 0 1 2 3 4 5 6 7 ; do
#     echo "devices: ${i}"
#     CMD="CUDA_VISIBLE_DEVICES=$i nohup python ${cli_path}
#         --model-name ${model_name} \
#         --amateur-model-name ${amateur_model_name} \
#         --num-gpus 1 \
#         --amateur-model-nums-gpus 1 \
#         --data-path ${data_path} \
#         --output-path ${output_path}"/result" \
#         --is-chat \
#         --mode W_Entropy-contrastive-decoding \
#         --parallel \
#         --total-shard 8 \
#         --shard-id $i \
#         ${generation_args} \
#         >${output_path}/shard_${i}.log 2>&1 &"
#     echo $CMD
#     eval $CMD

# done
# wait
