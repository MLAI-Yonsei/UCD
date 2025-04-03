# UCD

This README provides setup instructions and usage guidance to run evaluations on the TruthfulQA dataset using both a baseline model and the proposed UCD method.

---

## 📦 Environment Setup

### 1. Create and Activate Conda Environment
```bash
conda create -n ucd python==3.10
conda activate ucd
```

### 2. Install Required Packages
```bash
pip install -r requirements.txt
```

### 3. Install Transformers Locally (Editable Mode)
```bash
cd ./transformers
pip install --editable ./
```

---

## 🧪 Run TruthfulQA Evaluation (Table 1)

### Step 1: Navigate to the Benchmark Script Directory
```bash
cd ./exp_scripts/benchmark
```

### Step 2: Edit the Script
Before running `truthfulqa.sh`, make sure to edit the following variables in the script:

- `project_root_path`: Root path to your project (e.g., `"UCD"`)
- `model_name`: Name or path of your experiment model
- `amateur_model_name`: Name or path of your amateur model (only used for UCD)
- `output_path`: Where to store the results and logs

> **Note:** In this code, "amateur_model" refers to the base model in our paper.
> ⚠️ **Important:** Make sure that the models you intend to use (`model_name` and `amateur_model_name`) are already downloaded or accessible. This script does not handle model downloading.

### Step 3: Run the Script
```bash
sh truthfulqa.sh
```

---

## ⚙️ Script Structure (truthfulqa.sh)

The script is structured to run across 8 GPUs using parallel execution with sharding. It supports two modes:

### 🔹 Baseline Inference
Evaluates using a single model in greedy decoding mode.

```bash
CMD="CUDA_VISIBLE_DEVICES=$i nohup python ${cli_path} \
    --model-name ${model_name} \
    --num-gpus 1 \
    --data-path ${data_path} \
    --output-path ${output_path}/result \
    --is-chat \
    --mode greedy \
    --parallel \
    --total-shard 8 \
    --shard-id $i \
    ${generation_args} \
    >${output_path}/shard_${i}.log 2>&1 &"
```

### 🔹 UCD Method
Evaluates using both the main model and an amateur model. Requires `--mode UCD` to activate the method.

```bash
CMD="CUDA_VISIBLE_DEVICES=$i nohup python ${cli_path} \
    --model-name ${model_name} \
    --amateur-model-name ${amateur_model_name} \
    --num-gpus 1 \
    --amateur-model-nums-gpus 1 \
    --data-path ${data_path} \
    --output-path ${output_path}/result \
    --is-chat \
    --mode UCD \
    --parallel \
    --relative_top 0.0 \
    --total-shard 8 \
    --shard-id $i \
    ${generation_args} \
    >${output_path}/shard_${i}.log 2>&1 &"
```


---

## 📁 Directory Structure Example
```
UCD/
├── data/
│   └── truthfulqa/
├── src/
│   └── benchmark_evaluation/
│       └── truthfulqa_eval.py
├── transformers/
│   └── (local transformers repo)
├── exp_scripts/
│   └── benchmark/
│       └── truthfulqa.sh
├── requirements.txt
└── README.md
```

---

Feel free to reach out for help if paths or settings don't work as expected!

