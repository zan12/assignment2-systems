#!/bin/bash
set -e

GCS_BUCKET="gs://permanent-us-central2-0rxn/aonanzhang/projects/nsys"

# Model configurations: name d_model d_ff num_layers num_heads
MODELS=(
    "small 768 3072 12 12"
    "medium 1024 4096 24 16" 
    "large 1280 5120 36 20"
    "xl 1600 6400 48 25"
    "2-7b 2560 10240 32 32"
)

profile_and_upload() {
    local name=$1 d_model=$2 d_ff=$3 num_layers=$4 num_heads=$5 amp_suffix=$6 amp_flag=$7
    local output="result-${name}-mp-${amp_suffix}"
    
    uv run nsys profile -o $output python -m cs336_systems.ref_transformer \
        --context_length 512 --d_model $d_model --d_ff $d_ff \
        --num_layers $num_layers --num_heads $num_heads $amp_flag
    
    gsutil cp ./${output}.nsys-rep $GCS_BUCKET
}

# Profile with and without AMP
for model in "${MODELS[@]}"; do
    read -r name d_model d_ff num_layers num_heads <<< "$model"
    profile_and_upload $name $d_model $d_ff $num_layers $num_heads "on" "--use_amp"
    profile_and_upload $name $d_model $d_ff $num_layers $num_heads "off" ""
done