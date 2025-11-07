#!/bin/bash
set -e

GCS_BUCKET="gs://permanent-us-central2-0rxn/aonanzhang/projects/nsys/transformer"

# Model configurations: name d_model d_ff num_layers num_heads
MODELS=(
    "2-7b 2560 10240 32 32"
)

profile_and_upload() {
    local name=$1 d_model=$2 d_ff=$3 num_layers=$4 num_heads=$5 compile_suffix=$6 compile_flag=$7 profile_memory_flag=$8
    local output="result-${name}-compile-${compile_suffix}"
    
    uv run nsys profile -o $output python -m cs336_systems.ref_transformer \
        --context_length 512 --d_model $d_model --d_ff $d_ff \
        --num_layers $num_layers --num_heads $num_heads $compile_flag $profile_memory_flag
    
    gsutil cp ./${output}.nsys-rep $GCS_BUCKET
}

# Profile with and without AMP
for model in "${MODELS[@]}"; do
    read -r name d_model d_ff num_layers num_heads <<< "$model"
    # profile_and_upload $name $d_model $d_ff $num_layers $num_heads "on" "--compile_model" "--profile_memory"
    profile_and_upload $name $d_model $d_ff $num_layers $num_heads "off" "" "--profile_memory"
done