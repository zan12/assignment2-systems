#!/bin/bash
set -e

GCS_BUCKET="gs://permanent-us-central2-0rxn/aonanzhang/projects/nsys/transformer"

for ctx_len in 128 256 512 1024; do
    output="result-c${ctx_len}"
    uv run nsys profile -o $output python -m cs336_systems.ref_transformer --context_length $ctx_len
    gsutil cp ./${output}.nsys-rep $GCS_BUCKET
done