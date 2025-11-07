for d_model in 16 32 64 128; do
  for seq_len in 256 1024 4096 8192 16384; do
    echo "Running for d_model=$d_model, seq_len=$seq_len"
    uv run nsys profile -o result-d${d_model}-t${seq_len}-compiled python -m cs336_systems.my_attention --d_model ${d_model} --seq_len ${seq_len}
    gsutil cp memory_snapshot_d${d_model}_t${seq_len}_compiled.pickle gs://permanent-us-central2-0rxn/aonanzhang/projects/memory_snapshot/attention
    gsutil cp result-d${d_model}-t${seq_len}-compiled.nsys-rep gs://permanent-us-central2-0rxn/aonanzhang/projects/nsys/attention
  done
done