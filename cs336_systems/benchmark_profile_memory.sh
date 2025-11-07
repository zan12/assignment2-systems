# uv run python -m cs336_systems.benchmark --context_length 128 --d_model 2560 --d_ff 10240 --num_layers 32 --num_heads 32 --num_steps 1 --use_amp --profile_memory
# gsutil cp ./memory_snapshot_2_7b_c128_forward.pickle gs://permanent-us-central2-0rxn/aonanzhang/projects/memory_snapshot
# uv run python -m cs336_systems.benchmark --context_length 256 --d_model 2560 --d_ff 10240 --num_layers 32 --num_heads 32 --num_steps 1 --use_amp --profile_memory
# gsutil cp ./memory_snapshot_2_7b_c256_forward.pickle gs://permanent-us-central2-0rxn/aonanzhang/projects/memory_snapshot
# uv run python -m cs336_systems.benchmark --context_length 512 --d_model 2560 --d_ff 10240 --num_layers 32 --num_heads 32 --num_steps 1 --use_amp --profile_memory
# gsutil cp ./memory_snapshot_2_7b_c512_forward.pickle gs://permanent-us-central2-0rxn/aonanzhang/projects/memory_snapshot
uv run python -m cs336_systems.benchmark --context_length 512 --d_model 2560 --d_ff 10240 --num_layers 32 --num_heads 32 --num_steps 1 --profile_memory
gsutil cp ./memory_snapshot_2_7b_c512_full_fp.pickle gs://permanent-us-central2-0rxn/aonanzhang/projects/memory_snapshot