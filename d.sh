uv run python -m src.dataset_operation.generate_parallel \
  --preset munich_elaa_m_1k_28g --num_snapshots 20000 --num_ue 9 \
  --gpus $(for g in 0 1 2 3 4 5; do for i in $(seq 8); do echo -n "$g "; done; done) \
  --trajectories assets/data/shared_trajectories/trajectories.npz \
  --data_dir assets/data/channels_elaa_m_1k_28g_temporal \
  --start_snapshot 0

uv run python -m src.dataset_operation.generate_parallel \
  --preset munich_5g_mimo_3g5 --num_snapshots 20000 --num_ue 9 \
  --gpus $(for i in $(seq 8); do echo -n "6 "; done) \
  --trajectories assets/data/shared_trajectories/trajectories.npz \
  --data_dir assets/data/channels_5g_mimo_3g5_temporal \
  --start_snapshot 0