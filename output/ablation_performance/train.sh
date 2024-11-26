echo "Current working directory: $(pwd)"
export PYTHONPATH='.'
weight_names=(
  "equal"
  "std"
  "mean"
  "max_min"
)

gpu_ids=(0 1 2 3 4 5 6 7)

train_with_weight(){
  local weight_name=$1
  local gpu_id=$2

  echo "[START] - Start training on GPU $gpu_id, $weight_name"
  CUDA_VISIBLE_DEVICES=$gpu_id python ablation_train_performance_model.py --weight_name "$weight_name"
  echo "[FINISH] - Finish Evaluation on GPU $gpu_id, $weight_name"
}

# Run tuning and evaluation in parallel on GPUs 0-7
for i in "${!weight_names[@]}"; do
    weight_name=${weight_names[$i]}
    gpu_id=${gpu_ids[$i]}
    train_with_weight "$weight_name" "$gpu_id" &
done
wait  # Wait for all tuning and evaluation proce