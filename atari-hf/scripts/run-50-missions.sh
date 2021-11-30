# Decision Transformer (DT)
for seed in 123 231 312
do
    CUDA_VISIBLE_DEVICES=0 python experiment.py --seed $seed --data_dir $DATA_DIR --val_dir $VAL_DIR --path ./final_logs/untrained_word_to_idx_50missions_seed-$seed --num_missions 50
done
