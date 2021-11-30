# Decision Transformer (DT)
for seed in 123 231 312
do
    CUDA_VISIBLE_DEVICES=3 python experiment.py --seed $seed --data_dir $DATA_DIR --val_dir $VAL_DIR --path ./final_logs/untrained_word_to_idx_allmissions_6layers-512embed-seed-$seed --embed_dim 512 --bert_encode --gpt2_dataset --first_state --bc --K 1
done
