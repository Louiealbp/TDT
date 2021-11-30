# Decision Transformer (DT)
for seed in 123 231 312
do
    CUDA_VISIBLE_DEVICES=2 python experiment.py --seed $seed --data_dir $DATA_DIR --val_dir $VAL_DIR --path ./final_logs/bert_allmissions_seed-$seed --gpt2_dataset --bert_encode --first_state --embed_dim 768
done
