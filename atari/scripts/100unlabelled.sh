# Generate pre-trained models on 100 missions
for seed in 123 231 312
do
    CUDA_VISIBLE_DEVICES=3 python run_dt_atari.py --context_length 20 --top_k 2 --ckpt_path ./../logs/create-unlabelled-encoder-100-$seed/1 --unlabelled_extra_data --num_unlabelled_missions 100 --no_labelled_missions --seed $seed --data_dir $DATA_DIR
done
