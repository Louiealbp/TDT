# Decision Transformer (DT)
for seed in 123 231 312
do
    CUDA_VISIBLE_DEVICES=2 python run_dt_atari.py --context_length 20 --top_k 2 --ckpt_path ./../logs/create-unlabelled-encoder-10-$seed/1 --unlabelled_extra_data --num_unlabelled_missions 10 --no_labelled_missions --data_dir $DATA_DIR
done
