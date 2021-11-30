# Script for pretraining on 200 missions
for seed in 123 231 312
do
    CUDA_VISIBLE_DEVICES=3 python run_dt_atari.py --context_length 20 --top_k 2 --ckpt_path ./../logs/create-unlabelled-encoder-all-$seed/1 --unlabelled_extra_data --no_labelled_missions --num_unlabelled_missions 200 --seed $seed --data_dir $DATA_DIR
done
