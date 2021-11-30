# Script to generate data for transferring vision encoder with pre-training on 10 missions
for seed in 123 231 312
do
    CUDA_VISIBLE_DEVICES=4 python run_dt_atari.py --context_length 20 --top_k 2 --ckpt_path ./../logs-rerun/create-unlabelled-encoder-10-$seed/1 --unlabelled_extra_data --num_unlabelled_missions 10 --no_labelled_missions --seed $seed --data_dir $DATA_DIR --skip_eval 2 --epochs 10
done

for seed in 123 231 312
do
    CUDA_VISIBLE_DEVICES=4 taskset -c 120-130 python run_dt_atari.py --context_length 20 --top_k 2 --ckpt_path ./../logs-rerun/load-unlabelled-encoder-reach-level-10-$seed/1 --load_vision_encoder --vision_encoder_path ./../logs-rerun/create-unlabelled-encoder-10-$seed/1encoder.pt --only_reach_level --seed $seed --data_dir $DATA_DIR --skip_eval 2 --epochs 10
done

for seed in 123 231 312
do
    CUDA_VISIBLE_DEVICES=4 taskset -c 120-130 python run_dt_atari.py --context_length 20 --top_k 2 --ckpt_path ./../logs-rerun/load-unlabelled-encoder-max-points-10-$seed/1 --load_vision_encoder --vision_encoder_path ./../logs-rerun/create-unlabelled-encoder-10-$seed/1encoder.pt --only_max_points --seed $seed --data_dir $DATA_DIR --skip_eval 2 --epochs 10
done

for seed in 123 231 312
do
    CUDA_VISIBLE_DEVICES=4 taskset -c 120-130 python run_dt_atari.py --context_length 20 --top_k 2 --ckpt_path ./../logs-rerun/load-unlabelled-encoder-certain-points-10-$seed/1 --load_vision_encoder --vision_encoder_path ./../logs-rerun/create-unlabelled-encoder-10-$seed/1encoder.pt --only_certain_points --seed $seed --data_dir $DATA_DIR --skip_eval 2 --epochs 10
done

for seed in 123 231 312
do
    CUDA_VISIBLE_DEVICES=4 taskset -c 120-130 python run_dt_atari.py --context_length 20 --top_k 2 --ckpt_path ./../logs-rerun/load-unlabelled-encoder-only-jump-10-$seed/1 --load_vision_encoder --vision_encoder_path ./../logs-rerun/create-unlabelled-encoder-10-$seed/1encoder.pt --only_jump --seed $seed --data_dir $DATA_DIR --skip_eval 2 --epochs 10
done

for seed in 123 231 312
do
    CUDA_VISIBLE_DEVICES=4 taskset -c 120-130 python run_dt_atari.py --context_length 20 --top_k 2 --ckpt_path ./../logs-rerun/load-unlabelled-encoder-only-left-right-10-$seed/1 --load_vision_encoder --vision_encoder_path ./../logs-rerun/create-unlabelled-encoder-10-$seed/1encoder.pt --only_left_right --seed $seed --data_dir $DATA_DIR --skip_eval 2 --epochs 10
done

for seed in 123 231 312
do
    CUDA_VISIBLE_DEVICES=4 taskset -c 120-130 python run_dt_atari.py --context_length 20 --top_k 2 --ckpt_path ./../logs-rerun/load-unlabelled-encoder-only-dont-move-10-$seed/1 --load_vision_encoder --vision_encoder_path ./../logs-rerun/create-unlabelled-encoder-10-$seed/1encoder.pt --only_dont_move --seed $seed --data_dir $DATA_DIR --skip_eval 2 --epochs 10
done

for seed in 123 231 312
do
    CUDA_VISIBLE_DEVICES=4 taskset -c 120-130 python run_dt_atari.py --context_length 20 --top_k 2 --ckpt_path ./../logs-rerun/load-unlabelled-encoder-only-certain-ice-floe-10-$seed/1 --load_vision_encoder --vision_encoder_path ./../logs-rerun/create-unlabelled-encoder-10-$seed/1encoder.pt --only_certain_ice_floe --seed $seed --data_dir $DATA_DIR --skip_eval 2 --epochs 10
done

for seed in 123 231 312
do
    CUDA_VISIBLE_DEVICES=4 taskset -c 120-130 python run_dt_atari.py --context_length 20 --top_k 2 --ckpt_path ./../logs-rerun/load-unlabelled-encoder-only-die-level-10-$seed/1 --load_vision_encoder --vision_encoder_path ./../logs-rerun/create-unlabelled-encoder-10-$seed/1encoder.pt --only_die_level --seed $seed --data_dir $DATA_DIR --skip_eval 2 --epochs 10
done
