# Script for generating data for transferring vision encoder with pre-training on 50 missions on 1 context
for seed in 123 231 312
do
    CUDA_VISIBLE_DEVICES=3 python run_dt_atari.py --context_length 1 --top_k 2 --ckpt_path ./../logs-1context/create-unlabelled-encoder-50-$seed/1 --unlabelled_extra_data --num_unlabelled_missions 50 --no_labelled_missions --seed $seed --data_dir $DATA_DIR --skip_eval 2 --epochs 10
done

for seed in 123 231 312
do
    CUDA_VISIBLE_DEVICES=3 taskset -c 120-130 python run_dt_atari.py --context_length 1 --top_k 2 --ckpt_path ./../logs-1context/load-unlabelled-encoder-reach-level-50-$seed/1 --load_vision_encoder --vision_encoder_path ./../logs-1context/create-unlabelled-encoder-50-$seed/1encoder.pt --only_reach_level --seed $seed --data_dir $DATA_DIR --skip_eval 2 --epochs 10
done

for seed in 123 231 312
do
    CUDA_VISIBLE_DEVICES=3 taskset -c 120-130 python run_dt_atari.py --context_length 1 --top_k 2 --ckpt_path ./../logs-1context/load-unlabelled-encoder-max-points-50-$seed/1 --load_vision_encoder --vision_encoder_path ./../logs-1context/create-unlabelled-encoder-50-$seed/1encoder.pt --only_max_points --seed $seed --data_dir $DATA_DIR --skip_eval 2 --epochs 10
done

for seed in 123 231 312
do
    CUDA_VISIBLE_DEVICES=3 taskset -c 120-130 python run_dt_atari.py --context_length 1 --top_k 2 --ckpt_path ./../logs-1context/load-unlabelled-encoder-certain-points-50-$seed/1 --load_vision_encoder --vision_encoder_path ./../logs-1context/create-unlabelled-encoder-50-$seed/1encoder.pt --only_certain_points --seed $seed --data_dir $DATA_DIR --skip_eval 2 --epochs 10
done

for seed in 123 231 312
do
    CUDA_VISIBLE_DEVICES=3 taskset -c 120-130 python run_dt_atari.py --context_length 1 --top_k 2 --ckpt_path ./../logs-1context/load-unlabelled-encoder-only-jump-50-$seed/1 --load_vision_encoder --vision_encoder_path ./../logs-1context/create-unlabelled-encoder-50-$seed/1encoder.pt --only_jump --seed $seed --data_dir $DATA_DIR --skip_eval 2 --epochs 10
done

for seed in 123 231 312
do
    CUDA_VISIBLE_DEVICES=3 taskset -c 120-130 python run_dt_atari.py --context_length 1 --top_k 2 --ckpt_path ./../logs-1context/load-unlabelled-encoder-only-left-right-50-$seed/1 --load_vision_encoder --vision_encoder_path ./../logs-1context/create-unlabelled-encoder-50-$seed/1encoder.pt --only_left_right --seed $seed --data_dir $DATA_DIR --skip_eval 2 --epochs 10
done

for seed in 123 231 312
do
    CUDA_VISIBLE_DEVICES=3 taskset -c 120-130 python run_dt_atari.py --context_length 1 --top_k 2 --ckpt_path ./../logs-1context/load-unlabelled-encoder-only-dont-move-50-$seed/1 --load_vision_encoder --vision_encoder_path ./../logs-1context/create-unlabelled-encoder-50-$seed/1encoder.pt --only_dont_move --seed $seed --data_dir $DATA_DIR --skip_eval 2 --epochs 10
done

for seed in 123 231 312
do
    CUDA_VISIBLE_DEVICES=3 taskset -c 120-130 python run_dt_atari.py --context_length 1 --top_k 2 --ckpt_path ./../logs-1context/load-unlabelled-encoder-only-certain-ice-floe-50-$seed/1 --load_vision_encoder --vision_encoder_path ./../logs-1context/create-unlabelled-encoder-50-$seed/1encoder.pt --only_certain_ice_floe --seed $seed --data_dir $DATA_DIR --skip_eval 2 --epochs 10
done

for seed in 123 231 312
do
    CUDA_VISIBLE_DEVICES=3 taskset -c 120-130 python run_dt_atari.py --context_length 1 --top_k 2 --ckpt_path ./../logs-1context/load-unlabelled-encoder-only-die-level-50-$seed/1 --load_vision_encoder --vision_encoder_path ./../logs-1context/create-unlabelled-encoder-50-$seed/1encoder.pt --only_die_level --seed $seed --data_dir $DATA_DIR --skip_eval 2 --epochs 10
done
