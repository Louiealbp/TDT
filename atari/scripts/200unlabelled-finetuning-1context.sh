# Script for generating data for vision encoder transfer with pretraining on 200 missions and 1 context
for seed in 123 231 312
do
    CUDA_VISIBLE_DEVICES=5 python run_dt_atari.py --context_length 1 --top_k 2 --ckpt_path ./../logs-1context/create-unlabelled-encoder-200-$seed/1 --unlabelled_extra_data --no_labelled_missions --num_unlabelled_missions 200 --seed $seed --data_dir $DATA_DIR --skip_eval 2 --epochs 10
done

for seed in 123 231 312
do
    CUDA_VISIBLE_DEVICES=5 taskset -c 120-130 python run_dt_atari.py --context_length 1 --top_k 2 --ckpt_path ./../logs-1context/load-unlabelled-encoder-reach-level-200-$seed/1 --load_vision_encoder --vision_encoder_path ./../logs-1context/create-unlabelled-encoder-200-$seed/1encoder.pt --only_reach_level --seed $seed --data_dir $DATA_DIR --skip_eval 2 --epochs 10
done

for seed in 123 231 312
do
    CUDA_VISIBLE_DEVICES=5 taskset -c 120-130 python run_dt_atari.py --context_length 1 --top_k 2 --ckpt_path ./../logs-1context/load-unlabelled-encoder-max-points-200-$seed/1 --load_vision_encoder --vision_encoder_path ./../logs-1context/create-unlabelled-encoder-200-$seed/1encoder.pt --only_max_points --seed $seed --data_dir $DATA_DIR --skip_eval 2 --epochs 10
done

for seed in 123 231 312
do
    CUDA_VISIBLE_DEVICES=5 taskset -c 120-130 python run_dt_atari.py --context_length 1 --top_k 2 --ckpt_path ./../logs-1context/load-unlabelled-encoder-certain-points-200-$seed/1 --load_vision_encoder --vision_encoder_path ./../logs-1context/create-unlabelled-encoder-200-$seed/1encoder.pt --only_certain_points --seed $seed --data_dir $DATA_DIR --skip_eval 2 --epochs 10
done

for seed in 123 231 312
do
    CUDA_VISIBLE_DEVICES=5 taskset -c 120-130 python run_dt_atari.py --context_length 1 --top_k 2 --ckpt_path ./../logs-1context/load-unlabelled-encoder-only-jump-200-$seed/1 --load_vision_encoder --vision_encoder_path ./../logs-1context/create-unlabelled-encoder-200-$seed/1encoder.pt --only_jump --seed $seed --data_dir $DATA_DIR --skip_eval 2 --epochs 10
done

for seed in 123 231 312
do
    CUDA_VISIBLE_DEVICES=5 taskset -c 120-130 python run_dt_atari.py --context_length 1 --top_k 2 --ckpt_path ./../logs-1context/load-unlabelled-encoder-only-left-right-200-$seed/1 --load_vision_encoder --vision_encoder_path ./../logs-1context/create-unlabelled-encoder-200-$seed/1encoder.pt --only_left_right --seed $seed --data_dir $DATA_DIR --skip_eval 2 --epochs 10
done

for seed in 123 231 312
do
    CUDA_VISIBLE_DEVICES=5 taskset -c 120-130 python run_dt_atari.py --context_length 1 --top_k 2 --ckpt_path ./../logs-1context/load-unlabelled-encoder-only-dont-move-200-$seed/1 --load_vision_encoder --vision_encoder_path ./../logs-1context/create-unlabelled-encoder-200-$seed/1encoder.pt --only_dont_move --seed $seed --data_dir $DATA_DIR --skip_eval 2 --epochs 10
done

for seed in 123 231 312
do
    CUDA_VISIBLE_DEVICES=5 taskset -c 120-130 python run_dt_atari.py --context_length 1 --top_k 2 --ckpt_path ./../logs-1context/load-unlabelled-encoder-only-certain-ice-floe-200-$seed/1 --load_vision_encoder --vision_encoder_path ./../logs-1context/create-unlabelled-encoder-200-$seed/1encoder.pt --only_certain_ice_floe --seed $seed --data_dir $DATA_DIR --skip_eval 2 --epochs 10
done

for seed in 123 231 312
do
    CUDA_VISIBLE_DEVICES=5 taskset -c 120-130 python run_dt_atari.py --context_length 1 --top_k 2 --ckpt_path ./../logs-1context/load-unlabelled-encoder-only-die-level-200-$seed/1 --load_vision_encoder --vision_encoder_path ./../logs-1context/create-unlabelled-encoder-200-$seed/1encoder.pt --only_die_level --seed $seed --data_dir $DATA_DIR --skip_eval 2 --epochs 10
done
