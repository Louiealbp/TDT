# Script for generating data for full model transfer with pretraining on 100 missions
for seed in 123 231 312
do
    CUDA_VISIBLE_DEVICES=3 python run_dt_atari.py --context_length 20 --top_k 2 --ckpt_path ./../logs/create-unlabelled-encoder-100-$seed/1 --unlabelled_extra_data --num_unlabelled_missions 100 --no_labelled_missions --seed $seed --data_dir $DATA_DIR
done

for seed in 123 231 312
do
    CUDA_VISIBLE_DEVICES=0 taskset -c 110-120 python run_dt_atari.py --context_length 20 --top_k 2 --ckpt_path ./../logs-final/load-unlabelled-model-reach-level-100-$seed/1 --load_model --vision_encoder_path ./../logs/create-unlabelled-encoder-100-$seed/1encoder.pt --only_reach_level --seed $seed --data_dir $DATA_DIR
done

for seed in 123 231 312
do
    CUDA_VISIBLE_DEVICES=0 taskset -c 110-120 python run_dt_atari.py --context_length 20 --top_k 2 --ckpt_path ./../logs-final/load-unlabelled-model-max-points-100-$seed/1 --load_model --vision_encoder_path ./../logs/create-unlabelled-encoder-100-$seed/1encoder.pt --only_max_points --seed $seed --data_dir $DATA_DIR
done

for seed in 123 231 312
do
    CUDA_VISIBLE_DEVICES=0 taskset -c 110-120 python run_dt_atari.py --context_length 20 --top_k 2 --ckpt_path ./../logs-final/load-unlabelled-model-certain-points-100-$seed/1 --load_model --vision_encoder_path ./../logs/create-unlabelled-encoder-100-$seed/1encoder.pt --only_certain_points --seed $seed --data_dir $DATA_DIR
done

for seed in 123 231 312
do
    CUDA_VISIBLE_DEVICES=0 taskset -c 110-120 python run_dt_atari.py --context_length 20 --top_k 2 --ckpt_path ./../logs-final/load-unlabelled-model-only-jump-100-$seed/1 --load_model --vision_encoder_path ./../logs/create-unlabelled-encoder-100-$seed/1encoder.pt --only_jump --seed $seed --data_dir $DATA_DIR
done

for seed in 123 231 312
do
    CUDA_VISIBLE_DEVICES=0 taskset -c 110-120 python run_dt_atari.py --context_length 20 --top_k 2 --ckpt_path ./../logs-final/load-unlabelled-model-only-left-right-100-$seed/1 --load_model --vision_encoder_path ./../logs/create-unlabelled-encoder-100-$seed/1encoder.pt --only_left_right --seed $seed --data_dir $DATA_DIR
done

for seed in 123 231 312
do
    CUDA_VISIBLE_DEVICES=0 taskset -c 110-120 python run_dt_atari.py --context_length 20 --top_k 2 --ckpt_path ./../logs-final/load-unlabelled-model-only-dont-move-100-$seed/1 --load_model --vision_encoder_path ./../logs/create-unlabelled-encoder-100-$seed/1encoder.pt --only_dont_move --seed $seed --data_dir $DATA_DIR
done

for seed in 123 231 312
do
    CUDA_VISIBLE_DEVICES=0 taskset -c 110-120 python run_dt_atari.py --context_length 20 --top_k 2 --ckpt_path ./../logs-final/load-unlabelled-model-only-certain-ice-floe-100-$seed/1 --load_model --vision_encoder_path ./../logs/create-unlabelled-encoder-100-$seed/1encoder.pt --only_certain_ice_floe --seed $seed --data_dir $DATA_DIR
done

for seed in 123 231 312
do
    CUDA_VISIBLE_DEVICES=0 taskset -c 110-120 python run_dt_atari.py --context_length 20 --top_k 2 --ckpt_path ./../logs-final/load-unlabelled-model-only-die-level-100-$seed/1 --load_model --vision_encoder_path ./../logs/create-unlabelled-encoder-100-$seed/1encoder.pt --only_die_level --seed $seed --data_dir $DATA_DIR
done
