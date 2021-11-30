# Script for generating data for full model transfer with pretraining on 200 missions
for seed in 123 231 312
do
    CUDA_VISIBLE_DEVICES=3 python run_dt_atari.py --context_length 20 --top_k 2 --ckpt_path ./../logs/create-unlabelled-encoder-all-$seed/1 --unlabelled_extra_data --no_labelled_missions --num_unlabelled_missions 200 --seed $seed --data_dir $DATA_DIR
done

for seed in 123 231 312
do
    CUDA_VISIBLE_DEVICES=3 taskset -c 100-110 python run_dt_atari.py --context_length 20 --top_k 2 --ckpt_path ./../logs-final/load-unlabelled-model-reach-level-all-$seed/1 --load_model --vision_encoder_path ./../logs/create-unlabelled-encoder-all-$seed/1encoder.pt --only_reach_level --seed $seed --data_dir $DATA_DIR
done

for seed in 123 231 312
do
    CUDA_VISIBLE_DEVICES=3 taskset -c 100-110 python run_dt_atari.py --context_length 20 --top_k 2 --ckpt_path ./../logs-final/load-unlabelled-model-max-points-all-$seed/1 --load_model --vision_encoder_path ./../logs/create-unlabelled-encoder-all-$seed/1encoder.pt --only_max_points --seed $seed --data_dir $DATA_DIR
done

for seed in 123 231 312
do
    CUDA_VISIBLE_DEVICES=3 taskset -c 100-110 python run_dt_atari.py --context_length 20 --top_k 2 --ckpt_path ./../logs-final/load-unlabelled-model-certain-points-all-$seed/1 --load_model --vision_encoder_path ./../logs/create-unlabelled-encoder-all-$seed/1encoder.pt --only_certain_points --seed $seed --data_dir $DATA_DIR
done

for seed in 123 231 312
do
    CUDA_VISIBLE_DEVICES=3 taskset -c 100-110 python run_dt_atari.py --context_length 20 --top_k 2 --ckpt_path ./../logs-final/load-unlabelled-model-only-jump-all-$seed/1 --load_model --vision_encoder_path ./../logs/create-unlabelled-encoder-all-$seed/1encoder.pt --only_jump --seed $seed --data_dir $DATA_DIR
done

for seed in 123 231 312
do
    CUDA_VISIBLE_DEVICES=3 taskset -c 100-110 python run_dt_atari.py --context_length 20 --top_k 2 --ckpt_path ./../logs-final/load-unlabelled-model-only-left-right-all-$seed/1 --load_model --vision_encoder_path ./../logs/create-unlabelled-encoder-all-$seed/1encoder.pt --only_left_right --seed $seed --data_dir $DATA_DIR
done

for seed in 123 231 312
do
    CUDA_VISIBLE_DEVICES=3 taskset -c 100-110 python run_dt_atari.py --context_length 20 --top_k 2 --ckpt_path ./../logs-final/load-unlabelled-model-only-dont-move-all-$seed/1 --load_model --vision_encoder_path ./../logs/create-unlabelled-encoder-all-$seed/1encoder.pt --only_dont_move --seed $seed --data_dir $DATA_DIR
done

for seed in 123 231 312
do
    CUDA_VISIBLE_DEVICES=3 taskset -c 100-110 python run_dt_atari.py --context_length 20 --top_k 2 --ckpt_path ./../logs-final/load-unlabelled-model-only-certain-ice-floe-all-$seed/1 --load_model --vision_encoder_path ./../logs/create-unlabelled-encoder-all-$seed/1encoder.pt --only_certain_ice_floe --seed $seed --data_dir $DATA_DIR
done

for seed in 123 231 312
do
    CUDA_VISIBLE_DEVICES=3 taskset -c 100-110 python run_dt_atari.py --context_length 20 --top_k 2 --ckpt_path ./../logs-final/load-unlabelled-model-only-die-level-all-$seed/1 --load_model --vision_encoder_path ./../logs/create-unlabelled-encoder-all-$seed/1encoder.pt --only_die_level --seed $seed --data_dir $DATA_DIR
done
