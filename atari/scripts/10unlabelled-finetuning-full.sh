# Script to generate data for transferring full model with pre-training on 10 missions
for seed in 123 231 312
do
    CUDA_VISIBLE_DEVICES=2 python run_dt_atari.py --context_length 20 --top_k 2 --ckpt_path ./../logs/create-unlabelled-encoder-10-$seed/1 --unlabelled_extra_data --num_unlabelled_missions 10 --no_labelled_missions --data_dir $DATA_DIR --seed $SEED
done

for seed in 123 231 312
do
    CUDA_VISIBLE_DEVICES=5 taskset -c 140-160 python run_dt_atari.py --context_length 20 --top_k 2 --ckpt_path ./../logs-final/load-unlabelled-model-reach-level-10-$seed/1 --load_model --vision_encoder_path ./../logs/create-unlabelled-encoder-10-$seed/1encoder.pt --only_reach_level --seed $seed --data_dir $DATA_DIR
done

for seed in 123 231 312
do
    CUDA_VISIBLE_DEVICES=5 taskset -c 140-160 python run_dt_atari.py --context_length 20 --top_k 2 --ckpt_path ./../logs-final/load-unlabelled-model-max-points-10-$seed/1 --load_model --vision_encoder_path ./../logs/create-unlabelled-encoder-10-$seed/1encoder.pt --only_max_points --seed $seed --data_dir $DATA_DIR
done

for seed in 123 231 312
do
    CUDA_VISIBLE_DEVICES=5 taskset -c 140-160 python run_dt_atari.py --context_length 20 --top_k 2 --ckpt_path ./../logs-final/load-unlabelled-model-certain-points-10-$seed/1 --load_model --vision_encoder_path ./../logs/create-unlabelled-encoder-10-$seed/1encoder.pt --only_certain_points --seed $seed --data_dir $DATA_DIR
done

for seed in 123 231 312
do
    CUDA_VISIBLE_DEVICES=5 taskset -c 140-160 python run_dt_atari.py --context_length 20 --top_k 2 --ckpt_path ./../logs-final/load-unlabelled-model-only-jump-10-$seed/1 --load_model --vision_encoder_path ./../logs/create-unlabelled-encoder-10-$seed/1encoder.pt --only_jump --seed $seed --data_dir $DATA_DIR
done

for seed in 123 231 312
do
    CUDA_VISIBLE_DEVICES=5 taskset -c 140-160 python run_dt_atari.py --context_length 20 --top_k 2 --ckpt_path ./../logs-final/load-unlabelled-model-only-left-right-10-$seed/1 --load_model --vision_encoder_path ./../logs/create-unlabelled-encoder-10-$seed/1encoder.pt --only_left_right --seed $seed --data_dir $DATA_DIR
done

for seed in 123 231 312
do
    CUDA_VISIBLE_DEVICES=5 taskset -c 140-160 python run_dt_atari.py --context_length 20 --top_k 2 --ckpt_path ./../logs-final/load-unlabelled-model-only-dont-move-10-$seed/1 --load_model --vision_encoder_path ./../logs/create-unlabelled-encoder-10-$seed/1encoder.pt --only_dont_move --seed $seed --data_dir $DATA_DIR
done

for seed in 123 231 312
do
    CUDA_VISIBLE_DEVICES=5 taskset -c 140-160 python run_dt_atari.py --context_length 20 --top_k 2 --ckpt_path ./../logs-final/load-unlabelled-model-only-certain-ice-floe-10-$seed/1 --load_model --vision_encoder_path ./../logs/create-unlabelled-encoder-10-$seed/1encoder.pt --only_certain_ice_floe --seed $seed --data_dir $DATA_DIR
done

for seed in 123 231 312
do
    CUDA_VISIBLE_DEVICES=5 taskset -c 140-160 python run_dt_atari.py --context_length 20 --top_k 2 --ckpt_path ./../logs-final/load-unlabelled-model-only-die-level-10-$seed/1 --load_model --vision_encoder_path ./../logs/create-unlabelled-encoder-10-$seed/1encoder.pt --only_die_level --seed $seed --data_dir $DATA_DIR
done
