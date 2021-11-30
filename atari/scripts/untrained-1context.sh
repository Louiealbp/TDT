# Training models without pre-training using 1 context
for seed in 123 231 312
do
    CUDA_VISIBLE_DEVICES=0 taskset -c 110-120 python run_dt_atari.py --context_length 1 --seed $seed --data_dir $DATA_DIR --top_k 2 --ckpt_path ./../logs-1context/untrained-encoder-reach-level-$seed/1 --only_reach_level --skip_eval 2 --epochs 10
done

for seed in 123 231 312
do
    CUDA_VISIBLE_DEVICES=0 taskset -c 110-120 python run_dt_atari.py --context_length 1 --seed $seed --data_dir $DATA_DIR --top_k 2 --ckpt_path ./../logs-1context/untrained-encoder-max-points-$seed/1 --only_max_points --skip_eval 2 --epochs 10
done

for seed in 123 231 312
do
    CUDA_VISIBLE_DEVICES=0 taskset -c 110-120 python run_dt_atari.py --context_length 1 --seed $seed --data_dir $DATA_DIR --top_k 2 --ckpt_path ./../logs-1context/untrained-encoder-certain-points-$seed/1 --only_certain_points --skip_eval 2 --epochs 10
done

for seed in 123 231 312
do
    CUDA_VISIBLE_DEVICES=0 taskset -c 110-120 python run_dt_atari.py --context_length 1 --seed $seed --data_dir $DATA_DIR --top_k 2 --ckpt_path ./../logs-1context/untrained-encoder-only-jump-$seed/1 --only_jump --skip_eval 2 --epochs 10
done

for seed in 123 231 312
do
    CUDA_VISIBLE_DEVICES=0 taskset -c 110-120 python run_dt_atari.py --context_length 1 --seed $seed --data_dir $DATA_DIR --top_k 2 --ckpt_path ./../logs-1context/untrained-encoder-only-left-right-$seed/1 --only_left_right --skip_eval 2 --epochs 10
done

for seed in 123 231 312
do
    CUDA_VISIBLE_DEVICES=0 taskset -c 110-120 python run_dt_atari.py --context_length 1 --seed $seed --data_dir $DATA_DIR --top_k 2 --ckpt_path ./../logs-1context/untrained-encoder-only-dont-move-$seed/1 --only_dont_move --skip_eval 2 --epochs 10
done

for seed in 123 231 312
do
    CUDA_VISIBLE_DEVICES=0 taskset -c 110-120 python run_dt_atari.py --context_length 1 --seed $seed --data_dir $DATA_DIR --top_k 2 --ckpt_path ./../logs-1context/untrained-encoder-only-certain-ice-floe-$seed/1 --only_certain_ice_floe --skip_eval 2 --epochs 10
done

for seed in 123 231 312
do
    CUDA_VISIBLE_DEVICES=0 taskset -c 110-120 python run_dt_atari.py --context_length 1 --seed $seed --data_dir $DATA_DIR --top_k 2 --ckpt_path ./../logs-1context/untrained-encoder-only-die-level-$seed/1 --only_die_level --skip_eval 2 --epochs 10
done
