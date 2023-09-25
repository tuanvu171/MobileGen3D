nGPU=$1
scene=$2

python3 -m torch.distributed.launch --nproc_per_node=$nGPU --use_env  main.py \
    --project_name $scene \
    --dataset_type Colmap \
    --pseudo_dir model/teacher/ngp_pl/Pseudo/$scene \
    --root_dir dataset/nerf_data \
    --run_train \
    --num_workers 12 \
    --batch_size 10 \
    --num_iters 600000 \
    --input_height 64 \
    --input_width 64 \
    --output_height 512 \
    --output_width 512 \
    --scene $scene \
    --ff \
    --ndc \
    --amp \
    --i_testset 1000 \
    --lrate 0.0005
