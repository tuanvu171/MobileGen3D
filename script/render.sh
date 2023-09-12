scene=$1
ckpt=$2

python3 -m torch.distributed.launch --nproc_per_node=1 --use_env  main.py \
    --project_name $scene \
    --dataset_type Colmap \
    --pseudo_dir model/teacher/ngp_pl/Pseudo/$scene \
    --root_dir dataset \
    --run_render \
    --input_height 64 \
    --input_width 64 \
    --output_height 512 \
    --output_width 512 \
    --scene $scene \
    --ff \
    --ndc \
    --amp \
    --ckpt_dir $ckpt

