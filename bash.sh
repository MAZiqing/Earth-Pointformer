CUDA_VISIBLE_DEVICES=0

# log=logs/tsno2_minist_2.log

# rm $log

# nohup \
        # python ./scripts/cuboid_transformer/moving_mnist/train_tsnov2_mnist.py --gpus 1 \
        # --cfg ./scripts/cuboid_transformer/moving_mnist/cfg_tsnov2.yaml \
        # --ckpt_name last.ckpt \
        # --save tmp_mnist_tsnov24 > $log &

# log=logs/tsno2_minist_cross2.log

task_name=cuboid_mnist_2
log=logs/${task_name}.log
ckpt=tmp_${task_name}

echo $log
echo $ckpt

rm $log

nohup \
python ./scripts/cuboid_transformer/moving_mnist/train_cuboid_mnist.py --gpus 1 \
--cfg ./scripts/cuboid_transformer/moving_mnist/cfg.yaml \
--ckpt_name last.ckpt \
--save $ckpt > $log &
