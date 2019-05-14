# PROTO: [model: proto, te_BN: True/False, task_num: 1, max_epoch 60000]
# PROTO: [model: maml, te_BN: True, task_num: N, max_epoch 60000, update_step M]

CUDA_CACHE_PATH=/st1/hayeon/tmp python main.py \
    --data 'omni_rot' \
    --shot 1 \
    --way 20 \
    --query 5 \
    --gpu 6 \
    --model 'ours' \
    --test_bn 'True' \
    --test_epi 2000 \
    --test_batch 1000 \
    --speed 'False' \
    --max_episode 150000 \
    --save_episode 10000 \
    --fname 'new_omni_rot-20w1s-ul0.01-us1-tn10-fcT-st20-G9-6' \
    --xdim 1 \
    --hdim 64 \
    --zdim 64 \
    --dim 64 \
    --size 28 \
    --update_lr 0.01 \
    --update_step 1 \
    --task_num 10 \
    --fc 'True' \
    --fc_dim 64 \
    --dropout_rate 0 \
    --w_decay 0 \
    --step_size 20


