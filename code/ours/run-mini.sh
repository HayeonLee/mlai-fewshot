#trte_spt: 'label', 'shuffle'
#tr_qry: 'label', 'shuffle', 'cluster'
#te_qry: 'label', 'shuffle', 'cluster'

CUDA_CACHE_PATH=/st1/hayeon/tmp python main.py \
    --data 'mini' \
    --shot 5 \
    --way 5 \
    --gpu 2 \
    --model 'ours' \
    --test_bn 'True' \
    --test_epi 2000 \
    --test_batch 600 \
    --speed 'False' \
    --max_episode 100000 \
    --save_episode 10000 \
    --fname 'reproduce-mini-5w5s-G2-1' \
    --hdim 64 \
    --zdim 64 \
    --dim '1600' \
    --update_lr 0.01 \
    --update_step 1 \
    --task_num 5 \
    --step_size 60 \
    --fc 'True' \
    --dropout_rate '0.5' \
    --w_decay 0.9


