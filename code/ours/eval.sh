#trte_spt: 'label', 'shuffle'
#tr_qry: 'label', 'shuffle', 'cluster'
#te_qry: 'label', 'shuffle', 'cluster
#--load 'tiered-5w5s-1600-uplr0.01-upst1-st60-dr0-wd0-G2-2' \

CUDA_CACHE_PATH=/st1/hayeon/tmp python test.py \
    --load 'tiered-5w5s-1600-uplr0.01-upst1-st60-dr0-wd0.5-G5-1' \
    --ckpt 'epoch-70.92.pth' \
    --gpu 0 \
    --fname "db" \
    --test_batch 600 \

