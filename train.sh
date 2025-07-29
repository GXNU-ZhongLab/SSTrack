CUDA_VISIBLE_DEVICES=2,3 python tracking/train.py \
--script sstrack --config dropmae_256_150ep \
--save_dir ./output \
--mode multiple --nproc_per_node 2 \
--use_wandb 0


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python tracking/test.py \
onemamba baseline_256_ndtetoken_rgbtoken_taskproduct_simba_resgate_notFFN \
--runid 14 --dataset_name visevent --threads 24 --num_gpus 8
