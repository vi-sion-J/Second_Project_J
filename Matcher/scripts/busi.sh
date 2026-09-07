#!/bin/bash

# 1. 튕김 방지용 경로 설정 (WSL 및 가상환경 cuDNN 경로)
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/junha/miniconda3/envs/sam_env/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

# 2. BUSI는 Fold가 0 하나뿐입니다.
for fold in 0;
do
  CUDA_VISIBLE_DEVICES=0 \
  python main_oss.py  \
    --benchmark busi \
    --nshot 1 \
    --max_sample_iterations 64 \
    --box_nms_thresh 0.65 \
    --sample-range "(1,6)" \
    --topk_scores_threshold 0.0 \
    --use_dense_mask 1 \
    --use_points_or_centers \
    --purity_filter 0.02 \
    --iou_filter 0.85 \
    --multimask_output 1 \
    --sel_stability_score_thresh 0.90 \
    --use_score_filter \
    --alpha 1.0 --beta 0. --exp 0. \
    --num_merging_mask 9  \
    --fold ${fold} \
    --log-root "output/busi/fold${fold}" \
    --visualize = 1 # 3. 평가가 끝난 후 정답/예측 이미지를 눈으로 비교하기 위해 추가
done
wait