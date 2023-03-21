# rationale generation
CUDA_VISIBLE_DEVICES=0,1 python main.py \
    --model allenai/unifiedqa-t5-base \
    --user_msg rationale --img_type detr \
    --bs 2 --eval_bs 2 --eval_acc 5 --output_len 256 \
    --final_eval --prompt_format QCM-LE \
    --evaluate_dir models/MM-CoT-UnifiedQA-base-Rationale
