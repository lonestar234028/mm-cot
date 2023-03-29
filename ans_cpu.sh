# answer inference
python main_v1.py \
    --model allenai/unifiedqa-t5-base \
    --user_msg answer --img_type detr \
    --bs 8 --eval_bs 4 --eval_acc 10 --output_len 64 \
    --final_eval --prompt_format QCMG-A \
    --evaluate_dir models/answer \
    --res_file res_v1.json \
    --eval_le models/rationale/predictions_ans_eval.json \
    --test_le models/rationale/predictions_ans_test.json \
