#!/usr/bin/env bash

device=$1
declare -a items=(action rot)
declare -a lms=(openai-gpt gpt2 gpt2-xl bart-large t5-large random_gpt2-xl)
declare -a splits=(dev test)


for split in "${splits[@]}"
do
    for lm in "${lms[@]}"
    do
        python -m sc.model.generate_texts \
            --in_file data/dataset/social-chemistry.v0.1.tsv \
            --split ${split} \
            --out_file output/generate/action_all_${lm}/${split}_predictions_p0.9.jsonl \
            --model_name_or_path output/action_all_${lm} \
            --p 0.9 \
            --device ${device} \
            --value_to_predict action \
            --input_type all \
            --output_type all &

         python -m sc.model.generate_texts \
            --in_file data/dataset/social-chemistry.v0.1.tsv \
            --split ${split} \
            --out_file output/generate/rot_all_${lm}/${split}_predictions_p0.9.jsonl \
            --model_name_or_path output/rot_all_${lm} \
            --p 0.9 \
            --device ${device} \
            --value_to_predict rot \
            --input_type all \
            --output_type all &

          wait
      done
done
