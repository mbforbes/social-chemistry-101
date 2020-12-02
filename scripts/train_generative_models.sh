#!/usr/bin/env bash

device=8
declare -a items=(action rot)
declare -a lms=(openai-gpt gpt2 gpt2-xl bart-large t5-large random_gpt2-xl)

for lm in "${lms[@]}"
do
    if [[ $lm == *t5* ]]; then
        script="encoder_decoder"
    elif [[ $lm == *bart* ]]; then
        script="encoder_decoder"
    else
        script="generative"
    fi

    # Find available device
    while [ $device -gt 7 ]
    do
        for ((i=0;i<=7;i++));
        do
            info=`nvidia-smi -i ${i}`
            if [[ $info == *"No running processes found"* ]]; then
                device=$i
                echo "Using device ${device}"
                break
            fi
        done

        if [[ $device -gt 7 ]]; then
            sleep 30s
        fi
    done

    curr_device=${device};
    device=8;

    for item in "${items[@]}"
    do
        python -m sc.model.${script} \
            --out_dir output/${item}_all_${lm} \
            --model_name_or_path ${lm} \
            --device ${curr_device} \
            --do_train \
            --do_eval \
            --eval_during_train \
            --value_to_predict ${item} \
            --input_type all \
            --output_type all \
            --num_train_epochs 5 \
            --gradient_accumulation_steps 8 \
            --train_batch_size 8 \
            --eval_batch_size 8 \
            --overwrite_cache &
    done

    sleep 60s
done
