#!/bin/bash

#
# This uses what we learned about which models are pareto-optimially best (i.e., most
# accurate) and fastest (tl;dr: roberta-base). This script is an attempt to consolidate
# a complete set of classifier baselines in one place.
#

# cols as input to prediction:
#                      +-----[ C = multi-class, ML = multi-label, MC = multiple-choice ]
#                      |
#                      v
# sit rot act jmt chr qst attr
# --- --- --- --- --- --- ---
#  x   x              C   rot-agree
#  x   x              ML  rot-moral-foundations
#  x   x              ML  rot-categorization
#  x   x           A  MC  rot-char-targeting
#  x       x          C   action-agency
#      x   x          C   action-moral-judgment (transcribe)
#  x       x          C   action-moral-judgment (rate)
#          x   x      C   action-agree
#  x       x          C   action-legal
#  x       x          C   action-pressure
#  x       x       A  MC  action-char-involved
#  x       x       1  C   action-hypothetical --------------------- NOTE: no C (for now)


# C: multi-class
# ---

# rot-agree
python -m sc.model.classify \
    --model_type roberta \
    --model_name_or_path roberta-base \
    --attr rot-agree \
    --input_situation \
    --input_rot \
    --do_train \
    --train_metrics \
    --do_eval \
    --evaluate_during_training \
    --evaluate_at_start \
    --output_dir checkpoints/rot-agree/roberta-base/

# action-agency
python -m sc.model.classify \
    --model_type roberta \
    --model_name_or_path roberta-base \
    --attr action-agency \
    --input_situation \
    --input_action \
    --do_train \
    --train_metrics \
    --do_eval \
    --evaluate_during_training \
    --evaluate_at_start \
    --output_dir checkpoints/action-agency/roberta-base/

# action-moral-judgment - transcribe
python -m sc.model.classify \
    --model_type roberta \
    --model_name_or_path roberta-base \
    --attr action-moral-judgment \
    --input_rot \
    --input_action \
    --do_train \
    --train_metrics \
    --do_eval \
    --evaluate_during_training \
    --evaluate_at_start \
    --output_dir checkpoints/action-moral-judgment-transcribe/roberta-base/

# action-moral-judgment - rate
python -m sc.model.classify \
    --model_type roberta \
    --model_name_or_path roberta-base \
    --attr action-moral-judgment \
    --input_situation \
    --input_action \
    --do_train \
    --train_metrics \
    --do_eval \
    --evaluate_during_training \
    --evaluate_at_start \
    --output_dir checkpoints/action-moral-judgment-rate/roberta-base/

# action-legal
python -m sc.model.classify \
    --model_type roberta \
    --model_name_or_path roberta-base \
    --attr action-legal \
    --input_situation \
    --input_action \
    --do_train \
    --train_metrics \
    --do_eval \
    --evaluate_during_training \
    --evaluate_at_start \
    --output_dir checkpoints/action-legal/roberta-base/

# action-pressure
python -m sc.model.classify \
    --model_type roberta \
    --model_name_or_path roberta-base \
    --attr action-pressure \
    --input_situation \
    --input_action \
    --do_train \
    --train_metrics \
    --do_eval \
    --evaluate_during_training \
    --evaluate_at_start \
    --output_dir checkpoints/action-pressure/roberta-base/

# action-hypothetical
# situation + action (+ char -- future)
python -m sc.model.classify \
    --model_type roberta \
    --model_name_or_path roberta-base \
    --attr action-hypothetical \
    --input_situation \
    --input_action \
    --do_train \
    --train_metrics \
    --do_eval \
    --evaluate_during_training \
    --evaluate_at_start \
    --output_dir checkpoints/action-hypothetical/roberta-base/

# action-agree
python -m sc.model.classify \
    --model_type roberta \
    --model_name_or_path roberta-base \
    --attr action-agree \
    --input_action \
    --input_judgment \
    --do_train \
    --train_metrics \
    --do_eval \
    --evaluate_during_training \
    --evaluate_at_start \
    --output_dir checkpoints/action-agree/roberta-base/

# ML: multi-label
# ---

# rot-moral-foundations
python -m sc.model.classify \
    --model_type roberta \
    --model_name_or_path roberta-base \
    --attr rot-moral-foundations \
    --input_situation \
    --input_rot \
    --do_train \
    --train_metrics \
    --do_eval \
    --evaluate_during_training \
    --evaluate_at_start \
    --output_dir checkpoints/rot-moral-foundations/roberta-base/

# rot-categorization
python -m sc.model.classify \
    --model_type roberta \
    --model_name_or_path roberta-base \
    --attr rot-categorization \
    --input_situation \
    --input_rot \
    --do_train \
    --train_metrics \
    --do_eval \
    --evaluate_during_training \
    --evaluate_at_start \
    --output_dir checkpoints/rot-categorization/roberta-base/


# MC: multiple-choice
# ---

# rot-char-targeting
# situation + rot [+ chars]
python -m sc.model.classify \
    --model_type roberta \
    --model_name_or_path roberta-base \
    --attr rot-char-targeting \
    --input_situation \
    --input_rot \
    --do_train \
    --train_metrics \
    --do_eval \
    --evaluate_during_training \
    --evaluate_at_start \
    --output_dir checkpoints/rot-char-targeting/roberta-base/

# action-char-involved
# situation + action [+ chars]
python -m sc.model.classify \
    --model_type roberta \
    --model_name_or_path roberta-base \
    --attr action-char-involved \
    --input_situation \
    --input_action \
    --do_train \
    --train_metrics \
    --do_eval \
    --evaluate_during_training \
    --evaluate_at_start \
    --output_dir checkpoints/action-char-involved/roberta-base/
