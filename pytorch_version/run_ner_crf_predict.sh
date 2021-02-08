CURRENT_DIR=`pwd`
#export BERT_BASE_DIR=$CURRENT_DIR/prev_trained_model/roberta_wwm_large_ext
export BERT_BASE_DIR=$CURRENT_DIR/prev_trained_model/bert-base
export GLUE_DIR=$CURRENT_DIR/datasets
export OUTPUR_DIR=$CURRENT_DIR/outputs
TASK_NAME="cluener"

python run_ner_crf.py \
  --model_type=bert \
  --model_name_or_path=$BERT_BASE_DIR \
  --task_name=$TASK_NAME \
  --do_predict \
  --do_lower_case \
  --data_dir=$GLUE_DIR/${TASK_NAME}/ \
  --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
  --seed=42