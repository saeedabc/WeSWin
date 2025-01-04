#!/bin/bash

###############
### General ###
###############

export TMP_HOME=..
OUT_ROOT=${TMP_HOME}/checkpoints

if [ ! -v HF_HOME ]; then
    export HF_HOME=${TMP_HOME}/.cache/huggingface
fi

: ${SEED:=42}
LOG_LEVEL=warning  # info
: ${CUDA_VISIBLE_DEVICES:=""}

: ${TASK_NAME:="multi_seg"}

: ${OUT_RESULTS:="results"}
_TMP="--out_results_path ${TMP_HOME}/out/${DATASET_NAME}/${OUT_RESULTS}.csv"
if [ -v OTHER_DATASET_NAME ]; then
    _TMP="${_TMP} --other_out_results_path ${TMP_HOME}/out/${OTHER_DATASET_NAME}/${OUT_RESULTS}.csv"
fi
OUT_RESULTS=${_TMP}
if [[ ! -z ${EXPR_NAME} ]]; then
    OUT_RESULTS="${OUT_RESULTS} --expr_name ${EXPR_NAME}"
fi
if [[ ! -z ${EXPR_THRESHOLD} ]]; then
    OUT_RESULTS="${OUT_RESULTS} --expr_threshold ${EXPR_THRESHOLD}"
fi
if [[ ! -z ${EXPR_HPM_PATH} ]]; then
    OUT_RESULTS="${OUT_RESULTS} --expr_hpm_path ${EXPR_HPM_PATH}"
fi
: ${EXPR_METRIC:="f1"}
OUT_RESULTS="${OUT_RESULTS} --expr_metric ${EXPR_METRIC}"

echo -e "OUT_RESULTS: ${OUT_RESULTS}"


###############
### Dataset ###
###############

### Data & Tokenization ###

: ${SENT_SEP:="snt"}  # "sep"
: ${ADD_PREFIX_SPACE:=""}  # "--add_prefix_space"

: ${PARTITION_STRATEGY:="ex"}
: ${PARTITION_VALUE:=0}
: ${PARTITION_DISABLE_STRATEGY:="na"}
: ${PARTITION_DISABLE_VALUE:=0}

DO_COMMON="--doc_id_column_name id --id_column_name ids --text_column_name sentences --label_column_name labels \
            ${ADD_PREFIX_SPACE} --sent_sep ${SENT_SEP} \
            --partition_strategy ${PARTITION_STRATEGY} --partition_value ${PARTITION_VALUE} \
            --partition_disable_value ${PARTITION_DISABLE_VALUE} --partition_disable_strategy ${PARTITION_DISABLE_STRATEGY}"

### Dataset ###
_SENT_TOKENIZER="nltk"

[ -v DATASET_NAME ] || { echo "DATASET_NAME is not set!"; exit 1; }

if [[ ${DATASET_NAME} = "wiki727k" ]]; then
    DATASET_ROOT=${TMP_HOME}/datasets/wiki727k/wiki_727
    : ${MAX_TRAIN_SAMPLES:=582160}
    : ${MAX_EVAL_SAMPLES:=1000}
    : ${MAX_PREDICT_SAMPLES:=1000}
    DATASET_TRAIN_PATH=${DATASET_ROOT}/train-wt.jsonl
    DATASET_EVAL_PATH=${DATASET_ROOT}/eval-wt.jsonl
    DATASET_PREDICT_PATH=${DATASET_ROOT}/predict-wt.jsonl
    : ${DROP_TITLES_PROBABILITY:=1}

elif [[ ${DATASET_NAME} = "wiki50" ]]; then
    DATASET_ROOT=${TMP_HOME}/datasets/wiki727k/wiki_test_50
    : ${MAX_PREDICT_SAMPLES:=50}
    DATASET_PREDICT_PATH=${DATASET_ROOT}/predict-wt.jsonl
    : ${DROP_TITLES_PROBABILITY:=1}

elif [[ ${DATASET_NAME} = "en_city" ]]; then
    DATASET_ROOT=${TMP_HOME}/datasets/wikisection/en_city
    : ${MAX_TRAIN_SAMPLES:=13679}
    : ${MAX_EVAL_SAMPLES:=1000}
    : ${MAX_PREDICT_SAMPLES:=1000}
    DATASET_TRAIN_PATH=${DATASET_ROOT}/train-wt-${_SENT_TOKENIZER}.jsonl
    DATASET_EVAL_PATH=${DATASET_ROOT}/eval-wt-${_SENT_TOKENIZER}.jsonl
    DATASET_PREDICT_PATH=${DATASET_ROOT}/predict-wt-${_SENT_TOKENIZER}.jsonl
    : ${DROP_TITLES_PROBABILITY:=1}

elif [[ ${DATASET_NAME} = "en_disease" ]]; then
    DATASET_ROOT=${TMP_HOME}/datasets/wikisection/en_disease
    : ${MAX_TRAIN_SAMPLES:=2513}
    : ${MAX_EVAL_SAMPLES:=1000}
    : ${MAX_PREDICT_SAMPLES:=1000}
    DATASET_TRAIN_PATH=${DATASET_ROOT}/train-wt-${_SENT_TOKENIZER}.jsonl
    DATASET_EVAL_PATH=${DATASET_ROOT}/eval-wt-${_SENT_TOKENIZER}.jsonl
    DATASET_PREDICT_PATH=${DATASET_ROOT}/predict-wt-${_SENT_TOKENIZER}.jsonl
    : ${DROP_TITLES_PROBABILITY:=1}

else 
    echo "Invalid DATASET_NAME: ${DATASET_NAME}"
    exit 1

fi

: ${NUM_WORKERS:=20}
: ${OVERWRITE_CACHE:=""}  # "--overwrite_cache"
DO_COMMON="${DO_COMMON} --dataset_log_name ${DATASET_NAME} --drop_titles_probability ${DROP_TITLES_PROBABILITY} ${OVERWRITE_CACHE} --preprocessing_num_workers ${NUM_WORKERS}"

if [ -v DATASET_TRAIN_PATH ]; then
    DO_TRAIN="--do_train --train_file ${DATASET_TRAIN_PATH}"
    if [ -v MAX_TRAIN_SAMPLES ]; then
        DO_TRAIN="${DO_TRAIN} --max_train_samples ${MAX_TRAIN_SAMPLES}"
    fi
fi

if [ -v DATASET_EVAL_PATH ]; then
    DO_EVAL="--do_eval --eval_file ${DATASET_EVAL_PATH}"
    if [ -v MAX_EVAL_SAMPLES ]; then
        DO_EVAL="${DO_EVAL} --max_eval_samples ${MAX_EVAL_SAMPLES}"
    fi
fi

if [ -v DATASET_PREDICT_PATH ]; then
    DO_PREDICT="--do_predict --predict_file ${DATASET_PREDICT_PATH}"
    if [ -v MAX_PREDICT_SAMPLES ]; then
        DO_PREDICT="${DO_PREDICT} --max_predict_samples ${MAX_PREDICT_SAMPLES}"
    fi
fi

echo -e "Dataset: ${DATASET_NAME}"

##### Other dataset #####

if [ -v OTHER_DATASET_NAME ]; then
    if [[ ${OTHER_DATASET_NAME} = "wiki727k" ]]; then
        OTHER_DATASET_ROOT=${TMP_HOME}/datasets/wiki727k/wiki_727
        : ${OTHER_MAX_TRAIN_SAMPLES:="$((${MAX_TRAIN_SAMPLES} / 2))"}
        # : ${OTHER_MAX_TRAIN_SAMPLES:="${MAX_TRAIN_SAMPLES}"}
        : ${OTHER_MAX_EVAL_SAMPLES:=1000}
        : ${OTHER_MAX_PREDICT_SAMPLES:=1000}
        OTHER_DATASET_TRAIN_PATH=${OTHER_DATASET_ROOT}/train-wt.jsonl
        OTHER_DATASET_EVAL_PATH=${OTHER_DATASET_ROOT}/eval-wt.jsonl
        OTHER_DATASET_PREDICT_PATH=${OTHER_DATASET_ROOT}/predict-wt.jsonl
        : ${OTHER_DROP_TITLES_PROBABILITY:=1}
    
    else
        echo "Invalid OTHER_DATASET_NAME: ${OTHER_DATASET_NAME}"
        exit 1
    fi

    : ${OTHER_NUM_WORKERS:="${NUM_WORKERS}"}
    : ${OTHER_OVERWRITE_CACHE:="${OVERWRITE_CACHE}"}
    DO_COMMON="${DO_COMMON} --other_dataset_log_name ${OTHER_DATASET_NAME} --other_drop_titles_probability ${OTHER_DROP_TITLES_PROBABILITY} ${OTHER_OVERWRITE_CACHE} --other_preprocessing_num_workers ${OTHER_NUM_WORKERS}"

    DO_TRAIN="${DO_TRAIN} --other_train_file ${OTHER_DATASET_TRAIN_PATH} --other_max_train_samples ${OTHER_MAX_TRAIN_SAMPLES}"
    DO_EVAL="${DO_EVAL} --other_eval_file ${OTHER_DATASET_EVAL_PATH} --other_max_eval_samples ${OTHER_MAX_EVAL_SAMPLES}"
    DO_PREDICT="${DO_PREDICT} --other_predict_file ${OTHER_DATASET_PREDICT_PATH} --other_max_predict_samples ${OTHER_MAX_PREDICT_SAMPLES}"

    echo -e "Other dataset: ${OTHER_DATASET_NAME} ..."
fi


#############
### Model ###
#############

[ -v RUN_CONFIG ] || { echo "RUN_CONFIG is not set!"; exit 1; }

if [ ${RUN_CONFIG} = "train_scratch" ] || [ ${RUN_CONFIG} = "train_resume" ]; then
    echo "Training..."

    : ${NUM_LOGITS:=2}
    : ${FROZENS:=""}  # FROZENS="bert", FROZENS="!classifier"

    [ -v MODEL_NAME ] || { echo "MODEL_NAME is not set!"; exit 1; }
    # : ${MODEL_NAME:="roberta-base"}
    : ${MAX_TOKENS_PER_SEQ:=512}
    : ${ATTENTION_WINDOW:=512}
    
    : ${BS:=8}
    : ${GAC:=1}
    : ${EPOCHS:=3}

    : ${LR:=1e-5}
    : ${LR_SCHEDULER:="constant"}
    : ${WARMUP_RATIO:=0}
    # : ${LR_SCHEDULER:="linear"}
    # : ${LR_SCHEDULER:="constant_with_warmup"}
    # : ${WARMUP_RATIO:=0.005}

    : ${BEST_METRIC:="f1"}
    if [ ${BEST_METRIC} = "loss" ]; then
        GREATER_IS_BETTER="False"
    else
        GREATER_IS_BETTER="True"
    fi
    : ${SAVE_TOTAL_LIMIT:=2}

    : ${EARLY_STOPPING_PATIENCE:=5}
    : ${EARLY_STOPPING_THRESHOLD:=-0.25}

    : ${EVAL_STEPS:=10000}
    SAVE_STEPS=${EVAL_STEPS}
    LOGGING_STEPS=$((EVAL_STEPS / 10))
    EVALUATION_STRATEGY="steps"
    LOGGING_STRATEGY="steps"

    if [ ${RUN_CONFIG} = "train_scratch" ]; then
        RUN_NAME=${MODEL_NAME}-${TASK_NAME}-${MAX_TOKENS_PER_SEQ}
        if [[ "${MODEL_NAME}" =~ .*longformer.* ]]; then
            RUN_NAME=${RUN_NAME}-${ATTENTION_WINDOW}
        fi
        if [ ! -z ${FROZENS} ]; then
            RUN_NAME=${RUN_NAME}-${FROZENS}
        fi
        RUN_NAME=${RUN_NAME}-${DATASET_NAME}-$((MAX_TRAIN_SAMPLES / 1000))k-dtp${DROP_TITLES_PROBABILITY}
        if [ -v OTHER_DATASET_NAME ]; then
            RUN_NAME=${RUN_NAME}-${OTHER_DATASET_NAME}-$((OTHER_MAX_TRAIN_SAMPLES / 1000))k-dtp${OTHER_DROP_TITLES_PROBABILITY}
        fi
        RUN_NAME=${RUN_NAME}-${PARTITION_STRATEGY}${PARTITION_VALUE}-${PARTITION_DISABLE_STRATEGY}${PARTITION_DISABLE_VALUE}-${SENT_SEP}
        RUN_NAME=${RUN_NAME}-lr${LR}-${LR_SCHEDULER}-bs${BS}-seed${SEED}
        RUN_NAME=${RUN_NAME}-$(date +%m-%d)

        if [ ${#RUN_NAME} -gt 255 ]; then
            RUN_NAME="${RUN_NAME:0:255}"
            echo "Truncated too long RUN_NAME: $RUN_NAME"
        fi

        if [ ! -d ${MODEL_NAME} ]; then 
            RUN_NAME=${OUT_ROOT}/${RUN_NAME}
        fi

        OVERWRITE_OUTPUT_DIR="--overwrite_output_dir"

    elif [ ${RUN_CONFIG} = "train_resume" ]; then
        [ -d ${MODEL_NAME} ] || { echo "MODEL_NAME directory does not exist!"; exit 1; }
        RUN_NAME=${MODEL_NAME}

        OVERWRITE_OUTPUT_DIR=""

        if [[ ! -z "${RESUME_FROM_CHECKPOINT}" ]]; then
            RESUME_FROM_CHECKPOINT="--resume_from_checkpoint ${RESUME_FROM_CHECKPOINT}"
        fi
    fi

    OUTPUT_DIR=${RUN_NAME}
    LOGGING_DIR=${OUTPUT_DIR}/runs

    echo -e "${RUN_CONFIG}: \n\t${MODEL_NAME} --> \n\t${OUTPUT_DIR}"

    python run_weswin.py \
        ${DO_COMMON} \
        ${DO_TRAIN} ${DO_EVAL} ${DO_PREDICT} \
        --frozens "${FROZENS}" --task_name ${TASK_NAME} --num_logits ${NUM_LOGITS} \
        --max_tokens_per_seq ${MAX_TOKENS_PER_SEQ} --attention_window "${ATTENTION_WINDOW}" \
        --model_name_or_path ${MODEL_NAME} \
        --output_dir ${OUTPUT_DIR} ${OVERWRITE_OUTPUT_DIR} ${RESUME_FROM_CHECKPOINT} \
        --logging_dir ${LOGGING_DIR} --logging_strategy ${LOGGING_STRATEGY} --logging_steps ${LOGGING_STEPS} \
        --log_level ${LOG_LEVEL} \
        --report_to tensorboard \
        --evaluation_strategy ${EVALUATION_STRATEGY} --eval_steps ${EVAL_STEPS} \
        --early_stopping_patience ${EARLY_STOPPING_PATIENCE} --early_stopping_threshold ${EARLY_STOPPING_THRESHOLD} \
        --save_steps ${SAVE_STEPS} --load_best_model_at_end --metric_for_best_model ${BEST_METRIC} --greater_is_better ${GREATER_IS_BETTER} --save_total_limit ${SAVE_TOTAL_LIMIT} \
        --num_train_epochs ${EPOCHS} \
        --learning_rate ${LR} \
        --per_device_train_batch_size ${BS} \
        --per_device_eval_batch_size ${BS} \
        --lr_scheduler_type ${LR_SCHEDULER} \
        --warmup_ratio ${WARMUP_RATIO} \
        --gradient_accumulation_steps ${GAC} \
        --seed ${SEED} \
        ${OUT_RESULTS}


elif [ ${RUN_CONFIG} = "predict" ]; then
    echo "Inference..."

    : ${BS:=16}

    [ -v MODEL_NAME ] || { echo "MODEL_NAME is not set!"; exit 1; }

    OUTPUT_DIR=${MODEL_NAME}
    LOGGING_DIR=${OUTPUT_DIR}/runs

    echo -e "${RUN_CONFIG}: \n\t${MODEL_NAME} --> \n\t${OUTPUT_DIR}"

    python run_weswin.py \
        ${DO_COMMON} \
        ${DO_EVAL} ${DO_PREDICT} \
        --model_name_or_path ${MODEL_NAME} \
        --task_name ${TASK_NAME} \
        --output_dir ${OUTPUT_DIR} \
        --logging_dir ${LOGGING_DIR} \
        --log_level ${LOG_LEVEL} \
        --per_device_eval_batch_size ${BS} \
        --seed ${SEED} \
        ${OUT_RESULTS}

else
    echo "Invalid RUN_CONFIG: ${RUN_CONFIG}"
    exit 1

fi