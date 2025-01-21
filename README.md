# WeSWin Chunker
This is the official codebase for the WeSWin Chunker, introduced in the paper **"[Neural Document Segmentation Using Weighted Sliding Windows with Transformer Encoders](https://aclanthology.org/2025.coling-industry.67/)"** (COLING 2025: Industry Track). We introduce a novel Transformer-based method for document segmentation, tailored for practical, real-world applications. This method utilizes overlapping text sequences with a unique position-aware weighting mechanism to enhance text segmentation accuracy.

---

## Setup Requirements

To setup the environment and install requirements:

```bash
cd code/

conda create -n wesenv python=3.9
conda activate wesenv

pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt')"
```

---

## Preprocess Raw Data

Preprocess the raw data into train, validation, and test JSONL splits:

### Wiki-727K

```bash
bash setup_wiki727k.sh
```

### WikiSection

```bash
bash setup_wikisection.sh
```

---

## Training

Run the following script to fine-tune a RoBERTa-based checkpoint on the Wiki-727K dataset: 

```bash
RUN_CONFIG="train_scratch" MODEL_NAME="roberta-base" \
DATASET_NAME="wiki727k" MAX_TRAIN_SAMPLES=100000 MAX_EVAL_SAMPLES=1000 MAX_PREDICT_SAMPLES=1000 \
PARTITION_DISABLE_STRATEGY="cr" PARTITION_DISABLE_VALUE=1 PARTITION_STRATEGY="ex" PARTITION_VALUE=0 \
SEED=42 CUDA_VISIBLE_DEVICES="0" BS=8 GAC=1 LR=1e-5 EPOCHS=3 EVAL_STEPS=10000 \
./run_weswin_api.sh
```

- `MODEL_NAME`: Specify any (remote or local) BERT, RoBERTa, or Longformer checkpoints, such as `bert-base-cased`, `roberta-base`, or `allenai/longformer-base-4096`.
  - Adjust sequence and attention window size with `MAX_TOKENS_PER_SEQ` and `ATTENTION_WINDOW` for Longformer models.
- `DATASET_NAME`: Set to `wiki727k`, `en_city`, or `en_disease`.
  - Configure the number of documents per split using `MAX_{TRAIN,EVAL,PREDICT}_SAMPLES`.
  - Use `OVERWRITE_CACHE=--overwrite_cache` for fresh tokenization.
- The sliding-window partitioning strategy is set to `cr-1` for training.
- Control early stopping using `EARLY_STOPPING_{PATIENCE,THRESHOLD}`.
- Fine-tuned checkpoints will be saved to `checkpoints/<checkpoint>`.

Refer to `run_weswin_api.sh` and `run_weswin.py` for additional arguments.

---

## Inference and Hyper-parameter Tuning

Run the inference script with various partitioning strategies (e.g., `cr-1`, `ss-6`, `ss-4`, `ss-2`) to optimize hyper-parameters on the validation set and predict on the test set.

```bash
partitions=("ex-0" "ss-6" "ss-4" "ss-2")
for partition in "${partitions[@]}"; do
    echo -e "\n---> Loop run for ${partition} <---\n"

    strategy=$(echo "$partition" | cut -d '-' -f 1)
    value=$(echo "$partition" | cut -d '-' -f 2)

    if [[ $strategy == "ex" ]]; then
        expr_name="t"
    else
        expr_name="tw"
    fi

    RUN_CONFIG="predict" MODEL_NAME="../checkpoints/<checkpoint>" \
    DATASET_NAME="wiki727k" MAX_EVAL_SAMPLES=1000 MAX_PREDICT_SAMPLES=1000 \
    PARTITION_DISABLE_STRATEGY="cr" PARTITION_DISABLE_VALUE=1 PARTITION_STRATEGY="${strategy}" PARTITION_VALUE="${value}" \
    SEED=13 CUDA_VISIBLE_DEVICES="0" BS=8 EXPR_NAME="${expr_name}" OVERWRITE_CACHE="--overwrite_cache" \
    ./run_weswin_api.sh
    
    sleep 3
done
```

- `MODEL_NAME`: Specify a fine-tuned checkpoint.
- Metrics for each run will be saved as rows in `out/<DATASET_NAME>/results.csv`.

---

## Citation

If you use `WeSWin` in your research, please cite our paper:

```
@inproceedings{abbasi-etal-2025-neural,
    title = "Neural Document Segmentation Using Weighted Sliding Windows with Transformer Encoders",
    author = "Abbasi, Saeed  and
      An, Aijun  and
      Davoudi, Heidar  and
      Di Carlantonio, Ron  and
      Farmaner, Gary",
    editor = "Rambow, Owen  and
      Wanner, Leo  and
      Apidianaki, Marianna  and
      Al-Khalifa, Hend  and
      Eugenio, Barbara Di  and
      Schockaert, Steven  and
      Darwish, Kareem  and
      Agarwal, Apoorv",
    booktitle = "Proceedings of the 31st International Conference on Computational Linguistics: Industry Track",
    month = jan,
    year = "2025",
    address = "Abu Dhabi, UAE",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.coling-industry.67/",
    pages = "807--816",
    abstract = "We introduce a novel Transformer-based method for document segmentation, tailored for practical, real-world applications. This method utilizes overlapping text sequences with a unique position-aware weighting mechanism to enhance segmentation accuracy. Through comprehensive experiments on both public and proprietary datasets, we demonstrate significant improvements, establishing new state-of-the-art standards by achieving up to a 10{\%} increase in segmentation F1 score compared to existing methods. Additionally, we explore the application of our segmentation method in downstream retrieval-augmented question answering tasks, where it improves the quality of generated responses by 5{\%} while achieving up to four times greater efficiency. These results underscore our model`s potential as a robust and scalable solution for real-world text segmentation challenges."
}
```
