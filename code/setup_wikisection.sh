#!/bin/bash

### WikiSection
TMP_HOME=..
WIKI_ROOT=${TMP_HOME}/datasets/wikisection
mkdir -p ${WIKI_ROOT}

# Download
WIKI_DOWNLOAD_URL="https://github.com/sebastianarnold/WikiSection/raw/master/wikisection_dataset_json.tar.gz"
WIKI_TAR_PATH=${WIKI_ROOT}/wikisection_dataset_json.tar.gz
if [ ! -e ${WIKI_TAR_PATH} ]; then
    echo ">> Download WikiSection:"
    curl -L ${WIKI_DOWNLOAD_URL} -o ${WIKI_TAR_PATH}
fi

# Extract
EN_CITY_ROOT=${WIKI_ROOT}/en_city
EN_DISEASE_ROOT=${WIKI_ROOT}/en_disease
if [ ! -e ${EN_CITY_ROOT} ] || [ ! -e ${EN_DISEASE_ROOT} ]; then
    echo ">> Extract WikiSection:"
    tar -xvzf ${WIKI_TAR_PATH} -C ${WIKI_ROOT}

    # en_city
    mkdir -p ${EN_CITY_ROOT}
    mv ${WIKI_ROOT}/wikisection_en_city_*.json ${EN_CITY_ROOT}

    # en_disease
    mkdir -p ${EN_DISEASE_ROOT}
    mv ${WIKI_ROOT}/wikisection_en_disease_*.json ${EN_DISEASE_ROOT}
fi

# Preprocess en_city
echo -e "\n>> Preprocess en_city:"
python -m data_util.wikisection_dataset --data_dir ${EN_CITY_ROOT}

# Preprocess en_disease
echo -e "\n>> Preprocess en_disease:"
python -m data_util.wikisection_dataset --data_dir ${EN_DISEASE_ROOT}
