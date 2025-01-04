#!/bin/bash

### Wiki-727K
TMP_HOME=..
WIKI_ROOT=${TMP_HOME}/datasets/wiki727k

WIKI_DOWNLOAD_URL="https://www.dropbox.com/sh/k3jh0fjbyr0gw0a/AACKW_gsxUf282QqrfH3yD10a/wiki_727K.tar.bz2?dl=0"
WIKI_TAR_PATH=${WIKI_ROOT}/wiki_727K.tar.bz2
WIKI_PATH=${WIKI_ROOT}/wiki_727

if [ ! -e ${WIKI_ROOT} ]; then 
    mkdir -p ${WIKI_ROOT}

    # Download
    if [ ! -e ${WIKI_TAR_PATH} ]; then
        echo ">> Download Wiki-727K..."
        curl -L ${WIKI_DOWNLOAD_URL} -o ${WIKI_TAR_PATH}
    fi

    # Extract
    if [ ! -e ${WIKI_PATH} ]; then
        echo -e ">> Extract Wiki-727K..."
        tar -xjf ${WIKI_TAR_PATH} -C ${WIKI_ROOT}
    fi
fi

# Preprocess
echo -e "\n>> Preprocess Wiki-727K:"
python -m data_util.wiki727k_dataset --data_dir ${WIKI_PATH}