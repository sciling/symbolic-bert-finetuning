#! /bin/bash

set -euxo pipefail

function get_corpus() {
  jq -r '.|to_entries[]| select(.value.correction != null) | [.key,.value.correction]|@csv' "$1"
}

function get_multicorpus() {
  jq -r '.|to_entries[]| select(.value.multicorrection != null) | [.key,.value.multicorrection]|@csv' "$1"
}

# scp medp-gpu:ibm_watson_test/sentiment-database.json .
# get_corpus sentiment-database.json | sort -u > manual-sentiment.csv
# poetry run src/bin/wn.py expand-entities --depth 2 --threshold 0.5  template.yaml --save-fn sentiment.csv --vars-fn neutral.vocab --vars-fn negative.vocab --vars-fn positive.vocab --max-syns 10000
# poetry run src/bin/classifier.py entities-to-dataset sentiment.csv sentiment-train.csv sentiment-dev.csv sentiment-test.csv --max-examples 20000 --entities "positive,negative,neutral" --dev 0.1 --test 0
# cat manual-sentiment.csv >> sentiment-train.csv
# poetry run src/bin/classifier.py train sentiment-train.csv sentiment-dev.csv --num-train-epochs 3 --output-dir train.dir

poetry run src/bin/wn.py expand-entities --depth 2 --threshold 0.5  multialimentos.yaml \
  --save-fn multialimentos.csv \
  --vars-fn alimentos-mini.vocab --vars-fn alimento-cantidad.vocab \
  --vars-fn alimento-unidad.vocab --vars-fn alimento-toma.vocab

poetry run src/bin/classifier.py entities-to-dataset multialimentos.csv \
  multialimentos-train.csv multialimentos-dev.csv multialimentios-test.csv \
  --test 0 --max-examples 20000

poetry run src/bin/classifier.py train-token multialimentos-train.csv multialimentos-dev.csv \
  --num-train-epochs 3 --output-dir train-ma.dir
