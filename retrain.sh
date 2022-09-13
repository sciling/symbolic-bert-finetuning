#! /bin/bash

set -euxo pipefail

function get_corpus() {
  jq -r '.|to_entries[]| select(.value.correction != null) | [.key,.value.correction]|@csv' "$1"
}

function get_multicorpus() {
  jq -r '.|to_entries[]| select(.value.multicorrection != null) | [.key,.value.multicorrection]|@csv' "$1"
}

scp medp-gpu:ibm_watson_test/sentiment-database.json .
get_corpus sentiment-database.json | sort -u > manual-sentiment.csv
poetry run src/bin/wn.py expand-entities --depth 2 --threshold 0.5  template.yaml --save-fn sentiment.csv --vars-fn neutral.vocab --vars-fn negative.vocab --vars-fn positive.vocab --max-syns 10000
poetry run src/bin/classifier.py entities-to-dataset sentiment.csv sentiment-train.csv sentiment-dev.csv sentiment-test.csv --max-examples 20000 --entities "positive,negative,neutral" --dev 0.1 --test 0
cat manual-sentiment.csv >> sentiment-train.csv
poetry run src/bin/classifier.py train sentiment-train.csv sentiment-dev.csv --num-train-epochs 3 --output-dir train.dir

scp medp-gpu:ibm_watson_test/mood-database.json .

# get_corpus mood-database.json | sort -u > manual-mood.csv
# poetry run src/bin/wn.py expand-entities --depth 2 --threshold 0.5 estado_animo_entity_template.yaml --save-fn estado_animo_entity_custom.csv --vars-fn custom.vocab
# cat estado_animo_entity_expanded.csv estado_animo_entity_custom.csv | sort -u > estado_animo_entity_expanded_custom.csv
# poetry run src/bin/classifier.py entities-to-dataset estado_animo_entity_expanded_custom.csv mood-train.csv mood-dev.csv mood-test.csv --max-examples 20000 --dev 0.1 --test 0
# cat manual-mood.csv >> mood-train.csv
# poetry run src/bin/classifier.py train mood-train.csv mood-dev.csv --num-train-epochs 3 --output-dir train2.dir

get_multicorpus mood-database.json | sort -u > manual-multimood.csv
poetry run src/bin/wn.py expand-entities --depth 2 --threshold 0.5 estado_animo_entity_template.yaml --save-fn estado_animo_entity_custom.csv --vars-fn custom.vocab
cat estado_animo_entity_expanded.csv estado_animo_entity_custom.csv | sort -u > estado_animo_entity_expanded_custom.csv
poetry run src/bin/classifier.py entities-to-dataset estado_animo_entity_expanded_custom.csv multimood-train.csv multimood-dev.csv multimood-test.csv --max-examples 20000 --dev 0.1 --test 0
# echo '"sentence","label"' > multimood-train.csv
poetry run src/bin/wn.py recombine-intents manual-multimood.csv multimood-recombined.csv
cat manual-multimood.csv multimood-recombined.csv >> multimood-train.csv
echo '"sentence","label"' > multimood-dev.csv
cat manual-multimood.csv >> multimood-dev.csv
poetry run src/bin/classifier.py train multimood-train.csv multimood-dev.csv --num-train-epochs 3 --problem-type multi_label_classification --output-dir train-ml.dir
