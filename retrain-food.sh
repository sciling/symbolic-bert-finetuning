#! /bin/bash

set -euxo pipefail

grep -h '\(cena\|desayun\|comer\|comida\|merienda\|postre\|hoy\|ayer\|mañana\)' enfemenino.txt foropsicologia.txt saludccm.txt \
  | tr -d '"|()' | awk -OFS=, '{print "otras,\""$0"\""}' | sort -u > otras.vocab

poetry run src/bin/wn.py expand-entities --depth 2 --threshold 0.5  multialimentos.yaml \
  --max-permutations 20000 \
  --save-fn multialimentos.csv \
  --vars-fn alimentos.vocab --vars-fn token-alimentos.vocab --vars-fn alimento-cantidad.vocab \
  --vars-fn alimento-unidad.vocab --vars-fn alimento-toma.vocab \
  --vars-fn otras.vocab

poetry run src/bin/classifier.py entities-to-dataset multialimentos.csv \
  multialimentos-train.csv multialimentos-dev.csv multialimentios-test.csv \
  --test 0 --max-examples 100000

poetry run src/bin/classifier.py train-token multialimentos-train.csv multialimentos-dev.csv \
  --num-train-epochs 3 --output-dir train-ma.dir
