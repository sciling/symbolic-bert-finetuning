#! /bin/bash

poetry run src/bin/wn.py create-db --vocab-fn vocab.json --ignore-fn ignore.json --redefinitions-fn redefinitions.csv --entities-fn alimento_tipo_xlsx_redef.csv data/alimento_tipo/food-es-wp.csv data/alimento_tipo/tuberculos-es-wp.csv data/alimento_tipo/distilled-drinks-es-wp.csv --save-fn test2.json
poetry run src/bin/wn.py summarize-db --description-type long --reuse-descriptions test2.json --vocab-fn vocab.json --ignore-fn ignore.json --save-fn test2-summarized.json
