#! /usr/bin/env python

import csv
import json
from pathlib import Path
from typing import List
import random

import typer
from ruamel.yaml import YAML
from transformers import pipeline
import torch
from torch.nn import CosineSimilarity

from medp.utils import EmbeddingsProcessor

app = typer.Typer()

cos = CosineSimilarity(dim=1, eps=1e-6)


@app.command()
def few_shot(entities_fn: Path, new_samples: List[str] = typer.Argument(None), interactive: bool = False):
    typer.echo(f"Processing terms {entities_fn}' ...")

    if entities_fn.suffix in ('.yaml', '.yml'):
        yaml = YAML(typ="safe")  # default, if not specfied, is 'rt' (round-trip)
        with open(entities_fn) as file:
            entities = yaml.load(file)

    else:
        entities = {}
        with open(entities_fn) as file:
            csvreader = csv.reader(file, delimiter=',', quotechar='"')
            for row in csvreader:
                entities[row[1]] = list(set(row[2:]))

    samples = [sample for value in entities.values() for sample in value]
    labels = [label for label, value in entities.items() for sample in value]

    n_good = 0
    X = EmbeddingsProcessor.pages_to_embeddings(samples)
    for i, (label, sample) in enumerate(zip(labels, samples)):
        scores = cos(X[i], X)
        index_sorted = torch.argsort(scores)
        output = [(labels[i], scores[i], samples[i]) for i in index_sorted if samples[i] != sample]
        print(f"'{sample}', '{label}', {list(reversed(output))[0:3]}")
        if label == labels[index_sorted[-2]]:
            n_good += 1

    total = len(samples)
    print(f"{n_good} / {total} = {n_good/total}")

    if new_samples:
        n_good = 0
        new_X = EmbeddingsProcessor.pages_to_embeddings(new_samples)
        for i, sample in enumerate(new_samples):
            scores = cos(new_X[i], X)
            index_sorted = torch.argsort(scores)
            output = [(labels[i], scores[i], samples[i]) for i in index_sorted if samples[i] != sample]
            print(sample, list(reversed(output))[0:3])


@app.command()
def zero_shot(entities_fn: Path, multi_label: bool = False):
    typer.echo(f"Processing terms {entities_fn}' ...")

    yaml = YAML(typ="safe")  # default, if not specfied, is 'rt' (round-trip)
    with open(entities_fn) as file:
        entities = yaml.load(file)

    candidate_labels = list(entities.keys())
    samples = [sample for value in entities.values() for sample in value]
    labels = [label for label, value in entities.items() for sample in value]

    classifier = pipeline("zero-shot-classification", model="Recognai/zeroshot_selectra_medium", device=0)

    res = classifier(
        samples,
        candidate_labels=candidate_labels,
        hypothesis_template="Este ejemplo es {}.",
        multi_class=multi_label
    )

    output = [r['labels'][0] for r in res]
    scores = [r['scores'][0] for r in res]
    print(json.dumps(list(zip(labels, output, scores, samples)), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    app()
