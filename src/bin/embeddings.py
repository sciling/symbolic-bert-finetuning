#! /usr/bin/env python

import csv
from pathlib import Path

import typer

from medp.utils import NLP, EmbeddingsProcessor


app = typer.Typer()


def generate_embeddings(csv_fn: Path):
    with open(csv_fn) as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')

        descriptions = [row[1] for row in reader]

    embeddings = EmbeddingsProcessor.pages_to_embeddings(descriptions)

    for desc, emb in zip(descriptions, embeddings):
        print(desc, emb)


@app.command()
def get_closest(csv_fn: Path, db_fn: Path):
    pass


if __name__ == "__main__":
    app()
