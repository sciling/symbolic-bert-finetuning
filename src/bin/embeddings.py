#! /usr/bin/env python

import csv
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

import typer
from sentence_transformers import SentenceTransformer
from sentence_transformers import util


app = typer.Typer()


class EmbeddingsProcessor:
    EMBEDDINGS_MODEL = None

    @classmethod
    def get_model(cls):
        if cls.EMBEDDINGS_MODEL is None:
            cls.EMBEDDINGS_MODEL = SentenceTransformer("paraphrase-distilroberta-base-v2")
        return cls.EMBEDDINGS_MODEL

    @classmethod
    def pages_to_embeddings(cls, pages_content: Iterable[str]) -> torch.Tensor:
        webs_sent_data = []
        indexes = []
        splitted_sentences = []
        for i, page in enumerate(pages_content):
            single_page = list(page.split("\n"))
            indexes.append((i, len(single_page)))
            splitted_sentences.extend(single_page)
        web_embedding = cls.get_model().encode(splitted_sentences)
        acumulator = 0
        for i, number in indexes:
            # web_embedding[acumulator: acumulator + number, :] are the phrases from a single document
            single_web_embedding = np.mean(web_embedding[acumulator : acumulator + number, :], 0)
            webs_sent_data.append(single_web_embedding)
            acumulator += number
        torch_vector = torch.from_numpy(np.array(webs_sent_data)).float()  # pylint: disable=no-member
        web_embeddings = util.normalize_embeddings(torch_vector)
        return web_embeddings


@app.command()
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
