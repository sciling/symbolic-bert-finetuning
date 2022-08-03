import json
import csv
from typing import Iterable

from unidecode import unidecode
import spacy_stanza
from inflector import Inflector, Spanish
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
import numpy as np
import torch
from torch.nn import CosineSimilarity
from tqdm import tqdm

cos = CosineSimilarity(dim=1, eps=1e-6)


class EmbeddingsProcessor:
    EMBEDDINGS_MODEL = None

    @classmethod
    def get_model(cls):
        if cls.EMBEDDINGS_MODEL is None:
            cls.EMBEDDINGS_MODEL = SentenceTransformer("paraphrase-distilroberta-base-v2")
            if torch.cuda.is_available():
                print(f"CUDA: {torch.cuda.get_device_name(0)}")
            else:
                print(f"CUDA: {torch.cuda.is_available()}")
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


class NLP:
    INFLECTOR = None
    NLP = None

    @classmethod
    def get_model(cls):
        if cls.NLP is None:
            cls.NLP = spacy_stanza.load_pipeline("es", processors="tokenize,lemma")
        return cls.NLP

    @classmethod
    def get_inflector(cls):
        if cls.INFLECTOR is None:
            cls.INFLECTOR = Inflector(Spanish)
        return cls.INFLECTOR

    @classmethod
    def nlp(cls, text):
        return cls.get_model()(text)

    @classmethod
    def singularize(cls, text):
        return cls.get_inflector().singularize(text)

    @classmethod
    def normalize(cls, sentence):
        seq = cls.nlp(sentence.lower())
        return [unidecode(cls.singularize_spacy_token(t)) for t in seq]

    @classmethod
    def singularize_spacy_token(cls, token):
        if token.lemma_ != token.text:
            return token.lemma_

        return cls.singularize(token.text)

    @classmethod
    def update_vocab(cls, vocab, vocab_fn, is_entity=False):
        with open(vocab_fn) as file:
            csvreader = csv.reader(file, delimiter=',', quotechar='"')
            rows = list(csvreader)
            for row in tqdm(rows, desc=f"Loading '{vocab_fn}'"):
                res = {
                    'description': row[1],
                    'embedding': EmbeddingsProcessor.pages_to_embeddings([row[1]])[0].tolist(),
                }

                for num, alt in enumerate([row[0]] + row[2:]):
                    token = '_'.join(cls.normalize(alt))
                    if token not in vocab:
                        vocab[token] = {'label': alt}
                        vocab[token].update(res)
                        if num == 0 and is_entity:
                            vocab[token]['is_entity'] = True
                # print(f"'{row[0]}' -> vocab[{token}] = {row[1][:80]}")
        return vocab

    @classmethod
    def tokenize(cls, sentence, vocab):
        max_ngram_len = max([len(w.split('_')) for w in vocab])
        sentence = cls.normalize(sentence)
        total_words = len(sentence)

        seq = []

        i = 0
        while i < total_words:
            ngram = None
            advance = 1
            for n_words in range(max_ngram_len, 0, -1):
                ngram = '_'.join(sentence[i : i + n_words])

                if ngram in vocab:
                    advance = n_words
                    break

            i += advance
            seq.append(ngram)
            # print(f"{n_words}-gram({i}+{advance}): {ngram}: {seq}")
        return seq

    @classmethod
    def describe(cls, sentence, vocab):
        seq = cls.tokenize(sentence, vocab)
        description = '.\n'.join(vocab[w].get('description', '') for w in seq if w in vocab)
        return seq, description


class SearchEngine:
    def __init__(self, db_fn):
        with open(db_fn) as file:
            self.vocab = json.load(file)

        self.entities = [(ent, data) for ent, data in self.vocab.items() if data.get('is_entity', False)]
        self.entity_embeddings = torch.stack([torch.FloatTensor(data.get('embedding')) for ent, data in self.entities])

    def search(self, sentence, nbest=4):
        seq, desc = NLP.describe(sentence, self.vocab)
        desc = f"{sentence}. {desc}"
        print(f"{sentence}: {seq}: {desc}")
        embedding = EmbeddingsProcessor.pages_to_embeddings([desc])[0]
        scores = cos(self.entity_embeddings, embedding)
        index_sorted = torch.argsort(scores)
        top_scores = reversed(index_sorted[-nbest:])

        return seq, desc, [(self.entities[i][1]['label'], scores[i].item(), self.entities[i][1]['description']) for i in top_scores]
