import re
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
from nltk.util import ngrams

cos = CosineSimilarity(dim=1, eps=1e-6)


class EmbeddingsProcessor:
    EMBEDDINGS_MODEL = None

    @classmethod
    def get_model(cls):
        if cls.EMBEDDINGS_MODEL is None:
            # cls.EMBEDDINGS_MODEL = SentenceTransformer("paraphrase-distilroberta-base-v2")
            cls.EMBEDDINGS_MODEL = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
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


noise_re = re.compile(r"\s*Root -> .*$")
punct_re = re.compile(r"[^\w\s]*")


class NLP:
    INFLECTOR = None
    NLP = None
    STOPWORDS = {'tipo'}

    @classmethod
    def get_model(cls):
        if cls.NLP is None:
            cls.NLP = spacy_stanza.load_pipeline("es", processors="tokenize,lemma")
            cls.NLP.Defaults.stop_words |= cls.STOPWORDS
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
        return [unidecode(cls.singularize_spacy_token(t)) for t in seq if not t.is_punct and not t.is_stop]

    @classmethod
    def split(cls, sentence):
        seq = cls.nlp(sentence.lower())
        return [t.text for t in seq if not t.is_punct]

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
                row[1] = row[1].split('###')[0]
                desc = noise_re.sub('', '. '.join(row))
                res = {
                    'description': desc,
                    'embedding': EmbeddingsProcessor.pages_to_embeddings([desc])[0].tolist(),
                }

                for num, alt in enumerate([row[0]]):  # enumerate([row[0]] + row[2:]):
                    token = '_'.join(cls.normalize(alt))
                    if token not in vocab:
                        vocab[token] = {'label': alt}
                        vocab[token].update(res)
                        if num == 0 and is_entity:
                            vocab[token]['is_entity'] = True
                    else:
                        vocab[token]['description'] = desc
                        vocab[token]['embedding'] = EmbeddingsProcessor.pages_to_embeddings([desc])[0].tolist()

                # print(f"'{row[0]}' -> vocab[{token}] = {row[1][:80]}")
        return vocab

    @classmethod
    def generate_embeddings(cls, vocab):
        return vocab

    @classmethod
    def is_valid_ngram(cls, sentence):
        seq = cls.nlp(sentence)
        return not seq[0].is_stop and not seq[0].like_num and not seq[0].is_punct and not seq[-1].is_stop and not seq[-1].is_punct

    @classmethod
    def get_all_ngrams(cls, sentence, max_ngram=5):
        sentence = [w for w in punct_re.sub('', sentence).lower().split(' ') if w]
        ngs = set()
        for n in range(1, max_ngram + 1):
            fullset = [' '.join(ng) for ng in ngrams(sentence, n)]
            valid = {n for n in fullset if cls.is_valid_ngram(n)}
            ngs.update(valid)
        return ngs

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
    def summarize(cls, sentence, vocab, long_description=True):
        seq, description = cls.describe(sentence, vocab, long_description=long_description)
        summary = {token for token in cls.tokenize(description, vocab) if token in vocab}
        return seq, summary

    @classmethod
    def summarizedb_entry(cls, entry, vocab, long_description=True):
        entry['summary'] = list(sorted(cls.summarize(entry['label'], vocab, long_description)[1]))
        desc = " ".join([w for token in entry['summary'] for w in token.split('_')])
        entry['summaryEmbedding'] = EmbeddingsProcessor.pages_to_embeddings([desc])[0].tolist()
        return entry

    @classmethod
    def describe(cls, sentence, vocab, long_description=False):
        if long_description:
            seq = ['_'.join(cls.normalize(token)) for token in cls.get_all_ngrams(sentence)]
        else:
            seq = cls.tokenize(sentence, vocab)
        description = '. '.join(vocab[w].get('description', '') for w in seq if w in vocab)
        return seq, description


class SearchEngine:
    def __init__(self, db_fn):
        with open(db_fn) as file:
            self.vocab = json.load(file)

        self.entities = [(ent, data) for ent, data in self.vocab.items() if data.get('is_entity', False)]
        self.entity_embeddings = torch.stack([torch.FloatTensor(data.get('embedding')) for ent, data in self.entities])
        self.entity_summary_embeddings = torch.stack([torch.FloatTensor(data.get('summaryEmbedding', [])) for ent, data in self.entities])

    def search(self, sentence, nbest=4, vocab=False):
        embedding = None
        if vocab:
            entry = NLP.summarizedb_entry({'label': sentence}, vocab)
            seq = entry['summary']
            desc = " ".join([w for token in entry['summary'] for w in token.split('_')])
            embedding = entry['summaryEmbedding']
        else:
            seq, desc = NLP.describe(sentence, self.vocab)
        desc = f"{sentence}. {desc}"
        print(f"{sentence}: {seq}: {desc}")

        return {
            'tokens': seq,
            'description': desc,
            'nbests': self.literal_search(desc, nbest, embedding=embedding),
        }

    def literal_search(self, desc, nbest=4, embedding=None):
        if not embedding:
            embedding = EmbeddingsProcessor.pages_to_embeddings([desc])[0]
        scores = cos(self.entity_embeddings, embedding)
        index_sorted = torch.argsort(scores)
        top_scores = reversed(index_sorted[-nbest:])

        return [
            {
                'entity': self.entities[i][1]['label'],
                'score': scores[i].item(),
                'description': self.entities[i][1]['description'],
            }
            for i in top_scores
        ]
