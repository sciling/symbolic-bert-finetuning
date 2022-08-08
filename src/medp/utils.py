import re
import json
import csv
from typing import Iterable
from enum import Enum

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
note_re = re.compile(r"\s*\([^\)]*\)")


class DescriptionType(Enum):
    DEFAULT = 'default'
    LONG = 'long'
    SHORT = 'short'


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
        seq = cls.nlp(note_re.sub('', sentence.lower()))
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
    def update_vocab(cls, vocab, vocab_fn, is_entity=False, redefinitions=None, override_definitions=True):
        with open(vocab_fn) as file:
            csvreader = csv.reader(file, delimiter=',', quotechar='"')
            rows = list(csvreader)
            for row in tqdm(rows, desc=f"Loading '{vocab_fn}'"):
                token = '_'.join(cls.normalize(row[0]))
                row[1] = row[1].split('###')[0]
                desc = noise_re.sub('', '. '.join(row))

                if redefinitions and token in redefinitions:
                    if redefinitions[token] == 'DROP':
                        continue
                    else:
                        desc = noise_re.sub('', redefinitions[token])

                if token not in vocab:
                    vocab[token] = {
                        'label': row[0],
                        'description': desc,
                        'embedding': EmbeddingsProcessor.pages_to_embeddings([desc])[0].tolist(),
                        'synonyms': [],
                    }
                elif override_definitions:
                    vocab[token].update({
                        'description': desc,
                        'embedding': EmbeddingsProcessor.pages_to_embeddings([desc])[0].tolist(),
                    })

                if is_entity:
                    vocab[token]['is_entity'] = True
                    vocab[token]['label'] = row[0]

                vocab[token]['synonyms'] = list(set(vocab[token]['synonyms']) | {'_'.join(cls.normalize(alt)) for alt in row[2:]})

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
        if vocab:
            max_ngram_len = min(1, max([len(w.split('_')) for w in vocab]))
        else:
            max_ngram_len = 1
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
    def summarize(cls, sentence, vocab, description_type=DescriptionType.DEFAULT, description=None):
        # print(f"INITDES: '{description}'")
        if not description:
            _, description = cls.describe(sentence, vocab, description_type=description_type)
        # print(f"POSTDES: '{description}'")
        if description_type == DescriptionType.LONG:
            all_ngrams = cls.get_all_ngrams(description)
            # print(f"NGRAMTOKS: {sentence} '{description}' {all_ngrams}")
            tokens = {'_'.join(cls.normalize(token)) for token in all_ngrams}
        elif description_type == DescriptionType.SHORT:
            tokens = cls.tokenize(description, vocab={})
        else:
            tokens = cls.tokenize(description, vocab)
        summary = {token for token in tokens if token in vocab}
        # print(f"SUMMARIZE({len(vocab)}, {description_type}): {sentence} {tokens} {summary}")
        return summary

    @classmethod
    def summarizedb_entry(cls, entry, vocab, description_type=DescriptionType.DEFAULT, reuse_description=False):
        description = None
        if reuse_description:
            description = entry.get('description', None)
        entry['summary'] = list(sorted(cls.summarize(entry['label'], vocab, description_type, description)))
        desc = " ".join([w for token in entry['summary'] for w in token.split('_')])
        # entry['summaryEmbedding'] = EmbeddingsProcessor.pages_to_embeddings([desc])[0].tolist()
        if 'embedding' in entry:
            entry['summaryEmbedding'] = entry['embedding']
        else:
            entry['summaryEmbedding'] = EmbeddingsProcessor.pages_to_embeddings([desc])[0].tolist()
        return entry

    @classmethod
    def describe(cls, sentence, vocab, description_type=DescriptionType.DEFAULT):
        if description_type == DescriptionType.LONG:
            all_ngrams = cls.get_all_ngrams(sentence)
            # print(f"DESCRIBE: {sentence} -> {all_ngrams}")
            seq = ['_'.join(cls.normalize(token)) for token in all_ngrams]
        elif description_type == DescriptionType.SHORT:
            seq = cls.tokenize(sentence, vocab={})
        else:
            seq = cls.tokenize(sentence, vocab)
        description = '. '.join(vocab[w].get('description', '') for w in seq if w in vocab)
        return seq, description

    @classmethod
    def to_multinomial(cls, summary, tok2id):
        vect = [0] * len(tok2id)
        for tok in summary:
            tokid = tok2id.get(tok, None)
            if tokid is not None:
                vect[tokid] = 1
        return torch.FloatTensor(vect)


class SearchEngine:
    def __init__(self, db_fn, ignore_fn=None):
        with open(db_fn) as file:
            self.vocab = json.load(file)

        if ignore_fn:
            with open(ignore_fn) as file:
                self.ignore = set(json.load(file))
        else:
            self.ignore = set()

        for token in list(self.vocab.keys()):
            if len(token) < 3 or token in self.ignore:
                del self.vocab[token]

        self.entities = [(ent, data) for ent, data in self.vocab.items() if data.get('is_entity', False)]
        self.entity_embeddings = torch.stack([torch.FloatTensor(data.get('embedding')) for ent, data in self.entities])
        self.entity_summary_embeddings = torch.stack([torch.FloatTensor(data.get('summaryEmbedding', [])) for ent, data in self.entities])

        self.tok2id = {tok: n for n, tok in enumerate(sorted(self.vocab.keys()))}
        self.id2tok = {n: tok for tok, n in self.tok2id.items()}

        # After id2tok is computed, because otherwise id2tok might be associated with a synonym and not the main lemma.
        for word, data in list(self.vocab.items()):
            synonyms = {tok: self.tok2id[word] for tok in data.get('synonyms', [])}
            # print(f"tok2id: {word} ({self.tok2id[word]}) -> {synonyms}")
            self.tok2id.update(synonyms)
            vocab_synonyms = {tok: self.vocab[word] for tok in data.get('synonyms', [])}
            # print(f"tok2id: {word} ({self.tok2id[word]}) -> {synonyms}")
            self.vocab.update(vocab_synonyms)

        # print(self.tok2id)
        self.entity_multinomial = []
        for ent, data in self.entities:
            self.entity_multinomial.append(NLP.to_multinomial(data.get('summary', set()), self.tok2id))
        self.entity_multinomial = torch.stack(self.entity_multinomial)
        # print(len(self.entities))

    def search(self, sentence, nbest=4, summarized=False, multinomial=False, description_type=DescriptionType.DEFAULT, reuse_description=True):
        embedding = None
        if summarized:
            entry = NLP.summarizedb_entry({'label': sentence}, self.vocab, description_type=description_type, reuse_description=reuse_description)
            seq = entry['summary']
            desc = " ".join([w for token in seq for w in token.split('_')])
            if multinomial:
                embedding = NLP.to_multinomial(seq, self.tok2id)
            else:
                embedding = torch.FloatTensor(entry['summaryEmbedding'])
        else:
            seq, desc = NLP.describe(sentence, self.vocab)
        desc = f"{sentence}. {desc}"
        # print(f"{sentence}: {seq}: {desc}")

        return {
            'tokens': seq,
            'description': desc,
            'nbests': self.literal_search(desc, nbest, embedding=embedding, multinomial=multinomial),
        }

    def literal_search(self, desc, nbest=4, embedding=None, multinomial=False):
        is_summary = embedding is not None
        if is_summary:
            if multinomial:
                entity_embeddings = self.entity_multinomial
            else:
                entity_embeddings = self.entity_summary_embeddings
        else:
            embedding = EmbeddingsProcessor.pages_to_embeddings([desc])[0]
            entity_embeddings = self.entity_embeddings

        scores = cos(entity_embeddings, embedding)
        if is_summary:
            mult = torch.mul(embedding, entity_embeddings).sum(dim=1)
            entity_coverage = mult / entity_embeddings.sum(dim=1)
            scores = (scores + entity_coverage + mult) / (embedding.sum() + 2.0)

        index_sorted = torch.argsort(scores)
        top_scores = reversed(index_sorted[-nbest:])
        # print([(scores[i], self.entities[i][1]['label']) for i in index_sorted])

        return [
            {
                'entity': self.entities[i][1]['label'],
                'score': scores[i].item(),
                'description': self.entities[i][1]['summary' if is_summary else 'description'],
            }
            for i in top_scores if scores[i].item() > 0
        ]
