import csv
from unidecode import unidecode
import spacy_stanza
from inflector import Inflector, Spanish


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
        return tuple([unidecode(cls.singularize_spacy_token(t)) for t in seq])

    @classmethod
    def singularize_spacy_token(cls, token):
        if token.lemma_ != token.text:
            return token.lemma_
        else:
            return cls.singularize(token.text)

    @classmethod
    def update_vocab(cls, vocab, vocab_fn):
        with open(vocab_fn) as file:
            csvreader = csv.reader(file, delimiter=',', quotechar='"')
            for row in csvreader:
                token = '_'.join(cls.normalize(row[0]))
                vocab[token] = row[1]
                # print(f"'{row[0]}' -> vocab[{token}] = {row[1][:80]}")
        return vocab

    @classmethod
    def tokenize(cls, sentence, vocab):
        max_ngram_len = max([len(w.split('_')) for w in vocab])
        sentence = cls.normalize(sentence)
        total_words = len(sentence)
        print(f"{sentence}: {total_words} {max_ngram_len}")

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
        description = '.\n'.join(vocab[w] for w in seq if w in vocab)
        return seq, description
