import re
import json
import csv
from typing import Iterable
from enum import Enum
from queue import Queue
import itertools

from unidecode import unidecode
import spacy_stanza
from inflector import Inflector, Spanish
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
import numpy as np
import torch
from torch.nn import CosineSimilarity
from tqdm import tqdm
from spellchecker import SpellChecker
import apertium
from nltk.util import ngrams
from nltk.corpus import wordnet as wn
from streamparser import LexicalUnit
from methodtools import lru_cache

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


@lru_cache(maxsize=None)
def get_syns(word, lang='spa', **kwargs):
    try:
        if isinstance(word, str):
            syns = [wn.synset(word.lower())]
        else:
            syns = [word]
    except ValueError:
        singular = join_blocks([w.lemma_ for w in NLP.nlp(word)])
        if singular == word:
            singular = join_blocks([NLP.singularize(w.text) for w in NLP.nlp(word)])
        syns = list(set(wn.synsets(word.lower(), lang=lang, **kwargs) + wn.synsets(singular.lower(), lang=lang, **kwargs)))
        # print(f"SING: {word} -> {singular} -> {syns}")

    return syns


noise_re = re.compile(r"\s*Root -> .*$")
punct_re = re.compile(r"[^\w\s]*")
note_re = re.compile(r"\s*\([^\)]*\)")

APERTIUM_FIX = {
    'para': LexicalUnit('para/para<pr>'),
    'siento': LexicalUnit('siento/sentir<vblex><pri><p1><sg>'),
}

FORMS = {
    'vblex': [
        "{lemma}<vblex><pri><{person}><{number}>",  # yo sollozo
        "haber<vbhaver><pri><{person}><{number}> {lemma}<vblex><pp><m><sg>",  # yo he sollozado
        "{lemma}<vblex><ifi><{person}><{number}>",  # yo sollocé
        # "haber<vbhaver><prs><{person}><{number}> {lemma}<vblex><pp><m><sg>",  # yo haya sollozado
        # "haber<vbhaver><fts><{person}><{number}> {lemma}<vblex><pp><m><sg>",  # yo hubiere sollozado
        # "{lemma}<vblex><fti><{person}><{number}>",  # yo sollozaré
        # "haber<vbhaver><pii><{person}><{number}> {lemma}<vblex><pp><m><sg>",  # yo había sollozado
        # "haber<vbhaver><cni><{person}><{number}> {lemma}<vblex><pp><m><sg>",  # yo habría sollozado
        # "{lemma}<vblex><pis><{person}><{number}>",  # yo sollozara
        "estar<vblex><pri><{person}><{number}>, {lemma}<vblex><ger>",  # estoy sollozando
        # "haber<vbhaver><inf> {lemma}<vblex><pp><m><sg>",  # haber sollozado
        # "{lemma}<vblex><inf>",  # sollozar
        # "{lemma}<vblex><pii><{person}><{number}>",  # yo sollozaba
        # "haber<vbhaver><ifi><{person}><{number}> {lemma}<vblex><pp><m><sg>",  # yo hube sollozado
        # "{lemma}<vblex><cni><{person}><{number}>",  # yo sollozaría
        # "{lemma}<vblex><prs><{person}><{number}>",  # yo solloce
        # "haber<vbhaver><pis><{person}><{number}> {lemma}<vblex><pp><m><sg>",  # yo hubiera sollozado
        # "haber<vbhaver><ger> {lemma}<vblex><pp><m><sg>",  # habiendo sollozado
    ],
    'n': ["{lemma}<n><{gender}><{number}>"],
    'adj': ["{lemma}<adj><{gender}><{number}>"],
    'pp': ["{lemma}<vblex><pp><{gender}><{number}>"],
}


def permutations(obj):
    if isinstance(obj, list):
        return itertools.product(*obj)

    if isinstance(obj, dict):
        keys, values = zip(*obj.items())
        return [dict(zip(keys, v)) for v in itertools.product(*values)]

    raise Exception(f"Don't know how to make permutations of {type(obj)}")


template_re = re.compile(r"(?:\(([^)]*)\)|([^()]*))")
spaces_re = re.compile(r"(?:^\s+|[\s]+$|(\s)\s+)")


def join_blocks(blocks, sep=' '):
    return spaces_re.sub(r"\1", sep.join(blocks))


def clean_spaces(string):
    return spaces_re.sub(r"\1", string.replace('/', ' '))


def expand_template(template):
    blocks = [m.group(1).strip() if m.group(1) else m.group().strip() for m in template_re.finditer(template)]
    blocks = [block.split('|') for block in blocks if block]
    # print(f"EXP: {template}: {blocks}: {list(permutations(blocks))}")
    return {join_blocks(parts) for parts in permutations(blocks)}


class DescriptionType(Enum):
    DEFAULT = 'default'
    LONG = 'long'
    SHORT = 'short'


class NLP:
    APERTIUM_ANALIZER = None
    APERTIUM_TAGGER = None
    APERTIUM_GENERATOR = None
    SPELLCHECKER = None
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
    def get_spellchecker(cls):
        if cls.SPELLCHECKER is None:
            cls.SPELLCHECKER = SpellChecker(language='es')
        return cls.SPELLCHECKER

    @classmethod
    def update_spellchecker(cls, vocab):
        sc = cls.get_spellchecker()
        wf = sc.word_frequency
        wf.dictionary.update({v for v in vocab if v not in wf.dictionary})
        wf._update_dictionary()

    @classmethod
    def get_apertium_analyzer(cls):
        if cls.APERTIUM_ANALIZER is None:
            cls.APERTIUM_ANALIZER = apertium.Analyzer('spa')
        return cls.APERTIUM_ANALIZER

    @classmethod
    def get_apertium_tagger(cls):
        if cls.APERTIUM_TAGGER is None:
            cls.APERTIUM_TAGGER = apertium.Tagger('spa')
        return cls.APERTIUM_TAGGER

    @classmethod
    def get_apertium_generator(cls):
        if cls.APERTIUM_GENERATOR is None:
            cls.APERTIUM_GENERATOR = apertium.Generator('spa')
        return cls.APERTIUM_GENERATOR

    @classmethod
    def fix_apertium(cls, analysis):
        return [APERTIUM_FIX.get(tok.wordform, tok) for tok in analysis]

    @classmethod
    def analyze(cls, sentence):
        return cls.fix_apertium(cls.get_apertium_analyzer().analyze(sentence))

    @classmethod
    def tag(cls, sentence):
        tags = cls.get_apertium_tagger().tag(sentence)
        toks = cls.get_apertium_analyzer().analyze(sentence)
        tags = [LexicalUnit(f"{tok.wordform}/{tag.lexical_unit}") for tag, tok in zip(tags, toks)]
        return cls.fix_apertium(tags)

    @classmethod
    def generate(cls, unit):
        return cls.get_apertium_generator().generate(f"^{unit}$").replace('~', '').replace('*', '').replace('#', '')

    @classmethod
    def get_variations_type(cls, lemma, _values, options):
        values = {'lemma': [lemma]}
        values.update(_values)

        return {option.format(**v) for v in permutations(values) for option in options}

    @classmethod
    def filter_tokens_by_tag(cls, unit, tag):
        return {token.baseform for reading in unit.readings for token in reading if tag in token.tags}

    @lru_cache(maxsize=None)
    @classmethod
    def get_variations(cls, unit, number, preunit=None):
        # https://wiki.apertium.org/wiki/List_of_symbols
        alts = {'gender': ['m', 'f', 'mf'], 'number': [number], 'person': ['p1']}
        tags = {tag for reading in unit.readings for token in reading for tag in token.tags}

        filters = ['n', 'adj', 'pp']
        if 'pp' not in tags and (preunit is None or preunit.wordform == 'me' or 'inf' not in tags):
            filters.append('vblex')
            if preunit is not None and preunit.wordform == 'me':
                alts['person'] = ['p3']

        units = set()
        for tag in filters:
            tokens = cls.filter_tokens_by_tag(unit, tag)
            # print(f"TAG[{tag}]: {tokens}: {unit}: {unit.readings}")

            units |= {o for tok in tokens for o in cls.get_variations_type(tok, alts, FORMS[tag])}

        if not units:
            units.add(unit.wordform)

        variations = {join_blocks([cls.generate(part) for part in unit.split(' ')]) for unit in units}
        # print(f"UNITS: {units} {variations}")
        return {v for v in variations if '#' not in v}

    @classmethod
    def correct(cls, text):
        return cls.get_spellchecker().correction(text)

    @classmethod
    def nlp(cls, text):
        return cls.get_model()(text)

    @classmethod
    def longest_common_prefix(cls, word, ref):
        pairs = list(zip(word, ref))
        try:
            return [ac==bc for ac, bc in pairs].index(False)
        except:
            return len(pairs)

    @classmethod
    def find_longest_common_prefix(cls, words, ref):
        if not words:
            return None
        return sorted([(cls.longest_common_prefix(word, ref), word) for word in words], reverse=True)[0][1]

    @classmethod
    def pluralize(cls, text):
        return cls.get_inflector().pluralize(text)

    @classmethod
    def singularize(cls, text):
        return cls.get_inflector().singularize(text)

    @classmethod
    def normalize(cls, sentence, fuzzy=None):
        # print(f"SENT: {sentence}")
        seq = cls.nlp(clean_spaces(note_re.sub('', sentence.lower())))
        # print(f"SEQ: {seq}")
        if fuzzy:
            seq = cls.nlp(join_blocks([NLP.correct(t.text) for t in seq]))
        normalized = [unidecode(cls.singularize_spacy_token(t)) for t in seq if not t.is_punct and not t.is_stop and not t.is_space and t.text]
        # print(f"NORMALIZED: {normalized}")
        return [tok for tok in normalized if tok]

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
    def clean_notes(cls, sentence):
        return note_re.sub('', sentence.lower())

    @classmethod
    def update_vocab(cls, vocab, vocab_fn, is_entity=False, redefinitions=None, override_definitions=True, compute_embeddings=False):
        with open(vocab_fn) as file:
            csvreader = csv.reader(file, delimiter=',', quotechar='"')
            rows = list(csvreader)
            for row in tqdm(rows, desc=f"Loading '{vocab_fn}'"):
                row = [clean_spaces(r) for r in row]
                token = join_blocks(cls.normalize(row[0]), sep='_')
                # print(f"TOK: {cls.normalize(row[0])} {token}")
                row[1] = noise_re.sub('', row[1].split('###')[0])
                desc = '. '.join(list({clean_spaces(r).lower() for r in row}))

                if redefinitions and token in redefinitions:
                    if redefinitions[token] == 'DROP':
                        continue
                    else:
                        desc = noise_re.sub('', redefinitions[token])

                embedding = None
                if compute_embeddings:
                    EmbeddingsProcessor.pages_to_embeddings([desc])[0].tolist()

                if token not in vocab:
                    vocab[token] = {
                        'label': row[0],
                        'description': desc,
                        'embedding': embedding,
                        'alternatives': [],
                        'synonyms': [],
                    }
                elif override_definitions or not vocab[token]['description']:
                    vocab[token].update({
                        'description': desc,
                        'embedding': embedding,
                    })

                if is_entity:
                    vocab[token]['is_entity'] = True
                    vocab[token]['label'] = row[0]

                vocab[token]['alternatives'] = list(set(vocab[token]['alternatives']) | {join_blocks(cls.normalize(alt), sep='_') for alt in row[2:]})
                vocab[token]['synonyms'] = list(set(vocab[token]['synonyms']) | {cls.clean_notes(alt) for alt in [row[0]] + row[2:]})

                # print(f"'{row[0]}' -> vocab[{token}] = {row[1][:80]}")
        return vocab

    @classmethod
    def generate_embeddings(cls, vocab):
        return vocab

    @classmethod
    def is_valid_ngram(cls, sentence):
        seq = cls.nlp(sentence)
        if not seq:
            return False
        return not seq[0].is_stop and not seq[0].like_num and not seq[0].is_punct and not seq[-1].is_stop and not seq[-1].is_punct

    @classmethod
    def get_all_ngrams(cls, sentence, max_ngram=5, fuzzy=None):
        sentence = cls.normalize(sentence, fuzzy=fuzzy)
        # sentence = [w for w in punct_re.sub('', sentence).lower().split(' ') if w]
        ngs = set()
        for n in range(1, max_ngram + 1):
            fullset = [join_blocks(ng) for ng in ngrams(sentence, n)]
            valid = {n for n in fullset if cls.is_valid_ngram(n)}
            ngs.update(valid)
        return ngs

    @classmethod
    def tokenize(cls, sentence, vocab, fuzzy=None, subtokens=False, max_ngram_len=None):
        if vocab and not max_ngram_len:
            max_ngram_len = max(1, max([len(w.split('_')) for w in vocab]))

        sentence = cls.normalize(sentence, fuzzy=fuzzy)
        total_words = len(sentence)

        max_ngram_len = min(max_ngram_len, total_words)
        # print(f"SENTENCE: {sentence}. max_ngram_len: {max_ngram_len}")

        seq = []

        i = 0
        while i < total_words:
            ngram = None
            advance = 1
            for n_words in range(min(max_ngram_len, total_words - i), 0, -1):
                ngram = join_blocks(sentence[i : i + n_words], sep='_')

                # print(f"NGRAM: sentence[{i} : {i} + {n_words}] = {ngram}: {ngram in vocab}")
                if ngram in vocab:
                    advance = n_words
                    if subtokens:
                        seq.append(ngram)
                        # print(f"SUB: {n_words}-gram({i}+{advance}): {ngram}: {seq}")
                    else:
                        break

            if not subtokens:
                seq.append(ngram)
                # print(f"{n_words}-gram({i}+{advance}): {ngram}: {seq}")

            i += advance
        return seq

    @classmethod
    def get_tokens(cls, text, vocab, description_type=DescriptionType.DEFAULT, max_ngram=5, fuzzy=None):
        # print(f"PREEDES: '{text}'")
        if description_type == DescriptionType.LONG:
            tokens = cls.tokenize(text, vocab, fuzzy=fuzzy, subtokens=True, max_ngram_len=max_ngram)
            # all_ngrams = cls.get_all_ngrams(text, fuzzy=fuzzy, max_ngram=max_ngram)
            # # print(f"NGRAMTOKS: {sentence} '{description}' {all_ngrams}")
            # tokens = {join_blocks(cls.normalize(token), sep='_') for token in all_ngrams}
        elif description_type == DescriptionType.SHORT:
            tokens = cls.tokenize(text, vocab={}, fuzzy=None)
        else:
            tokens = cls.tokenize(text, vocab, fuzzy=fuzzy)
        # print(f"POSTDES: '{tokens}'")
        return tokens

    @classmethod
    def summarize(cls, sentence, vocab, description_type=DescriptionType.DEFAULT, description=None, max_ngram=5):
        # print(f"INITSUM: '{description}'")
        if not description:
            _, description = cls.describe(sentence, vocab, description_type=description_type, max_ngram=max_ngram)
        # print(f"POSTSUM: '{description}'")
        tokens = cls.get_tokens(description, vocab, description_type=description_type, max_ngram=max_ngram)
        summary = {token for token in tokens if token in vocab}
        # print(f"SUMMARIZE({len(vocab)}, {description_type}): {sentence} {tokens} {summary}")
        return summary

    @classmethod
    def summarizedb_entry(cls, entry, vocab, description_type=DescriptionType.DEFAULT, reuse_description=False, max_ngram=5):
        description = None
        if reuse_description:
            description = entry.get('description', None)
        entry['summary'] = list(sorted(cls.summarize(entry['label'], vocab, description_type, description, max_ngram)))
        desc = join_blocks([w for token in entry['summary'] for w in token.split('_')])
        if 'embedding' in entry:
            entry['summaryEmbedding'] = entry['embedding']
        else:
            entry['summaryEmbedding'] = EmbeddingsProcessor.pages_to_embeddings([desc])[0].tolist()
        return entry

    @classmethod
    def describe(cls, sentence, vocab, description_type=DescriptionType.DEFAULT, max_ngram=5, fuzzy=None):
        seq = cls.get_tokens(sentence, vocab, description_type, max_ngram=max_ngram, fuzzy=fuzzy)
        # print([vocab[w] for w in seq if w in vocab])
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

    @classmethod
    def convert_form(cls, word, from_pos=None, to_pos=None):
        """ Transform words given from/to POS tags """
        # based on https://nlpforhackers.io/convert-words-between-forms/

        if from_pos:
            synsets = get_syns(word, pos=from_pos, lang='spa')
        else:
            synsets = get_syns(word, lang='spa')

        # Word not found
        if not synsets:
            return []

        # Get all lemmas of the word (consider 'a'and 's' equivalent)
        lemmas = [
            lemma
            for s in synsets
            for lemma in s.lemmas()
            if not from_pos or s.name().split('.')[1] == from_pos
        ]

        # Get related forms
        derivationally_related_forms = [(lemma, lemma.derivationally_related_forms()) for lemma in lemmas]
        # print(f"DERIVA: {derivationally_related_forms}")

        # filter only the desired pos (consider 'a' and 's' equivalent)
        related_noun_lemmas = [
            lemma
            for drf in derivationally_related_forms
            for lemma in drf[1]
            if not to_pos or lemma.synset().name().split('.')[1] == to_pos
        ]

        # Extract the words from the lemmas
        words = [word.name() for lemma in related_noun_lemmas for word in lemma.synset().lemmas(lang='spa')]
        len_words = len(words)

        # Build the result in the form of a list containing tuples (word, probability)
        result = [
            (w, (cls.longest_common_prefix(w, word) + (float(words.count(w)) / len_words)) / (len(word) + 1))
            for w in set(words)
        ]
        result.sort(key=lambda w: -w[1])

        return result

    @classmethod
    def convert_form_recursive(cls, word, from_pos=None, to_pos=None, threshold=None, depth=1):
        conversions = set()
        all_conversions = set()
        queue = Queue()
        queue.put((word, 1))

        while not queue.empty():
            word, curr_depth = queue.get()
            if curr_depth > depth:
                continue

            res = NLP.convert_form(word)
            # print(f"RES: {res}")
            if not res:
                continue

            conversions |= {word for word, score in res if (threshold is None or score > threshold)}
            all_conversions |= {word for word, _ in res}

            for word, score in res:
                if (threshold is None or score > threshold) and curr_depth + 1 <= depth:
                    queue.put((word, curr_depth + 1))

        return conversions


def load_db(db_fn, vocab_fn, ignore_fn):
    if isinstance(db_fn, dict):
        db = db_fn
    else:
        with open(db_fn) as file:
            db = json.load(file)

    if vocab_fn:
        with open(vocab_fn) as file:
            data = json.load(file)

        for token, summary in data.items():
            if token not in db:
                db[token] = {
                    "label": token,
                    "description": token,
                    "embedding": None,
                    "alternatives": [token],
                    "synonyms": [token],
                    "summary": [token],
                    "summaryEmbedding": None,
                }

            # print(f"VOCAB: {token} {type(summary)} {summary}")
            if isinstance(summary, list):
                print(f"REPLACE: {token}: {summary}")
                db[token]['summary'] = summary

    ignore = set()
    if ignore_fn:
        with open(ignore_fn) as file:
            data = json.load(file)

        ignore |= {w for w, v in data.items() if v is None}
        synonyms = {v: d for v, d in data.items() if d is not None}
        for token, syns in list(synonyms.items()):
            if token in db:
                synonyms = db[token].get('synonyms', [])
                db[token]['synonyms'] = list(set(synonyms + syns))

    NLP.update_spellchecker({v: 1 for v in db})
    return db, ignore


class SearchEngine:
    def __init__(self, db_fn, vocab_fn=None, ignore_fn=None):
        self.vocab, self.ignore = load_db(db_fn, vocab_fn, ignore_fn)

        NLP.update_spellchecker({v: 1 for v in self.vocab})

        entities = {ent for ent, data in self.vocab.items() if data.get('is_entity', False)}
        self.entity_syns = {
            syn: ent for ent, data in self.vocab.items()
            for syn in data.get('synonyms', [])
            if not data.get('is_entity', False) and syn not in entities
        }
        self.entity_syns.update({ent: ent for ent, data in self.vocab.items()})
        self.entity_syns.update({
            syn: ent for ent, data in self.vocab.items()
            for syn in data.get('synonyms', [])
            if data.get('is_entity', False) and syn not in entities
        })
        self.entity_syns.update({data['label']: ent for ent, data in self.vocab.items() if data.get('is_entity', False)})
        self.entity_names = {ent for syn, ent in self.entity_syns.items()}
        # print(self.entity_names)
        # print(self.entity_syns)

        self.entities = [(ent, data) for ent, data in self.vocab.items() if data.get('is_entity', False)]
        self.entity_lookup = {data['label']: data for ent, data in self.vocab.items() if data.get('is_entity', False)}
        if self.entities[0][1].get('embedding', None):
            self.entity_embeddings = torch.stack([torch.FloatTensor(data['embedding']) if 'embedding' in data else [] for ent, data in self.entities])
        if self.entities[0][1].get('summaryEmbedding', None):
            self.entity_summary_embeddings = torch.stack([torch.FloatTensor(data['summaryEmbedding']) if 'summaryEmbedding' in data else [] for ent, data in self.entities])

        self.tok2id = {tok: n for n, tok in enumerate(sorted(self.entity_names))}
        self.id2tok = {n: tok for tok, n in self.tok2id.items()}
        # print(f"ID2TOK: {self.id2tok}")

        # After id2tok is computed, because otherwise id2tok might be associated with a synonym and not the main lemma.
        for word, data in list(self.vocab.items()):
            data['synonyms'] = list({tok for tok in data.get('synonyms', []) if self.is_valid_token(tok, is_new=True)})
            # synonyms = [tok for syn in data.get('synonyms', []) for tok in join_blocks(NLP.normalize(syn), sep='_')]
            # data['summary'] = list({tok for tok in data.get('summary', []) + synonyms if self.is_valid_token(tok)})
            data['summary'] = list({tok for tok in data.get('summary', []) if self.is_valid_token(tok)})
            self.tok2id.update({tok: self.tok2id[self.entity_syns.get(word, word)] for tok in data['synonyms'] if tok not in entities})
            self.vocab.update({tok: self.vocab[self.entity_syns.get(word, word)] for tok in data['synonyms'] if tok not in entities})

        # print(f"ID2TOK: {self.id2tok}")
        self.entity_multinomial = []
        for ent, data in self.entities:
            self.entity_multinomial.append(NLP.to_multinomial(data.get('summary', set()), self.tok2id))
        self.entity_multinomial = torch.stack(self.entity_multinomial)
        # print(len(self.entities))

    def get_ibm_entities(self):
        return [[ent] + data.get('synonyms', []) for ent, data in self.entities]

    def normalize(self, token):
        if self.is_valid_token(token):
            return self.id2tok.get(self.tok2id.get(token, None), None)
        return None

    def is_valid_token(self, token, is_new=False):
        return len(token) >= 3 and token not in self.ignore

    def search(self, sentence, nbest=4, summarized=False, multinomial=False, description_type=DescriptionType.DEFAULT, reuse_description=True, fuzzy=True, use_alts=False, max_ngram=5):
        literal_entity = sentence
        sentence = clean_spaces(sentence)
        # print(f"SENT: {sentence}")

        # exact match
        if sentence in self.entity_lookup:
            return {
                'tokens': [sentence],
                'description': sentence,
                'nbests': [{
                    'entity': literal_entity,
                    'score': 1,
                    'description': sentence,
                }],
            }

        embedding = None
        literal_seq = []
        if summarized:
            seq, desc = NLP.describe(sentence, self.vocab, description_type=description_type, fuzzy=fuzzy, max_ngram=max_ngram)
            # print(f"DESC: {seq} {desc}")
            seq = NLP.get_tokens(sentence, self.vocab, description_type, fuzzy=fuzzy, max_ngram=max_ngram)
            # print(f"SEQ: {seq}")
            seq = {self.normalize(tok) for tok in seq}
            seq = {tok for tok in seq if tok}
            # print(f"DESC: {seq}")
            desc = join_blocks([w for token in seq for w in token.split('_')])
            if multinomial:
                literal_seq = [tok for tok in NLP.get_tokens(sentence, self.vocab, DescriptionType.DEFAULT, fuzzy=fuzzy, max_ngram=max_ngram) if self.is_valid_token(tok)]
                print(f"LITERAL: {literal_seq}")
                # print(f"LITERAL: {literal_seq} {self.vocab.get(literal_seq[0], {})}")
                if use_alts:
                    seq_non_entities = [tok for tok in seq if not self.vocab.get(tok, {}).get('is_entity', False)]
                    alts = {
                        self.normalize(alt)
                        for tok in seq_non_entities
                        for alt in self.vocab.get(tok, {}).get('summary', []) + seq_non_entities
                    }
                    seq |= alts

                elif len(literal_seq) == 1 and self.vocab.get(literal_seq[0], {}).get('is_entity', False):
                    token = self.normalize(literal_seq[0])
                    # print(f"LITERAL: {literal_seq} {token}")
                    seq = self.vocab.get(token, {}).get('summary', [])

                embedding = NLP.to_multinomial(seq, self.tok2id)
                seq = list(seq)
            else:
                entry = NLP.summarizedb_entry({'label': sentence}, self.vocab, description_type=description_type, reuse_description=reuse_description)
                seq = entry['summary']
                embedding = torch.FloatTensor(entry['summaryEmbedding'])
        else:
            seq, desc = NLP.describe(sentence, self.vocab, max_ngram=max_ngram)
        desc = f"{sentence}. {desc}"
        print(f"SEARCH: {sentence}: {seq}: {desc}")


        result = self.literal_search(desc, nbest, embedding=embedding, multinomial=multinomial)
        if len(result) == 0 and not use_alts:
            return self.search(sentence, nbest=nbest, summarized=summarized, multinomial=multinomial, description_type=description_type, reuse_description=reuse_description, use_alts=True)

        return {
            'tokens': seq,
            'description': desc,
            'nbests': result,
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

        if is_summary:
            mult = torch.mul(embedding, entity_embeddings).sum(dim=1)
            entity_coverage = mult / entity_embeddings.sum(dim=1)
            scores = (mult + entity_coverage) / (embedding.sum() + 1.0)
            # scores = mult / embedding.sum()
        else:
            scores = cos(entity_embeddings, embedding)

        index_sorted = torch.argsort(scores)
        top_scores = reversed(index_sorted[-nbest:])
        # print([(scores[i], self.entities[i][1]['label']) for i in index_sorted])

        if scores[top_scores[0]] >= 1.0:
            top_scores = [top_scores[0]]

        return [
            {
                'entity': self.entities[i][1]['label'],
                'score': scores[i].item(),
                'description': self.entities[i][1]['summary' if is_summary else 'description'],
            }
            for i in top_scores if scores[i].item() > 0
        ]
