#! /usr/bin/env python
# Based on https://raw.githubusercontent.com/data-henrik/watson-conversation-tool/master/watoolV2.py
# Copyright 2017-2018 IBM Corp. All Rights Reserved.
# See LICENSE for details.
#
# Author: Henrik Loeser
#
# Converse with your assistant based on IBM Watson Assistant service on IBM Cloud.
# See the README for documentation.
#

import warnings

import csv
import json
from collections import Counter
from collections import OrderedDict
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import List
from functools import lru_cache

import nltk
import ruamel.yaml
import typer
import spacy_stanza

from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Synset  # pylint: disable=no-name-in-module
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from inflector import Inflector, Spanish
from ruamel.yaml import YAML
from ruamel.yaml.representer import RoundTripRepresenter
from medp.utils import NLP


warnings.filterwarnings("ignore")

app = typer.Typer()


# https://stackoverflow.com/a/53875283
class MyRepresenter(RoundTripRepresenter):
    pass


ruamel.yaml.add_representer(OrderedDict, MyRepresenter.represent_dict, representer=MyRepresenter)


food_hyps = {"food.n.01", "food.n.02", "meal.n.01", "animal.n.01", "fungus.n.01", "plant.n.01", "plant.n.02"}
food_hyps.update({s.name() for s in wn.all_synsets("n") if s.name().startswith("edible_")})
# we don't want oil.n.01 because that's also used for petroleum
food_hyps.update({s.name() for s in wn.all_synsets("n") if "_oil" in s.name()})

aliases = {
    'food': food_hyps,
    'water': {
        "drinking_water.n.01", "bottled_water.n.01", "mineral_water.n.01", "mineral_water.n.01", "soda_water.n.01", "soda_water.n.01", "soda_water.n.01", "seltzer.n.01",
        "tea.n.01", "tea.n.05", "mate.n.09",
    },
    'other_drinks': {"soft_drink.n.01", 'alcohol.n.01'},
    'sports': {"sport.n.01"},
    'feelings': {"feeling.n.01", "temporary_state.n.01"},
}


@app.command()
def explain(word: str):
    typer.echo(f"Processing {word}")
    typer.echo("DEFINITIONS")
    typer.echo("-----")
    find_defs(word, True)
    typer.echo("")
    typer.echo("HYPERNYMS")
    typer.echo("-----")
    find_hyps(word, True)
    typer.echo("")
    typer.echo("SYNONYMS")
    typer.echo("-----")
    find_syns(word, True)
    typer.echo("")
    typer.echo("HYPONYMS")
    typer.echo("-----")
    find_hypos(word, True)
    typer.echo("")
    typer.echo("SIBLINGS")
    typer.echo("-----")
    find_siblings(word, True)
    typer.echo("")
    typer.echo("FOOD HYPONIMS")
    typer.echo("-----")
    find_food_hypos(word, True)


def get_lemmas(syn, lang="spa"):
    return syn.lemmas(lang=lang)


class LazyNlp:
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


@lru_cache(maxsize=None)
def get_syns(word):
    try:
        if isinstance(word, str):
            syns = [wn.synset(word.lower())]
        else:
            syns = [word]
    except ValueError:
        singular = ' '.join([w.lemma_ for w in LazyNlp.nlp(word)])
        if singular == word:
            singular = ' '.join([LazyNlp.singularize(w.text) for w in LazyNlp.nlp(word)])
        syns = list(set(wn.synsets(word.lower(), lang="spa") + wn.synsets(singular.lower(), lang="spa")))
        # print(f"SING: {word} -> {singular} -> {syns}")

    return syns


def find_defs(word, log=False):
    syns = get_syns(word)

    res = []
    for syn in syns:
        res.append(syn.definition())
        if log:
            typer.echo(f"{syn} -> {syn.definition()}")
    return res


def find_siblings(word, log=False):
    syns = get_syns(word)
    res = []
    for syn in syns:
        for hyp in syn.hypernyms():
            hyps = [lemma.name() for s in hyp.hyponyms() for lemma in get_lemmas(s)]
            res.extend(hyps)
            if log:
                hyps = [f"{lemma.name()}({s.name()})" for s in hyp.hyponyms() for lemma in get_lemmas(s)]
                typer.echo(f"{hyp} -> {hyps}")
    return res


def find_syns(word, log=False):
    syns = get_syns(word)

    res = []
    for syn in syns:
        res.extend([lemma.name() for lemma in get_lemmas(syn)])

        if log:
            hyps = [f"{lemma.name()}({syn.name()})" for lemma in get_lemmas(syn)]
            typer.echo(f"{syn} -> {hyps}")

    return res


def find_food_syns(word, log=False):
    syns = get_syns(word)

    res = []
    for syn in syns:
        if not find_foods_syns(syn):
            continue

        res.extend([lemma.name() for lemma in get_lemmas(syn)])

        if log:
            hyps = [f"{lemma.name()}({syn.name()})" for lemma in get_lemmas(syn)]
            typer.echo(f"{syn} -> {hyps}")

    return res


def find_hyps(word, log=False):
    syns = get_syns(word)

    res = []
    for syn in syns:
        hyps = list(syn.closure(lambda s: s.hypernyms()))
        res.extend(hyps)
        if log:
            hyps = [f"{lemma.name()}({h.name()})" for h in hyps for lemma in h.lemmas(lang="spa")]
            typer.echo(f"{syn} -> {hyps}")
    return res


def find_hypos(word, log=False):
    syns = get_syns(word)

    res = []
    for syn in syns:
        hypos = list(syn.closure(lambda s: s.hyponyms()))
        res.extend(hypos)
        if log:
            hypos = [f"{lemma.name()}({h.name()})" for h in hypos for lemma in h.lemmas(lang="spa")]
            typer.echo(f"{syn} -> {hypos}")
    return res


def find_food_hypos(word, log=False):
    syns = get_syns(word)

    res = set()
    for syn in syns:
        hypos = list(syn.closure(lambda s: s.hyponyms()))
        res.update(hypos)
        hypos = sorted({f"{lemma.name()}({h.name()})" for h in hypos for lemma in h.lemmas(lang="spa") if not find_hypos(h) and find_foods_syns(h)})
        if log:
            typer.echo(f"{syn} -> {hypos}")
    return res


def find_foods_syns(word):
    syns = get_syns(word)
    news = set()
    hypss = Counter()

    for syn in syns:
        hyps = list(syn.closure(lambda s: s.hypernyms()))
        hypss.update(hyps)
        for hyp in hyps:
            if hyp.name() in food_hyps:
                news.add(hyp)

    # typer.echo(f"{word} -> {hypss}")
    return news


def is_food(syn):
    hyps = list(syn.closure(lambda s: s.hypernyms()))
    for hyp in hyps:
        if hyp.name() in food_hyps:
            return True

    return False


@app.command()
def get_food_words(foods_fn: Path):
    typer.echo(f"Processing {foods_fn}")
    typer.echo(f"Food hyps {food_hyps}")
    typer.echo()

    nltk.download("omw")
    stopword = nltk.corpus.stopwords.words("spanish")

    yaml = YAML(typ="safe")  # default, if not specfied, is 'rt' (round-trip)
    with open(foods_fn) as file:
        foods = yaml.load(file)

    # typer.echo(foods)

    food_words = set()
    for sentence in foods:
        words = [w for w in word_tokenize(sentence.lower()) if w.isalpha()]
        fwords = []
        desc = []
        syns = set()
        seqs = words + ["_".join(n) for n in ngrams(words, 2)] + ["_".join(n) for n in ngrams(words, 3)]
        for word in seqs:
            if word in stopword:
                continue

            hyps = find_foods_syns(word)
            if word not in stopword and hyps:
                syns.update(find_food_syns(word))
                food_words.add(word.lower())
                fwords.append(word.lower())
                desc.extend([h.definition() for h in hyps])
        typer.echo(f">>> {sentence} -> ngrams:{seqs} -> food ngrams:{fwords}")
        typer.echo(f">>> {'. '.join(desc)}")
        typer.echo(f">>> {', '.join(syns)}")
        typer.echo()

    typer.echo(food_words)


ignore_ngrams = {"entero"}


def get_ngram_syns(sentence, max_ngram=5):
    sentence = word_tokenize(sentence.replace(",", "").lower())
    n_words = len(sentence)
    seq = []

    i = 0
    while i < n_words:
        for n in range(max_ngram, 0, -1):
            ngram = "_".join(sentence[i : i + n])
            if ngram in ignore_ngrams:
                continue
            syn = [s for s in get_syns(ngram) if find_foods_syns(s)]
            if syn:
                advance = n
                break

        if not syn:
            syn = [sentence[i]]
            advance = 1
            n = 0

        # typer.echo(f"{n}-gram({i}+{advance}): {ngram} -> {syn}")
        i += advance
        seq.append(syn)
    return seq


def expand_alternatives(syn, do_hypos=False):
    if isinstance(syn, str):
        return {syn}

    # typer.echo(f"ALT: {syn} -> {find_syns(syn)} -> {find_hypos(syn)}")
    alternatives = {a.replace("_", " ") for a in find_syns(syn)}
    if do_hypos:
        alternatives.update({a.replace("_", " ") for h in find_hypos(syn) for a in find_syns(h)})
    return alternatives


def generate_alternatives(alts):
    return set(map(" ".join, product(*alts)))


@app.command()
def get_food_alternatives(
    foods_fn: Path, default_foods_fn: Path = typer.Argument(None), do_hypos: bool = False,
    show_synsets: bool = False, show_definitions: bool = True, show_alternatives: bool = True,
    export_ibm: bool = False, export_csv_fn: Path = typer.Option(None)
):

    nltk.download("omw")

    yaml = YAML(typ="safe")  # default, if not specfied, is 'rt' (round-trip)
    with open(foods_fn) as file:
        foods = yaml.load(file)

    doc = {}
    for sentence in foods:
        alt = {}
        all_seqs = []
        doc[sentence] = alt

        parts = sentence.split(", ")
        prefs = [" ".join(parts[0:n]) for n in range(1, len(parts) + 1)]
        # typer.echo(f"{sentence} -> {parts} -> {prefs}")
        alts = set()

        for pref in prefs:
            seqs = get_ngram_syns(pref)
            all_seqs.append(seqs)
            expanded = [{a for w in seq for a in expand_alternatives(w, do_hypos)} for seq in seqs]
            gen = generate_alternatives(expanded)
            alts.update(gen)
            # typer.echo(f"  {pref} -> {seqs} -> {expanded} -> {gen}")
        typer.echo(f"{sentence} -> {alts}")
        if show_synsets:
            alt["synsets"] = [[[s.name() if isinstance(s, Synset) else s for s in seq] for seq in seqs] for seqs in all_seqs]
        if show_definitions:
            alt["definitions"] = {s.name(): s.definition() for seqs in all_seqs for seq in seqs for s in seq if isinstance(s, Synset)}
        if show_alternatives:
            alt["alternatives"] = list(alts)

    default_foods = None
    if default_foods_fn:
        yaml = YAML(typ="safe")  # default, if not specfied, is 'rt' (round-trip)
        with open(default_foods_fn) as file:
            default_foods = yaml.load(file)

        for synname, sentence in default_foods.items():
            syn = wn.synset(synname)
            alts = sorted({lemma.name().replace('_', ' ') for hs in find_food_hypos(syn) for lemma in get_lemmas(hs)})
            typer.echo(f"DEF: {syn} -> {sentence} -> {alts}")
            if 'hyponyms' in doc[sentence]:
                doc[sentence]['hyponyms'] = sorted(set(alts + doc[sentence]['hyponyms']))
            else:
                doc[sentence]['hyponyms'] = alts

    if export_ibm:
        with open("food_alts.csv", "w") as file:
            csvwriter = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for food, data in doc.items():
                csvwriter.writerow(['alimento_tipo', food] + list(set(data.get('alternatives', []) + data.get('hyponyms', []))))

    elif export_csv_fn:
        with open(export_csv_fn, "w") as file:
            csvwriter = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for food, data in doc.items():
                csvwriter.writerow([food, '. '.join(data.get('definitions', {}).values())] + list(set(data.get('alternatives', []))))

    else:
        with open("food_alts.yaml", "w") as file:
            yaml = YAML()
            yaml.Representer = MyRepresenter
            yaml.indent(mapping=2, sequence=4, offset=2)
            yaml.dump(doc, file)


def get_all_ngram_syns(sentence, max_ngram=5):
    sentence = word_tokenize(sentence.replace(",", "").lower())
    ngs = []
    for n in range(1, max_ngram + 1):
        all_ngrams = ngrams(sentence, n)
        for ngram in all_ngrams:
            ngram = "_".join(ngram)
            if ngram in ignore_ngrams:
                continue

            syns = get_syns(ngram)
            if syns:
                ngs.extend(syns)
    return ngs


@app.command()
def get_confusing_foods(foods_fn: Path, show_definitions: bool = True, show_lemmas: bool = True, show_alternatives: bool = True, prepare_defaults: bool = False):
    nltk.download("omw")

    yaml = YAML(typ="safe")  # default, if not specfied, is 'rt' (round-trip)
    with open(foods_fn) as file:
        foods = yaml.load(file)

    confusions = defaultdict(set)
    for sentence in foods:
        parts = sentence.split(", ")
        prefs = [" ".join(parts[0:n]) for n in range(1, len(parts) + 1)]

        all_ngrams = []
        for pref in prefs:
            ngrs = get_all_ngram_syns(pref)
            all_ngrams.extend(ngrs)
        synsets = {s for s in all_ngrams if isinstance(s, Synset) and find_hypos(s)}
        for syn in synsets:
            confusions[syn].add(sentence)
        typer.echo(f"{sentence} -> {synsets}")

    doc = {}
    typer.echo(f"CONFUSIONS: {confusions}")
    for syn, confs in sorted(confusions.items()):
        if not is_food(syn):
            continue
        alternatives = list({lemma.name().replace("_", " ") for s in find_hypos(syn) for lemma in get_lemmas(s)})
        confs = sorted(confs)
        if prepare_defaults and len(confs) == 1:
            doc[syn.name()] = confs[0]
        else:
            doc[syn.name()] = {
                "foods": confs,
            }
            if show_definitions:
                doc[syn.name()]["definition"] = syn.definition()
            if show_lemmas:
                doc[syn.name()]["lemmas"] = sorted(find_syns(syn))
            if show_alternatives:
                doc[syn.name()]["alternatives"] = sorted(alternatives)

    with open("food_conf.yaml", "w") as file:
        yaml = YAML()
        yaml.Representer = MyRepresenter
        yaml.indent(mapping=2, sequence=4, offset=2)
        yaml.dump(doc, file)


@app.command()
def get_synsets_with_prefix(prefix: str):
    typer.echo(f"Processing {prefix}")
    nltk.download("omw")

    syns = [syn for syn in wn.all_synsets(lang="spa") if syn.name().startswith(prefix)]
    typer.echo(syns)


@app.command()
def get_synsets_with_suffix(suffix: str):
    typer.echo(f"Processing {suffix}")
    nltk.download("omw")

    syns = [syn for syn in wn.all_synsets(lang="spa") if suffix in syn.name()]
    typer.echo(syns)


@app.command()
def find_all(synnames: List[str], export_csv_fn: Path = typer.Option(None)):
    typer.echo(f"Processing {synnames}")
    nltk.download("omw")

    parent_syns = set()
    for synname in synnames:
        if synname in aliases:
            parent_syns.update({wn.synset(synname) for synname in aliases[synname]})
        else:
            parent_syns.add(wn.synset(synname))

    typer.echo(f"parent_syns: {parent_syns}")
    syns = sorted([syn for syn in wn.all_synsets(lang="spa") if any([parent_syn in find_hyps(syn) for parent_syn in parent_syns])])
    for syn in syns:
        typer.echo(f"{syn}: {syn.definition()} {sorted({lm.name() for lm in get_lemmas(syn)})} <- {find_hyps(syn)}")

    if export_csv_fn:
        with open(export_csv_fn, "w") as file:
            csvwriter = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for syn in syns:
                typer.echo(f"{syn}: {syn.definition()} {sorted({lm.name() for lm in get_lemmas(syn)})} <- {find_hyps(syn)}")
                row = [syn.name(), syn.definition()] + sorted({lm.name() for lm in get_lemmas(syn)})
                csvwriter.writerow(row)


@app.command()
def describe(sentence: str, vocab_fns: List[Path]):
    vocab = {}

    for vocab_fn in vocab_fns:
        NLP.update_vocab(vocab, vocab_fn)

    with open('cached-vocab.json', 'w') as file:
        json.dump(vocab, file, indent=2, ensure_ascii=False)

    print(NLP.describe(sentence, vocab))


if __name__ == "__main__":
    app()
