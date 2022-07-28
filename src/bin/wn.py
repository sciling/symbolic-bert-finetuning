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

from collections import Counter
from collections import OrderedDict
from collections import defaultdict
from itertools import product
from pathlib import Path

import nltk
import ruamel.yaml
import typer

from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Synset  # pylint: disable=no-name-in-module
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from ruamel.yaml import YAML
from ruamel.yaml.representer import RoundTripRepresenter


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
beverage_hyps = {"beverage.n.01"}


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


def get_syns(word):
    try:
        if isinstance(word, str):
            syns = [wn.synset(word.lower())]
        else:
            syns = [word]
    except ValueError:
        syns = wn.synsets(word.lower(), lang="spa")

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
                hyps = [f"{lemma.name()}({syn.name()})" for s in hyp.hyponyms() for lemma in get_lemmas(s)]
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

    res = {}
    for syn in syns:
        hypos = list(syn.closure(lambda s: s.hyponyms()))
        res.update(res)
        hypos = [f"{lemma.name()}({h.name()})" for h in hypos for lemma in h.lemmas(lang="spa") if not find_hypos(h) and find_foods_syns(h)]
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
            syn = [s for s in wn.synsets(ngram, lang="spa") if find_foods_syns(s)]
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
def get_food_alternatives(foods_fn: Path, default_foods_fn: None, do_hypos: bool = False):
    nltk.download("omw")

    yaml = YAML(typ="safe")  # default, if not specfied, is 'rt' (round-trip)
    with open(foods_fn) as file:
        foods = yaml.load(file)

    doc = {}
    for sentence in foods:
        parts = sentence.split(", ")
        prefs = [" ".join(parts[0:n]) for n in range(1, len(parts) + 1)]
        # typer.echo(f"{sentence} -> {parts} -> {prefs}")
        alt = {}
        all_seqs = []
        doc[sentence] = alt
        alts = set()

        for pref in prefs:
            seqs = get_ngram_syns(pref)
            all_seqs.append(seqs)
            expanded = [{a for w in seq for a in expand_alternatives(w, do_hypos)} for seq in seqs]
            gen = generate_alternatives(expanded)
            alts.update(gen)
            # typer.echo(f"  {pref} -> {seqs} -> {expanded} -> {gen}")
        typer.echo(f"{sentence} -> {alts}")
        alt["synsets"] = [[[s.name() if isinstance(s, Synset) else s for s in seq] for seq in seqs] for seqs in all_seqs]
        alt["definitions"] = {s.name(): s.definition() for seqs in all_seqs for seq in seqs for s in seq if isinstance(s, Synset)}
        alt["alternatives"] = list(alts)

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
def get_confusing_foods(foods_fn: Path, show_definition: bool = True, show_lemmas: bool = True, show_alternatives: bool = True, prepare_defaults: bool = False):
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
        print(all_ngrams, synsets)
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
            if show_definition:
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


if __name__ == "__main__":
    app()
