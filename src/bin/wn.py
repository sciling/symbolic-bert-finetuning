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

import os
import re
import sys
import csv
import json
import random
from collections import Counter, OrderedDict, defaultdict
from itertools import product, islice
from pathlib import Path
from typing import List

from tqdm import tqdm
import nltk
import ruamel.yaml
import typer
import pandas as pd
import numpy as np

import spacy
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk.corpus.reader.wordnet import Synset  # pylint: disable=no-name-in-module
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from ruamel.yaml import YAML
from ruamel.yaml.representer import RoundTripRepresenter
from medp.utils import (
    SearchEngine, NLP, EmbeddingsProcessor, DescriptionType,
    get_syns, expand_template, clean_spaces, load_db, random_combinations,
)


warnings.filterwarnings("ignore")

app = typer.Typer()


# https://stackoverflow.com/a/53875283
class MyRepresenter(RoundTripRepresenter):
    pass


ruamel.yaml.add_representer(OrderedDict, MyRepresenter.represent_dict, representer=MyRepresenter)


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

food_hyps = {"food.n.01", "food.n.02", "meal.n.01", "animal.n.01", "fungus.n.01", "plant.n.02"}
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
        if not find_relevant_syns(syn):
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
        hypos = sorted({f"{lemma.name()}({h.name()})" for h in hypos for lemma in h.lemmas(lang="spa") if not find_hypos(h) and find_relevant_syns(h)})
        if log:
            typer.echo(f"{syn} -> {hypos}")
    return res


def find_relevant_syns(word, parent_hyps=food_hyps):
    syns = get_syns(word)
    news = set()
    hypss = Counter()

    for syn in syns:
        hyps = list(syn.closure(lambda s: s.hypernyms()))
        hypss.update(hyps)
        for hyp in hyps:
            if hyp.name() in parent_hyps:
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

            hyps = find_relevant_syns(word)
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


def get_ngram_syns_pairs(sentence, max_ngram=5):
    sentence = word_tokenize(sentence.replace(",", "").lower())
    n_words = len(sentence)
    seq = []

    i = 0
    while i < n_words:
        for n in range(max_ngram, 0, -1):
            ngram = "_".join(sentence[i : i + n])
            if ngram in ignore_ngrams:
                continue
            syn = [s for s in get_syns(ngram) if find_relevant_syns(s)]
            if syn:
                advance = n
                break

        if not syn:
            ngram = sentence[i]
            syn = [ngram]
            advance = 1
            n = 0

        # typer.echo(f"{n}-gram({i}+{advance}): {ngram} -> {syn}")
        i += advance
        seq.append((ngram.replace('_', ' '), syn))
    return seq


def get_ngram_syns(sentence, max_ngram=5):
    return [syn for _, syn in get_ngram_syns_pairs(sentence, max_ngram)]


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
    foods_fn: Path, default_foods_fn: Path = typer.Argument(None),
    extend_fn: Path = typer.Option(None), do_hypos: bool = False,
    show_synsets: bool = False, show_definitions: bool = True, show_alternatives: bool = True,
    export_ibm_fn: Path = typer.Option(None), export_csv_fn: Path = typer.Option(None),
    use_definitions: bool = False, entity_name: str = 'alimento_tipo'
):

    nltk.download("omw")

    seen = set()
    doc = {}
    foods = set()

    if extend_fn:
        with open(extend_fn) as file:
            csvreader = csv.reader(file, delimiter=',', quotechar='"')
            for row in csvreader:
                entity_name = row[0]
                foods.add(row[1])
                doc[row[1]] = {"alternatives": {r for r in row[2:] if r not in seen}}
                seen.update(row[2:])

    new_entries = 0
    if foods_fn.suffix in ('.csv', ):
        with open(foods_fn) as file:
            csvreader = csv.reader(file, delimiter=',', quotechar='"')
            for row in csvreader:
                entity_name = row[0]
                foods.add(row[1])
                alts = {r for r in row[2:] if r not in seen}
                seen.update(row[2:])

                if row[1] in doc:
                    doc[row[1]]['alternatives'] |= alts
                else:
                    new_entries += 1
                    print(f"NEW: '{row[1]}'")
                    doc[row[1]] = {"alternatives": alts}

    else:
        yaml = YAML(typ="safe")  # default, if not specfied, is 'rt' (round-trip)
        with open(foods_fn) as file:
            foods = yaml.load(file)
            doc = {food: {"alternatives": set([food])} for food in foods}

    print(f"NEW ENTRIES: {new_entries}")

    for food in foods:
        all_seqs = []
        alt = doc[food]

        parts = food.split(", ")
        prefs = [" ".join(parts[0:n]) for n in range(1, len(parts) + 1)]
        # typer.echo(f"{food} -> {parts} -> {prefs}")
        alts = set()

        for pref in prefs:
            seqs = get_ngram_syns(pref)
            all_seqs.append(seqs)
            expanded = [{a for w in seq for a in expand_alternatives(w, do_hypos)} for seq in seqs]
            gen = generate_alternatives(expanded)
            alts.update(gen)
            # typer.echo(f"  {pref} -> {seqs} -> {expanded} -> {gen}")
        typer.echo(f"{food} -> {alts}")
        if show_synsets:
            alt["synsets"] = [[[s.name() if isinstance(s, Synset) else s for s in seq] for seq in seqs] for seqs in all_seqs]
        if show_definitions:
            alt["definitions"] = {s.name(): s.definition() for seqs in all_seqs for seq in seqs for s in seq if isinstance(s, Synset)}
        if show_alternatives:
            alt["alternatives"].update({alt for alt in alts if alt not in seen})
        seen.update(alts)

    default_foods = None
    if default_foods_fn:
        yaml = YAML(typ="safe")  # default, if not specfied, is 'rt' (round-trip)
        with open(default_foods_fn) as file:
            default_foods = yaml.load(file)

        for synname, food in default_foods.items():
            syn = wn.synset(synname)
            alts = sorted({lemma.name().replace('_', ' ') for hs in find_food_hypos(syn) for lemma in get_lemmas(hs)})

            if food in doc:
                doc[food]["alternatives"].update({alt for alt in alts if alt not in seen})
                seen.update(alts)

                typer.echo(f"DEF: {syn} -> {food} -> {alts}")
                if 'hyponyms' in doc[food]:
                    doc[food]['hyponyms'] = sorted(set(alts + doc[food]['hyponyms']))
                else:
                    doc[food]['hyponyms'] = alts
            else:
                print(f"WARNING: default '{synname}' -> '{food}' not a valid entity")

    if export_ibm_fn:
        with open(export_ibm_fn, "w") as file:
            csvwriter = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for food, data in doc.items():
                syns = [syn for syn in set(list(data.get('alternatives', [])) + list(data.get('hyponyms', []))) if len(syn) <= 64]
                if len(syns) == 0:
                    syns = [food[0:64]]
                csvwriter.writerow([entity_name, food] + sorted(syns))

    elif export_csv_fn:
        with open(export_csv_fn, "w") as file:
            csvwriter = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for food, data in doc.items():
                if use_definitions:
                    definition = '. '.join(data.get('definitions', {}).values())
                else:
                    definition = ''
                csvwriter.writerow([food, definition] + list(set(data.get('alternatives', []))))

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
    parts_re = re.compile(r"([,:/]\s*|\s+[yio]\s+)")
    for sentence in foods:
        parts = parts_re.split(sentence)
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
def get_wordnet_synonyms(
    entity_fn: Path,
    export_csv_fn: Path = typer.Option(None),
    parent_hyps: str = typer.Option(None)
):
    nltk.download("omw")

    if parent_hyps:
        parent_hyps = set(','.split())
    else:
        parent_hyps = food_hyps

    entity_name = None
    doc = {}
    seen = set()
    entities = set()

    if entity_fn:
        with open(entity_fn) as file:
            csvreader = csv.reader(file, delimiter=',', quotechar='"')
            for nrow, row in enumerate(csvreader):
                if nrow == 0:
                    entity_name = row[0]

                entities.add(row[1])
                doc[row[1]] = {"synonyms": {r for r in row[2:] if r not in seen}}
                seen.update(row[2:])

    if not entity_name:
        typer.echo(f"Could not find entity name in {entity_fn}")
        raise typer.Exit()

    syns = defaultdict(set)
    seen = set()
    parts_re = re.compile(r"([,:/]\s*|\s+[yio]\s+)")
    with tqdm(doc.items()) as pbar:
        for entity, data in pbar:
            for name in tqdm([entity] + list(data.get('synonyms', [])), leave=False):
                parts = parts_re.split(name)
                prefs = [" ".join(parts[0:n]) for n in range(1, len(parts) + 1)]
                pbar.set_description(f"{name} -> {parts} -> {prefs}")

                for pref in tqdm(prefs, leave=False):
                    pbar.set_description(f"{entity} -> {pref}")
                    seqs = get_ngram_syns_pairs(pref)
                    for ngram, seq in seqs:
                        ngram_syns = {syn for syn in seq if isinstance(syn, Synset) and syn not in seen}
                        seen.update(ngram_syns)
                        syns[ngram].update(ngram_syns)

                    syns.update()
                    # print(f"PREF: {pref} {seqs} {syns}")

    tokens = {}
    for ngram, syn in syns.items():
        if not syn:
            continue

        lemmas = [lemma.name().replace('_', ' ') for s in syn for lemma in get_lemmas(s)]
        token = ngram
        print(syn, token, lemmas)

        alt = {}
        alt["definitions"] = '. '.join([s.definition() for s in syn])
        alt["synonyms"] = lemmas

        tokens[token] = alt

    if export_csv_fn:
        with open(export_csv_fn, "w") as file:
            csvwriter = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for token, data in sorted(tokens.items()):
                definition = data.get('definitions', '')
                csvwriter.writerow([token, definition] + list(sorted(set(data.get('synonyms', [])))))


def find_food_hypos(word, log=False):
    syns = get_syns(word)

    res = set()
    for syn in syns:
        hypos = list(syn.closure(lambda s: s.hyponyms()))
        res.update(hypos)
        hypos = sorted({f"{lemma.name()}({h.name()})" for h in hypos for lemma in h.lemmas(lang="spa") if not find_hypos(h) and find_relevant_syns(h)})
        if log:
            typer.echo(f"{syn} -> {hypos}")
    return res


@app.command()
def filter_by_wordnet(
    entity: str,
    all_entities_fn: Path,
    vocab_fn: Path,
    export_csv_fn: Path = typer.Option(None),
    parent_hyps: str = typer.Option(None),
    entity_value: str = 'vocabulario'
):
    nltk.download("omw")

    if parent_hyps:
        parent_hyps = set(','.split())
    else:
        parent_hyps = food_hyps

    vocab = set()

    entities = []
    seen = set()
    with open(all_entities_fn) as file:
        csvreader = csv.reader(file, delimiter=',', quotechar='"')
        for row in csvreader:
            if row[0] == entity and row[1] == 'vocabulario':
                continue
            entities.append(row)
            seen.update({clean_spaces(r).lower() for r in row[2:]})

    with open(vocab_fn) as file:
        csvreader = csv.reader(file, delimiter=',', quotechar='"')
        for row in csvreader:
            if not row:
                continue

            word = clean_spaces(row[0]).lower().replace('"', '')

            if len(word) < 3 or word in seen:
                continue

            accept_word = True
            if len(word) < 6:
                hyps = find_relevant_syns(word, parent_hyps)
                if not hyps:
                    # print(f"{word}")
                    accept_word = False
            if accept_word:
                vocab.add(word)

    vocab = sorted(vocab)

    if export_csv_fn:
        with open(export_csv_fn, "w") as file:
            csvwriter = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow([entity, entity_value] + vocab)
    else:
        print('\n'.join(vocab))


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
def create_db(
    definitions_fns: List[Path], entities_fn: Path = typer.Option(None),
    vocab_fn: Path = typer.Option(None), ignore_fn: Path = typer.Option(None),
    save_fn: Path = typer.Option(None),
    redefinitions_fn: Path = typer.Option(None),
    override_definitions: bool = False
):
    vocab = {}
    redefinitions = {}

    if redefinitions_fn:
        with open(redefinitions_fn) as file:
            csvreader = csv.reader(file, delimiter=',', quotechar='"')
            for row in csvreader:
                if len(row) >= 2:
                    redefinitions[clean_spaces(row[0])] = clean_spaces(row[1])

    if entities_fn:
        NLP.update_vocab(vocab, entities_fn, is_entity=True, redefinitions=redefinitions, override_definitions=override_definitions)

    for definition_fn in definitions_fns:
        NLP.update_vocab(vocab, definition_fn, redefinitions=redefinitions, override_definitions=override_definitions)

    load_db(vocab, vocab_fn, ignore_fn)

    for data in tqdm(vocab.values(), desc="Regenerating descriptions..."):
        if not data.get('description', None):
            data['description'] = NLP.describe(data['label'], vocab)
            # data['embedding'] = EmbeddingsProcessor.pages_to_embeddings([data['description']])[0].tolist()

    with open(save_fn, 'w') as file:
        json.dump(vocab, file, indent=2, ensure_ascii=False)


@app.command()
def normalize(words: List[str], fuzzy: bool = True):
    for word in words:
        print(f"NORMALIZE: {word} {NLP.normalize(word)}")


@app.command()
def tokenize(words: List[str], db_fn: Path = typer.Option(None), vocab_fn: Path = typer.Option(None), ignore_fn: Path = typer.Option(None), fuzzy: bool = True, subtokens: bool = False, max_ngram: int = 0):
    db, _ = load_db(db_fn, vocab_fn, ignore_fn)
    for word in words:
        print(f"TOKENIZE: {word} {NLP.tokenize(word, db, fuzzy, subtokens, max_ngram)}")


@app.command()
def describe(sentence: str, db_fn: Path = typer.Option(None), vocab_fn: Path = typer.Option(None), ignore_fn: Path = typer.Option(None), description_type: DescriptionType = DescriptionType.DEFAULT, fuzzy: bool = True, max_ngram: int = 5):
    db, _ = load_db(db_fn, vocab_fn, ignore_fn)

    print(NLP.describe(sentence, db, description_type=description_type, fuzzy=fuzzy, max_ngram=max_ngram))


@app.command()
def summarize(sentence: str, db_fn: Path = typer.Option(None), vocab_fn: Path = typer.Option(None), ignore_fn: Path = typer.Option(None), save_fn: Path = typer.Option(None), description_type: DescriptionType = DescriptionType.DEFAULT, reuse_descriptions: bool = False, max_ngram: int = 5):
    db, _ = load_db(db_fn, vocab_fn, ignore_fn)
    entry = NLP.summarizedb_entry({'label': sentence}, db, description_type=description_type, reuse_description=reuse_descriptions)
    seq = entry['summary']
    print(seq)


@app.command()
def summarize_db(db_fn: Path, vocab_fn: Path = typer.Option(None), ignore_fn: Path = typer.Option(None), save_fn: Path = typer.Option(None), description_type: DescriptionType = DescriptionType.DEFAULT, reuse_descriptions: bool = False, max_ngram: bool = 0):
    db, _ = load_db(db_fn, vocab_fn, ignore_fn)
    new_db = {}
    for text, data in tqdm(db.items()):
        new_db[text] = NLP.summarizedb_entry(data, db, description_type=description_type, reuse_description=reuse_descriptions, max_ngram=max_ngram)

    with open(save_fn, 'w') as file:
        json.dump(new_db, file, indent=2, ensure_ascii=False)


def split_args(line):
    spl = line.split(';')
    return spl[0], set(spl[1:])

@app.command()
def search(_texts: List[str], db_fn: Path = typer.Option(None), vocab_fn: Path = typer.Option(None), ignore_fn: Path = typer.Option(None), nbest: int = 4, summarized: bool = False, multinomial: bool = False, description_type: DescriptionType = DescriptionType.DEFAULT, reuse_description: bool = False, fuzzy: bool = True, max_ngram: int = 5, use_alts: bool = False):
    searcher = SearchEngine(db_fn, vocab_fn=vocab_fn, ignore_fn=ignore_fn)

    texts = []
    for text in _texts:
        if os.path.isfile(text):
            with open(text) as file:
                texts.extend([split_args(line) for line in file.read().split('\n') if clean_spaces(line)])
        else:
            texts.append(split_args(text))

    errors = set()
    for text, labels in tqdm(texts):
        print(f"Searching: '{text}'")
        res = searcher.search(text, nbest, summarized=summarized, multinomial=multinomial, description_type=description_type, reuse_description=reuse_description, fuzzy=fuzzy, max_ngram=max_ngram, use_alts=use_alts)
        resl = {e['entity'] for e in res['nbests']}
        exact = False
        if 'None' in labels:
            exact = True
            labels.remove('None')

        found = all([label in resl for label in labels])
        if exact:
            found = found and all([label in labels for label in resl])
        print(f"FOUND: {found}")
        if not found:
            errors.add(text)

        print(json.dumps(res, indent=2, ensure_ascii=False))

    eprint(f"ERRORS: {errors}")


@app.command()
def search_ibm_export(db_fn: Path = typer.Option(None), ignore_fn: Path = typer.Option(None)):
    searcher = SearchEngine(db_fn, ignore_fn)
    res = searcher.get_ibm_entities()

    print(json.dumps(res, indent=2, ensure_ascii=False))


@app.command()
def literal_search(text: str, db_fn: Path = typer.Option(None), nbest: int = 4):
    searcher = SearchEngine(db_fn)
    res = searcher.literal_search(text, nbest)

    print(json.dumps(res, indent=2, ensure_ascii=False))


@app.command()
def parse_excel(excel_fns: List[str], save_fn: Path = typer.Option(None)):
    all_docs = {}
    for excel_fn in tqdm(excel_fns):
        params = excel_fn.split(':')
        title_idx = int(params[1]) if len(params) > 1 else 2
        desc_idx = int(params[2]) if len(params) > 2 else None
        sheets_dict = pd.read_excel(params[0], sheet_name=None)

        for name, sheet in tqdm(sheets_dict.items(), leave=False):
            # We assume the first column contains a numerical index for all files
            # so we ignore the rows without this index.
            sheet = sheet[pd.to_numeric(sheet[sheet.columns[0]], errors='coerce').notnull()]
            if desc_idx is None:
                data = {str(title): '' for title in sheet.iloc[:, title_idx]}
            else:
                data = {title: desc for title, desc in zip(sheet.iloc[:, title_idx], sheet.iloc[:, desc_idx]) if isinstance(title, str)}
            print(name, sheet.shape, len(data), data.keys())
            # print(name, sheet.shape, len(data), sorted(data.keys()))
            all_docs.update(data)

    if save_fn:
        with open(save_fn, "w") as file:
            csvwriter = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for name, desc in sorted(all_docs.items()):
                row = [name, str(desc).replace(',', '').replace('\n', '. '), name.lower()]
                csvwriter.writerow(row)
    else:
        print(all_docs)


@app.command()
def extract_vocab(entities_fn: Path, save_fn: Path = typer.Option(None)):
    vocab = Counter()
    with open(entities_fn) as file:
        csvreader = csv.reader(file, delimiter=',', quotechar='"')
        for row in csvreader:
            doc = NLP.nlp('. '.join([clean_spaces(r) for r in row]))
            vocab.update([w.lemma_.lower() for w in doc if w.text and len(w.text) > 1 and w.is_alpha and not w.is_punct and not w.is_stop and not w.is_space])
            print(doc, {w.text for w in doc if w.pos_ == 'VERB' and len(w.text) > 1 and w.is_alpha and not w.is_punct and not w.is_stop and not w.is_space})

    if save_fn:
        with open(save_fn, 'w') as file:
            print(vocab)
            json.dump(vocab, file, indent=2, ensure_ascii=False)
    else:
        print(json.dumps(vocab, indent=2, ensure_ascii=False))


@app.command()
def redefine_entities(entities_fn: Path, definitions_fn: List[Path], vocab_fn: Path = typer.Option(None), ignore_fn: Path = typer.Option(None), save_fn: Path = typer.Option(None)):
    vocab = set()
    ignore = set()
    definitions = {}
    if vocab_fn:
        with open(vocab_fn) as file:
            vocab = set(json.load(file))

    if ignore_fn:
        with open(ignore_fn) as file:
            ignore = {w for w, v in json.load(file).items() if v is None}

    for definition_fn in tqdm(definitions_fn):
        print(f"PROCESSING: {definition_fn}")

        with open(definition_fn) as file:
            csvreader = csv.reader(file, delimiter=',', quotechar='"')
            for row in tqdm(csvreader):
                name = '_'.join(NLP.normalize(row[0]))
                if name not in definitions:
                    definitions[name] = row[1]

    entities = []
    with open(entities_fn) as file:
        csvreader = csv.reader(file, delimiter=',', quotechar='"')
        for row in tqdm(csvreader):
            name = '_'.join(NLP.normalize(row[0]))
            if not row[1]:
                row[1] = definitions.get(name, row[0])
            entities.append(row)

    for row in tqdm(entities):
        doc = NLP.nlp('. '.join(list({clean_spaces(r).lower() for r in row})))
        row[1] = ' '.join([
            w.text.lower()
            for w in doc
            if (not vocab or w.lemma_.lower() in vocab) and (not ignore or w.lemma_.lower() not in ignore)
        ])

    if save_fn:
        with open(save_fn, 'w') as file:
            csvwriter = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for row in entities:
                csvwriter.writerow(row)
    else:
        print(json.dumps(sorted(vocab), indent=2, ensure_ascii=False))


@app.command()
def fix_entities(entities_fn: Path, save_fn: Path, max_examples: int = 0):
    with open(entities_fn) as fileread, open(save_fn, "w") as filewrite:
        csvreader = csv.reader(fileread, delimiter=',', quotechar='"')
        csvwriter = csv.writer(filewrite, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in csvreader:
            entity = clean_spaces(row[0])
            value = clean_spaces(row[1])
            if len(value) > 64:
                print(f"WARNING: '{entity}' '{value}' larger then 64 chars (len: {len(value)})")
                continue
            syns = [
                clean_spaces(syn)
                for syn in sorted(set([r.strip().lower() for r in row[2:] if not re.match(r"^\s*$", r)]))
                if len(syn) < 64
            ]
            if max_examples and len(syns) > max_examples:
                syns = random.sample(syns, k=max_examples)
            csvwriter.writerow([entity, value] + syns)


@app.command()
def entities_to_intents(entities_fn: Path, save_fn: Path, max_examples: int = 0):
    with open(entities_fn) as fileread, open(save_fn, "w") as filewrite:
        csvreader = csv.reader(fileread, delimiter=',', quotechar='"')
        csvwriter = csv.writer(filewrite, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in csvreader:
            intent_name = f"{row[0]}_{row[1]}".replace(' ', '_')
            examples = list(set(row[1:]))
            if max_examples and len(examples) > max_examples:
                examples = random.sample(examples, k=max_examples)
            for cell in sorted(examples):
                csvwriter.writerow([clean_spaces(cell), intent_name])


@app.command()
def recombine_intents(intents_fn: Path, save_fn: Path, separators: str = 'y,por,con,a causa de,porque', max_length: int = 3, max_examples: int = 1000):
    with open(intents_fn) as file:
        csvreader = csv.reader(file, delimiter=',', quotechar='"')
        intents = [row for row in csvreader]
        del intents[0]

    random.shuffle(intents)
    separators = separators.split(',')

    with open(save_fn, "w") as file:
        csvwriter = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for size in range(2, max_length + 1):
            for pack in tqdm(random_combinations(intents, size, max_examples), total=max_examples):
                text = [pack[0][0]]
                intent_names = {pack[0][1]}
                for r in pack[1:]:
                    text.append(random.choice(separators))
                    text.append(r[0])
                    intent_names.add(r[1])

                csvwriter.writerow([clean_spaces(' '.join(text)), ';'.join(intent_names)])


@app.command()
def variations(words: List[str]):
    for word in words:
        analysis = NLP.tag(word)
        for number in ('sg', 'pl'):
            tokens = [NLP.get_variations(token, number) for token in analysis]
            print("TOK", analysis, tokens)
            print(number.upper(), generate_alternatives([t for t in tokens]))


@app.command()
def convert_form(words: List[str], depth: int=1, threshold: float=None):
    for word in words:
        conversions = NLP.convert_form_recursive(word, depth=depth, threshold=threshold)
        print(f"CONVERSIONS: {word} {conversions}")


@app.command()
def expand_entities(
        templates_fn: List[Path] = typer.Argument(None), entities_fn: Path = typer.Option(None), vars_fn: List[Path] = typer.Option(None),
        save_fn: Path = typer.Option(None), depth: int = 1, threshold: float = None, max_syns: int = 0, prefix: str = '', entity_values: str = None,
        max_permutations: int = sys.maxsize
):
    seen = {}
    entities = defaultdict(set)
    if entity_values is not None:
        entity_values = entity_values.split(',')

    if entities_fn:
        with open(entities_fn) as file:
            csvreader = csv.reader(file, delimiter=',', quotechar='"')
            entity_rows = [row[:] for row in csvreader]

        for row in entity_rows:
            entity = ':'.join(row[:2])
            for syn in row[1:]:
                if syn in seen:
                    if seen[syn] != entity:
                        print(f"WARNING: '{syn}' in '{entity}' is already a synonym of '{seen[syn]}'")
                    continue
                seen[syn] = entity
                entities[entity].add(syn)

    if templates_fn is None:
        templates_fn = []

    with tqdm(templates_fn, position=0) as pbar:
        for template_fn in pbar:
            print(f"PROCESSING: {template_fn}")
            templates = []
            yaml = YAML(typ="safe")  # default, if not specfied, is 'rt' (round-trip)
            with open(template_fn) as file:
                data = yaml.load(file)
                variables = data.get('variables', {})

                if vars_fn:
                    for var_fn in tqdm(vars_fn, position=1, leave=False):
                        with open(var_fn) as file:
                            csvreader = csv.reader(file, delimiter=',', quotechar='"')
                            varls = defaultdict(set)
                            for row in tqdm(list(csvreader), position=2, leave=False):
                                pbar.set_description(f"VAR_FN: {var_fn}: {row}")
                                varls[row[0]].add(row[1])

                            for vl, vs in tqdm(varls.items(), position=2, leave=False):
                                pbar.set_description(f"VAR_DS: {var_fn}: {vl}")
                                if vl not in variables:
                                    variables[vl] = '|'.join(vs)
                                else:
                                    variables[vl] += '|' + '|'.join(vs)

                for ent, values in tqdm(data.get('entities', {}).items(), position=1, leave=False):
                    pbar.set_description(f"TEMPLATE: {ent}")
                    for val in tqdm(values, position=2, leave=False):
                        if entity_values and val['value'] not in entity_values:
                            continue
                        pbar.set_description(f"TEMPLATE: {ent}: {val['value'][:80]}")
                        entity = f"{ent}:{prefix}{val['value']}"
                        # print(f"VAL: {val}: {variables}")
                        temps = []
                        for temp in tqdm(val.get('templates', []), position=3, leave=False):
                            if isinstance(temp, str):
                                text, params = temp, {}
                            elif isinstance(temp, dict):
                                text, params = temp['template'], temp
                            else:
                                raise Exception(f"Invalid template {temp}")
                            pbar.set_description(f"TEMPLATE: {ent}: {val['value'][:80]}: {text[:80]}")
                            temps.append((text.format(**variables), params))
                        templates.append((entity, temps))

            for entity, elems in tqdm(templates, position=1, leave=False):
                vocab = set()
                pbar.set_description(f"GENERATE: {entity}")
                for template, params in tqdm(elems, position=2, leave=False):
                    # print(f"PROCESSING: {entity} {template[:80]} ...")
                    pbar.set_description(f"GENERATE: {entity}: {template[:80]}")
                    for sentence in tqdm(expand_template(template, max_num=params.get('max', max_permutations)), position=3, leave=False):
                        pbar.set_description(f"GENERATE: {entity}: {template[:80]}: {sentence[:80]}")
                        # print(f"TEMPLATE: {template}: {sentence}")
                        words = {sentence}

                        if False:
                            analysis = NLP.tag(sentence)
                            if len(analysis) == 1:
                                words |= NLP.convert_form_recursive(sentence, depth=depth, threshold=threshold)

                        if False:
                            for word in list(words):
                                subanalysis = NLP.tag(word)
                                # print(f"SUB: {word}: {subanalysis}")
                                for number in ('sg', ):  # 'pl'):
                                    tokens = [
                                        NLP.get_variations(token, number, pretoken)
                                        for pretoken, token in zip([None] + subanalysis, subanalysis)
                                    ]
                                    alts = generate_alternatives([t for t in tokens])
                                    # print(f"ALTS[{number}] = {alts}")
                                    words |= alts

                        for syn in list(words):
                            if seen.get(syn, entity) != entity:
                                # print(f"WARNING: '{syn}' in '{entity}' from template '{template}' is already a synonym of '{seen[syn]}'")
                                print(f"WARNING: '{syn}' in '{entity}' is already a synonym of '{seen[syn]}'")
                                words.remove(syn)
                            else:
                                seen[syn] = entity

                        vocab |= words

                # print(f"ENTITY: {entity} = {vocab}")
                entities[entity] |= vocab

    if save_fn:
        with open(save_fn, 'w') as file:
            csvwriter = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for entity, synonyms in entities.items():
                if max_syns and len(synonyms) > max_syns:
                    synonyms = list(synonyms)
                    random.shuffle(synonyms)
                    synonyms = synonyms[:max_syns]
                csvwriter.writerow(entity.split(':') + sorted(synonyms))
    else:
        print(json.dumps(entities, indent=2, ensure_ascii=False))


@app.command()
def dump_sentiwordnet(fix_fn: Path = typer.Option(None), sel_fn: Path = typer.Option(None), threshold: float = 0.25, save_fn: Path = typer.Option(None)):
    syns = list(swn.all_senti_synsets())
    lemmas = defaultdict(list)
    for ssyn in syns:
        syn = ssyn.synset
        for lemma in get_lemmas(syn):
            lemma = lemma.name().replace('_', ' ')
            classes = ['positive', 'neutral', 'negative']
            scores = [ssyn.pos_score(), ssyn.obj_score(), ssyn.neg_score()]
            ci = np.argmax(scores)

            lemmas[lemma].append([classes[ci], lemma, scores[ci], syn.name(), syn.pos()] + scores)

    if sel_fn:
        with open(sel_fn) as file:
            csvreader = csv.reader(file, delimiter=',', quotechar='"')
            for row in csvreader:
                lemma = clean_spaces(row[1])
                if lemma not in lemmas:
                    sentiment = row[7]
                    score = float(row[6])
                    if score < threshold:
                        sentiment = 'neutral'
                        score = 1.0 - score
                        scores = [(1.0 - score) / 2, score, (1.0 - score) / 2]
                    else:
                        if clean_spaces(row[7]) in {'Alegría', 'Sorpresa'}:
                            sentiment = 'positive'
                            scores = [score, (1.0 - score) / 2, (1.0 - score) / 2]
                        elif clean_spaces(row[7]) in {'Enojo', 'Miedo', 'Repulsión', 'Tristeza'}:
                            sentiment = 'negative'
                            scores = [(1.0 - score) / 2, (1.0 - score) / 2, score]
                        else:
                            print(f"WARNING: unknown sentiment '{sentiment}'")
                            continue

                    lemmas[lemma].append([sentiment, lemma, score, '<unk>', '<unk>'] + scores)

    if fix_fn:
        with open(fix_fn) as file:
            csvreader = csv.reader(file, delimiter=',', quotechar='"')
            for row in csvreader:
                lemma = clean_spaces(row[1])
                sentiment = row[0]
                score = float(row[3])
                if sentiment == 'neutral':
                    scores = [(1.0 - score) / 2, score, (1.0 - score) / 2]
                elif sentiment == 'positive':
                    scores = [score, (1.0 - score) / 2, (1.0 - score) / 2]
                elif sentiment == 'negative':
                    scores = [(1.0 - score) / 2, (1.0 - score) / 2, score]
                else:
                    print(f"WARNING: unknown sentiment '{sentiment}'")
                    continue

                lemmas[lemma] = [[sentiment, lemma, score, '<unk>', '<unk>'] + scores]

    if save_fn:
        with open(save_fn, 'w') as file:
            csvwriter = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for lemma, options in sorted(lemmas.items()):
                for option in sorted(options):
                    csvwriter.writerow(option)
    else:
        print(json.dumps(lemmas, indent=2, ensure_ascii=False))


@app.command()
def pre_classify(sent_fn: Path, text_fns: List[Path], save_fn: Path = typer.Option(None), min_count: int = 0):
    nlp = spacy.load('es_core_news_lg')

    positive = {}
    negative = {}
    vocab = defaultdict(Counter)
    with open(sent_fn) as file:
        csvreader = csv.reader(file, delimiter=',', quotechar='"')
        for row in tqdm(csvreader):
            if row[0] == 'positive':
                positive[row[1]] = float(row[2])
            elif row[0] == 'negative':
                negative[row[1]] = float(row[2])

    for text_fn in tqdm(text_fns):
        with open(text_fn) as file:
            lines = file.read().split('\n')

        for line in tqdm(lines, leave=False):
            doc = nlp(line)
            toks = [t for t in doc if not t.is_punct and not t.is_stop and not t.is_space]
            poses = {t.pos_ for t in toks}
            for pos in poses:
                vocab[pos].update([t.lemma_ for t in toks if t.pos_ == pos])
                print(f"POS {pos}: { {t.lemma_ for t in toks if t.pos_ == pos} }")
            pos = sum([positive.get(t.lemma_, 0) for t in doc])
            neg = sum([negative.get(t.lemma_, 0) for t in doc])
            is_first_person = any(['1' in t.morph.get("Person") and 'Sing' in t.morph.get("Number") for t in doc])
            print(f"{is_first_person} # {pos - neg}: {doc} -> +{pos}: {[t.lemma_ for t in doc if t.lemma_ in positive]} | -{neg}: {[t.lemma_ for t in doc if t.lemma_ in negative]}")

    if save_fn:
        with open(save_fn, 'w') as file:
            csvwriter = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for pos, counts in vocab.items():
                for count, word in sorted([(c, w) for w, c in counts.items() if not min_count or c >= min_count], reverse=True):
                    csvwriter.writerow([pos, word, count, positive.get(word, None), negative.get(word, None)])
    else:
        print(json.dumps(vocab, indent=2, ensure_ascii=False))


@app.command()
def pre_classify_vocab(sent_fn: Path, text_fns: List[Path], save_fn: Path = typer.Option(None), min_count: int = 0):
    nlp = spacy.load('es_core_news_lg')

    positive = {}
    negative = {}
    vocab = defaultdict(Counter)
    with open(sent_fn) as file:
        csvreader = csv.reader(file, delimiter=',', quotechar='"')
        for row in tqdm(csvreader):
            if row[0] == 'positive':
                positive[row[1]] = float(row[2])
            elif row[0] == 'negative':
                negative[row[1]] = float(row[2])

    for text_fn in tqdm(text_fns):
        with open(text_fn) as file:
            lines = file.read().split('\n')

        for line in tqdm(lines, leave=False):
            line = clean_spaces(line).split(' ')
            if not line:
                continue

            print(f"LINE: {line}")
            try:
                count = int(line[0])
            except:
                continue

            doc = nlp(line[1])
            toks = [t for t in doc if not t.is_punct and not t.is_stop and not t.is_space and not t.is_oov]
            poses = {t.pos_ for t in toks}
            for pos in poses:
                vocab[pos].update({t.lemma_: count for t in toks if t.pos_ == pos})
                print(f"POS {pos}: { {t.lemma_ for t in toks if t.pos_ == pos} }")
            pos = sum([positive.get(t.lemma_, 0) for t in doc])
            neg = sum([negative.get(t.lemma_, 0) for t in doc])
            is_first_person = any(['1' in t.morph.get("Person") and 'Sing' in t.morph.get("Number") for t in doc])
            print(f"{is_first_person} # {pos - neg}: {doc} -> +{pos}: {[t.lemma_ for t in doc if t.lemma_ in positive]} | -{neg}: {[t.lemma_ for t in doc if t.lemma_ in negative]}")

    if save_fn:
        with open(save_fn, 'w') as file:
            csvwriter = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for pos, counts in vocab.items():
                for count, word in sorted([(c, w) for w, c in counts.items() if not min_count or c >= min_count], reverse=True):
                    csvwriter.writerow([pos, word, count, positive.get(word, None), negative.get(word, None)])
    else:
        print(json.dumps(vocab, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    app()
