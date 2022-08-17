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

import re
import csv
import json
import random
from collections import Counter
from collections import OrderedDict
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import List

from tqdm import tqdm
import nltk
import ruamel.yaml
import typer
import pandas as pd

from unidecode import unidecode
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Synset  # pylint: disable=no-name-in-module
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from ruamel.yaml import YAML
from ruamel.yaml.representer import RoundTripRepresenter
from medp.utils import SearchEngine, NLP, EmbeddingsProcessor, DescriptionType, get_syns, expand_template, clean_spaces


warnings.filterwarnings("ignore")

app = typer.Typer()


# https://stackoverflow.com/a/53875283
class MyRepresenter(RoundTripRepresenter):
    pass


ruamel.yaml.add_representer(OrderedDict, MyRepresenter.represent_dict, representer=MyRepresenter)


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
def describe(sentence: str, db_fn: Path = typer.Option(None)):
    with open(db_fn) as file:
        vocab = json.load(file)

    print(NLP.describe(sentence, vocab))


@app.command()
def summarize(sentence: str, db_fn: Path = typer.Option(None)):
    with open(db_fn) as file:
        vocab = json.load(file)

    print(NLP.summarize(sentence, vocab))


@app.command()
def create_db(
    vocab_fns: List[Path], entities_fn: Path = typer.Option(None),
    save_fn: Path = typer.Option(None),
    redefinitions_fn: Path = typer.Option(None),
    override_definitions: bool = True
):
    vocab = {}
    redefinitions = {}

    if redefinitions_fn:
        with open(redefinitions_fn) as file:
            csvreader = csv.reader(file, delimiter=',', quotechar='"')
            for row in csvreader:
                if len(row) >= 2:
                    redefinitions[row[0]] = row[1]

    for vocab_fn in vocab_fns:
        NLP.update_vocab(vocab, vocab_fn, redefinitions=redefinitions, override_definitions=override_definitions)

    if entities_fn:
        NLP.update_vocab(vocab, entities_fn, is_entity=True, redefinitions=redefinitions, override_definitions=override_definitions)

    for data in tqdm(vocab.values(), desc="Regenerating descriptions..."):
        if not data.get('description', None):
            data['description'] = NLP.describe(data['label'], vocab)
            data['embedding'] = EmbeddingsProcessor.pages_to_embeddings([data['description']])[0].tolist()

    with open(save_fn, 'w') as file:
        json.dump(vocab, file, indent=2, ensure_ascii=False)


@app.command()
def summarize_db(db_fn: Path, save_fn: Path = typer.Option(None), description_type: DescriptionType = DescriptionType.DEFAULT, reuse_descriptions: bool = False):
    with open(db_fn) as file:
        db = json.load(file)

    ignore_token = {'conservar'}
    for token in ignore_token:
        del db[token]

    new_db = {}
    for text, data in tqdm(db.items()):
        new_db[text] = NLP.summarizedb_entry(data, db, description_type, reuse_descriptions)

    with open(save_fn, 'w') as file:
        json.dump(new_db, file, indent=2, ensure_ascii=False)


@app.command()
def search(text: str, db_fn: Path = typer.Option(None), ignore_fn: Path = typer.Option(None), nbest: int = 4, summarized: bool = False, multinomial: bool = False, description_type: DescriptionType = DescriptionType.DEFAULT, reuse_description: bool = True, fuzzy=True):
    searcher = SearchEngine(db_fn, ignore_fn)
    res = searcher.search(text, nbest, summarized=summarized, multinomial=multinomial, description_type=description_type, reuse_description=reuse_description, fuzzy=fuzzy)

    print(json.dumps(res, indent=2, ensure_ascii=False))


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
                row = [name, str(desc).replace(',', '').replace('\n', ' '), name.lower()]
                csvwriter.writerow(row)
    else:
        print(all_docs)


@app.command()
def fix_entities(entities_fn: Path, save_fn: Path):
    with open(entities_fn) as fileread, open(save_fn, "w") as filewrite:
        csvreader = csv.reader(fileread, delimiter=',', quotechar='"')
        csvwriter = csv.writer(filewrite, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in csvreader:
            csvwriter.writerow([row[0].strip(), row[1].strip()] + list(sorted(set([r.strip().lower() for r in row[2:] if not re.match(r"^\s*$", r)]))))


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
def expand_entities(entities_fn: Path, templates_fn: List[Path] = typer.Argument(None), save_fn: Path = typer.Option(None), depth: int = 1, threshold: float = None):
    with open(entities_fn) as file:
        csvreader = csv.reader(file, delimiter=',', quotechar='"')
        entity_rows = [row[:] for row in csvreader]

    seen = {}
    entities = defaultdict(set)

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

    for template_fn in tqdm(templates_fn):
        print(f"PROCESSING: {template_fn}")
        templates = []
        yaml = YAML(typ="safe")  # default, if not specfied, is 'rt' (round-trip)
        with open(template_fn) as file:
            data = yaml.load(file)
            variables = data.get('variables', {})
            for ent, values in data.get('entities', {}).items():
                for val in values:
                    entity = f"{ent}:{val['value']}"
                    print(f"VAL: {val}: {variables}")
                    temps = [temp.format(**variables) for temp in val.get('templates', [])]
                    templates.append((entity, temps))

        for entity, elems in tqdm(templates, leave=False):
            vocab = set()
            for template in tqdm(elems, leave=False):
                for sentence in tqdm(expand_template(template), leave=False):
                    print(f"TEMPLATE: {template}: {sentence}")
                    analysis = NLP.tag(sentence)
                    words = {sentence}

                    if len(analysis) == 1:
                        words |= NLP.convert_form_recursive(sentence, depth=depth, threshold=threshold)

                    for word in list(words):
                        subanalysis = NLP.tag(word)
                        print(f"SUB: {word}: {subanalysis}")
                        for number in ('sg', ):  # 'pl'):
                            tokens = [NLP.get_variations(token, number, pos) for pos, token in enumerate(subanalysis)]
                            alts = generate_alternatives([t for t in tokens])
                            # print(f"ALTS[{number}] = {alts}")
                            words |= alts

                    for syn in list(words):
                        if seen.get(syn, entity) != entity:
                            print(f"WARNING: '{syn}' in '{entity}' from template '{template}' is already a synonym of '{seen[syn]}'")
                            words.remove(syn)
                        else:
                            seen[syn] = entity

                    vocab |= words

            print(f"ENTITY: {entity} = {vocab}")
            entities[entity] |= vocab

    if save_fn:
        with open(save_fn, 'w') as file:
            csvwriter = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for entity, synonyms in entities.items():
                csvwriter.writerow(entity.split(':') + sorted(synonyms))
    else:
        print(json.dumps(entities, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    app()
