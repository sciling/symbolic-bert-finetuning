#! /usr/bin/env python

import re
import sys
import csv
import json
from queue import Queue
from pathlib import Path
from typing import List
from urllib.parse import unquote

import typer
from tqdm import tqdm
import wikipedia
import wikipediaapi
from wikipedia.exceptions import DisambiguationError, PageError
import requests


app = typer.Typer()


wd_id_re = re.compile(r"^[PQL]\d+$")
wp_pars_re = re.compile(r"\s*\(.*\)$")


wd_aliases = {
    'dish': 'Q746549',
    'food': 'Q2095',
    'sport': 'Q349',
    'feeling': 'Q9415',
}

query_template = """
SELECT ?item ?itemLabel ?itemDescription ?lemma ?sitelink
WHERE {
  ?item wdt:P31 wd:%s.
  ?sitelink schema:about ?item;
    schema:isPartOf <https://%s.wikipedia.org/>;
    schema:name ?lemma.

  SERVICE wikibase:label {
        bd:serviceParam wikibase:language "%s".
   }
}
"""


def wd_search(term: str, langs: str, wiki_lang: str = 'en'):
    term = wd_aliases.get(term, term)
    res = requests.get('https://query.wikidata.org/sparql', params={'format': 'json', 'query': query_template % (term, wiki_lang, langs)})
    return res.json().get('results', {}).get('bindings', [])


@app.command()
def search_wikidata(terms: List[str], export_csv_fn: Path = typer.Option(None)):
    typer.echo(f"Processing terms {terms} in to '{export_csv_fn}' ...")

    docs = {}
    for term in terms:
        data = wd_search(term, 'es,en')

        for item in data:
            docs[item.get('item', {}).get('value', None)] = {
                'label': item.get('itemLabel', {}).get('value', None),
                'description': item.get('itemDescription', {}).get('value', None),
            }

        data = wd_search(term, 'en')

        for item in data:
            item_id = item.get('item', {}).get('value', None)
            wiki_url = item.get('sitelink', {}).get('value', '')
            wiki_id = unquote(re.sub(r'^.*/wiki/', '', wiki_url.replace('_', ' ')).lower())

            label = item.get('itemLabel', {}).get('value', None)
            try:
                print(f"WIKISEARCH: {label}: {wiki_url}: {wiki_id}")
                description = wikipedia.summary(wiki_id)
                print(f"WIKI: {wiki_id}: {description}")
            except (PageError, DisambiguationError) as err:
                print(f"ERROR: {err}")
                description = item.get('itemDescription', {}).get('value', None)

            if item_id not in docs:
                docs[item_id] = {
                    'label': label,
                    'description': description,
                }
            else:
                if not docs[item_id]['label']:
                    docs[item_id]['label'] = item.get('itemLabel', {}).get('value', None)

                if description:
                    docs[item_id]['description'] = description

    if export_csv_fn:
        with open(export_csv_fn, "w") as file:
            csvwriter = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for item_id, doc in docs.items():
                if wd_id_re.match(doc['label']):
                    continue
                row = [doc['label'], doc['description']]
                csvwriter.writerow(row)
    else:
        json.dump(docs, sys.stdout, indent=2, ensure_ascii=False)


wp_aliases = {
    'food': ['Categoría:Alimentos_por_tipo'],
    'sport': ['Categoría:Deportes'],
    'feeling': ['Categoría:Emociones'],
}


def find_all_wp(terms: List[str], do_subcats: bool = False):
    wiki = wikipediaapi.Wikipedia('es')

    docs = {}
    queued = set()
    queue = Queue()

    for term in terms:
        aliases = wp_aliases.get(term, [term])
        for alias in aliases:
            if alias not in queued:
                queue.put(alias)
                queued.add(alias)

    with tqdm(total=queue.qsize()) as pbar:
        while not queue.empty():  # and pbar.n < 10:
            pbar.update()
            term = queue.get()
            if isinstance(term, str):
                print(f"SEARCHING: {term}")
                page = wiki.page(term)
            else:
                print(f"PROCESSING: {term.title}")
                page = term
                term = term.title

            if page.exists():
                if 'en' in page.langlinks:
                    description = page.langlinks['en'].summary
                else:
                    description = ''

                if description:
                    docs[term] = {
                        'label': wp_pars_re.sub('', term),
                        'description': description,
                    }

                try:
                    for member in page.categorymembers.values():
                        if member not in queued and (do_subcats or not member.title.startswith('Categoría:')):
                            queue.put(member)
                            queued.add(member)
                    pbar.total = queue.qsize()
                    pbar.refresh()
                except KeyError:
                    pass
            else:
                print(f"ERROR: page '{term}' does not exist")

    return docs


@app.command()
def search(terms: List[str], do_subcats: bool = False, export_csv_fn: Path = typer.Option(None)):
    typer.echo(f"Processing terms {terms} in to '{export_csv_fn}' ...")
    docs = find_all_wp(terms, do_subcats=do_subcats)

    if export_csv_fn:
        with open(export_csv_fn, "w") as file:
            csvwriter = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for doc in docs.values():
                if wd_id_re.match(doc['label']):
                    continue
                row = [doc['label'], doc['description']]
                csvwriter.writerow(row)
    else:
        json.dump(docs, sys.stdout, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    app()
