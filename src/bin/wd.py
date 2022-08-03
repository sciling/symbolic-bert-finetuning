#! /usr/bin/env python

import re
import sys
import csv
import json
from pathlib import Path
from typing import List
from urllib.parse import unquote

import typer
import wikipedia
from wikipedia.exceptions import DisambiguationError, PageError
import requests


app = typer.Typer()


wd_id_re = re.compile(r"^[PQL]\d+$")


aliases = {
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
    term = aliases.get(term, term)
    res = requests.get('https://query.wikidata.org/sparql', params={'format': 'json', 'query': query_template % (term, wiki_lang, langs)})
    return res.json().get('results', {}).get('bindings', [])


@app.command()
def search(terms: List[str], export_csv_fn: Path = typer.Option(None)):
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


@app.command()
def get_closest(csv_fn: Path, db_fn: Path):
    pass


if __name__ == "__main__":
    app()
