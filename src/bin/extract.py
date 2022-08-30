#! /usr/bin/env python

import re
import json
from pathlib import Path
from typing import List

import typer
from bs4 import BeautifulSoup
import spacy

from medp.utils import clean_spaces
from medp.html import CleanHtml
from medp.html import D as Document

app = typer.Typer()


@app.command()
def extract_title(html_fn: Path):
    with open(html_fn) as file:
        doc = Document(file.read())
    print(doc.title())


@app.command()
def extract_summary(html_fn: Path, clean: bool = True, summarize: bool = True):
    bs = get_summary(html_fn, clean=clean, summarize=summarize)
    print(bs)


def remove(soup, tags, ids, classes):
    for tag in tags:
        elements = soup.find_all(tag)
        for element in elements:
            element.decompose()

    for _id in ids:
        elements = soup.find_all(id=_id)
        for element in elements:
            element.decompose()

    for _class in classes:
        elements = soup.find_all(class_=_class)
        for element in elements:
            element.decompose()


def get_clean_soup(soup):
    remove(soup, [
        'header', 'footer', 'aside', 'menu', 'nav',
    ], [
        'threads', 'replies', 'header',
        'af-seealso-video-container', 'forum-threads-help',
    ], [
        'fa-clock', 'hide_mobile',
        'mycode_u', 'mycode_i', 'mycode_quote', 'mycode_hr',
        'post_author', 'post_date',
        'answer-to', 'af-post-actions',
        'for_buttons_bar', 'ccm_header',
        'for_answer__navigation', 'app_layout_right',
        'app_layout_top', 'topic_nav',
        'for_topic__question__more',
        'af-see-last-thread-container',
        'af-cta-cant-find-answer',
        'ad-container',
        'af-forum-seo-links',
        'af-bread-crumb', 'unfyas', 'af-last-response-header',
        'af-post-title', 'af-toggle-text', 'af-post-header',
        'af-btns-wrapper', 'forum-expert-club-products',
    ])
    return soup


def get_clean_text(soup):
    soup = get_clean_soup(soup)
    text = soup.get_text()
    text = text.replace('<![CDATA[', '').replace(']]>', '').replace('_', ' ')
    return text


def get_summary(html_fn: Path, clean: bool = True, summarize: bool = True):
    cleaner = CleanHtml()
    try:
        with open(html_fn) as file:
            html = file.read()
    except UnicodeDecodeError:
        print(f"WARNING: {html_fn} could not be processed")
        return None
    except IsADirectoryError:
        return None

    html = cleaner.clean(html)

    if summarize:
        html = Document(html).summary()

    bs = BeautifulSoup(html, "lxml")

    if clean:
        bs = get_clean_soup(bs)
        cleaner.remove_inline_tags(bs)
        bs = cleaner.regenerate_tree(bs)
    return bs


@app.command()
def extract_text(html_fns: List[Path], clean: bool = True, summarize: bool = True):
    cleaner = CleanHtml()
    lists = [fn for fn in html_fns if fn.suffix == '.list']
    html_fns = [fn for fn in html_fns if not fn.suffix == '.list']

    for fn in lists:
        with open(fn) as file:
            html_fns.extend([Path(line) for line in file.read().split('\n') if clean_spaces(line)])

    print(f"Processing {len(html_fns)} urls ...")
    for html_fn in html_fns:
        bs = get_summary(html_fn, clean=clean, summarize=summarize)
        if not bs:
            continue

        text_lines = cleaner.extract_lines(bs, do_sentence_split=True)

        print('\n'.join(text_lines))


@app.command()
def extract_recetasderechupete(html_fns: List[Path], clean: bool = True, summarize: bool = True):
    lists = [fn for fn in html_fns if fn.suffix == '.list']
    html_fns = [fn for fn in html_fns if not fn.suffix == '.list']

    for fn in lists:
        with open(fn) as file:
            html_fns.extend([Path(line) for line in file.read().split('\n') if clean_spaces(line)])

    print(f"Processing {len(html_fns)} urls ...")
    for html_fn in html_fns:
        try:
            with open(html_fn) as file:
                html = file.read()
        except UnicodeDecodeError:
            print(f"WARNING: {html_fn} could not be processed")
            return None
        except IsADirectoryError:
            return None

        bs = BeautifulSoup(html, "lxml")

        script = bs.find('script', {"id": 'recipejson'})
        if script:
            text = script.get_text().replace('\t', '').replace('\n', '')
            if text:
                try:
                    data = json.loads(text)
                    string = json.dumps(data, indent=None, ensure_ascii=False)
                    print(string)
                except:
                    print(f"KK: {html_fn} {text}")
            else:
                print(f"EMPTY: {html_fn}")
        else:
            print(f"UNK: {html_fn}")


if __name__ == "__main__":
    app()
