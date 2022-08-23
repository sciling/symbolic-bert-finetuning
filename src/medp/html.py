import re

from collections import Counter
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional
from xml.etree.ElementTree import ParseError  # nosec
import spacy

from bs4 import BeautifulSoup
from bs4 import Comment
from bs4 import Tag
from readability.readability import Document
from readability.readability import Unparseable


class safedoc:
    def __init__(self, doc) -> None:
        self.D = doc

    def __enter__(self):
        # logger.debug('Processing with safedoc...')
        return self.D.doc

    def __exit__(self, exc_type, exc_val, traceback):
        # logger.debug(f"Finish processing with safedoc... {exc_type} {exc_val} {exc_type == ParserError and str(exc_val) == 'Document is empty'}")

        # There was no exception.
        if exc_type is None:
            return True

        # Ignore 'Document is empty' errors.
        if exc_type in (ParseError, Unparseable) and str(exc_val) == "Document is empty":
            return True

        # Ignore other errors, but log them
        html = self.D.html
        if len(html) > 100:
            html = html[:100] + "..."

        print(f"ERROR: Readability failed to provide results for: '{html}'")
        return True


# https://stackoverflow.com/q/35446015
class OrderedCounter(Counter, OrderedDict):
    "Counter that remembers the order elements are first seen"

    @classmethod
    def fromkeys(cls, iterable, v=None):
        return OrderedDict.fromkeys(iterable, v)

    def __repr__(self):
        return f"{self.__class__.__name__!s}({OrderedDict(self)!r})"

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)


class D:
    def __init__(self, html: str) -> None:
        self.doc = None
        self.html = html
        try:
            self.doc = Document(html)
        except Exception:  # pylint: disable=broad-except
            html = self.html
            if len(html) > 100:
                html = html[:100] + "..."

            print(f"Unparsable '{html}', title and summary will be empty")

    def content(self):
        with safedoc(self) as doc:
            return doc.content()
        return ""

    def summary(self):
        with safedoc(self) as doc:
            return doc.summary()
        return ""

    def title(self):
        with safedoc(self) as doc:
            return doc.title()
        return ""


class CleanHtml():
    _re_is_space = re.compile(r"^\s*$", re.MULTILINE)
    _re_spaces = re.compile(r"\s+", re.MULTILINE)
    _re_ignore_section = re.compile(r"(cookie)", re.IGNORECASE)

    _extra_tags = {
        "main",
        "p",
    }

    _collapsable_tags = {
        "div",
        "blockquote",
    }

    _layout_tags = {
        "aside",
        "figure",
        "footer",
        "header",
        "nav",
        "time",
        "img",
        # These are perhaps not layout tags but should also be removed.
        "script",
        "style",
        "button",
        "noscript",
    }

    _inline_tags = {
        "a",
        "span",
        "em",
        "strong",
        "u",
        "i",
        "font",
        "mark",
        "label",
        "s",
        "sub",
        "sup",
        "tt",
        "bdo",
        "cite",
        "del",
        "b",
        "font",
    }

    _nlp = spacy.load('es_core_news_lg')

    def __init__(
        self,
        do_extract=True,
        do_sentence_split=False,
        min_clean_length=10,
    ):
        self.do_extract = do_extract
        self.do_sentence_split = do_sentence_split
        self.min_clean_length = min_clean_length

    def clean(self, text):
        return self._re_spaces.sub(" ", text).strip()

    @staticmethod
    def split_sent(strings, do_sentence_split=False):
        if do_sentence_split:
            return [str(s) for string in strings for s in CleanHtml._nlp(string.strip()).sents if len(s) > 0]
        return strings

    @staticmethod
    def regenerate_tree(bs):
        return BeautifulSoup(str(bs), "lxml")

    def remove_inline_tags(self, bs):
        for tag in self._inline_tags:
            for el in bs.findAll(tag):
                el.replaceWithChildren()

    def clean_tags(self, bs):
        body = bs.find("body")
        if not body:
            html = bs.find("html")
            body = bs if not html else bs.new_tag("body")
            if html:
                html.append(body)

        for tag in bs.find_all("meta"):
            name = tag.attrs.get("name", None)
            if name in ("keywords", "description"):
                ntag = bs.new_tag(name)
                if name == "keywords":
                    for keyword in tag.attrs.get("content", "").split(","):
                        litag = bs.new_tag("li")
                        litag.string = keyword.strip()
                        ntag.append(litag)
                else:
                    ntag.string = tag.attrs.get("content", "")

                body.append(ntag)

        # Remove all attributes.
        for tag in bs.find_all(lambda tag: len(tag.attrs) > 0):
            tag.attrs.clear()

        # Remove all comments and empty text nodes.
        for tag in bs.find_all(text=lambda text: isinstance(text, Comment) or self._re_is_space.match(text)):
            tag.extract()

        # Remove all tags with no text.
        for tag in bs.find_all():
            if len(tag.get_text(strip=True)) == 0:
                tag.extract()

        # Collapse all nested divs.
        for tag in bs.find_all(lambda tag: tag.name in self._collapsable_tags and len(list(tag.children)) == 1 and isinstance(next(tag.children), Tag)):
            tag.replaceWithChildren()

    def delete_layout_tags(self, bs):
        for tag in self._layout_tags:
            for el in bs.findAll(tag):
                el.extract()

    def clean_html(self, html):
        bs = BeautifulSoup(self.clean(html), "lxml")

        # Remove inline tags so only block tags remain.
        self.remove_inline_tags(bs)

        # Delete layout nodes so we don't get menus, etc.
        self.delete_layout_tags(bs)

        # Remove tags that contain no text.
        self.clean_tags(bs)

        # Delete cookie section
        self.delete_cookie_section(bs)

        return str(bs)

    def delete_cookie_section(self, bs, parent_cookie_ratio=0.75):
        cookies = bs.findAll(text=self._re_ignore_section)
        # Wrap cookies in an element that we can identify.
        cookies = [c.wrap(bs.new_tag("cookies")) for c in cookies]

        # Look for cookie tag ancestors for which cookie uses more
        # than parent_cookie_ratio
        if len(cookies) > 0:
            parents_count = {}
            parent_xpaths = {}
            for e in cookies:
                size = len(e.get_text("\n", strip=True))
                px = xpath_soup(e)
                parent_xpaths[px] = e
                parents_count[px] = 1

                for p in list(e.parents)[:-3]:
                    px = xpath_soup(p)
                    psize = len(p.get_text("\n", strip=True))
                    if size > psize * parent_cookie_ratio:
                        parent_xpaths[px] = p
                        if px in parents_count:
                            parents_count[px] += 1
                        else:
                            parents_count[px] = 1
                    else:
                        break

            for px in parents_count:
                parent_xpaths[px].extract()

    def extract_lines(self, bs, do_sentence_split=False):
        # get_text with a \n which will only separate block tags now
        # and then split by \n to separate by blocks,
        texts = bs.get_text("\n", strip=True).split("\n")

        # Separate blocks into sentences.
        summary = self.split_sent(texts, do_sentence_split)

        # logger.info("SUMMARY: %s", summary)

        return summary

    def extract_summary(self, html, do_sentence_split=False):
        doc = D(html)

        bs = BeautifulSoup(doc.summary(), "lxml")

        # Remove inline tags so only block tags remain.
        self.remove_inline_tags(bs)

        # Make sure that contiguous text nodes have been merged into a single text node.
        bs = self.regenerate_tree(bs)

        # Add title if it exists.
        title = self.clean(doc.title())
        if title == "[no-title]":
            title = None

        return title, self.extract_lines(bs)

    def extract_heuristic(self, html, do_sentence_split=False):
        html = self.clean_html(html)
        bs = BeautifulSoup(html, "lxml")

        # get_text with a \n which will only separate block tags now
        # and then split by \n to separate by blocks,
        texts = bs.get_text("\n", strip=True).split("\n")

        # Add title if it exists.
        title = bs.find("title")
        if title:
            title = self.clean(title.get_text())
        if not title:
            title = None

        # Separate blocks into sentences.
        summary = OrderedCounter(self.split_sent(texts, do_sentence_split))
        # Assume that texts that appear more than once and are exactly the same
        # come from a template and do not provide meaning to the page.
        summary = [t for t, c in summary.items() if c == 1]

        # logger.info("SUMMARY: %s", summary)

        return title, summary

    # def transform(self, single_item: RawHtmlPage, config=None):
    #     bstext = ""
    #     title = ""
    #     if single_item.html and str(single_item.status_code) == "200":
    #         sentry_sdk.set_context("rawHtmlPage", {"url": single_item.html})

    #         # Normalize spaces since they are not semantically relevant in
    #         # a HTML document.
    #         html = self.clean(single_item.html)

    #         do_extract = self.do_extract
    #         do_sentence_split = self.do_sentence_split
    #         logger.debug(f"Transforming '{single_item.url}' EXTRACT_HEURISTIC={do_extract} SENTENCE_SPLIT={do_sentence_split}...")

    #         if do_extract:
    #             # Extract the summary of the document.
    #             title, text_lines = self.extract_heuristic(html, do_sentence_split=do_sentence_split)
    #         else:
    #             # Extract the summary of the document.
    #             title, text_lines = self.extract_summary(html, do_sentence_split=do_sentence_split)

    #         if title and title not in text_lines:
    #             text_lines.append(title)

    #         bstext = "\n".join((line for line in text_lines if line)).strip()

    #         scraped = ScrapedHtmlPage(url=str(single_item.url), title=title, clean=bstext, lemma="")
    #         if len(scraped.clean) > self.min_clean_length:
    #             return scraped
    #     return None


# https://gist.github.com/ergoithz/6cf043e3fdedd1b94fcf
def xpath_soup(element):
    """
    Generate xpath from BeautifulSoup4 element.
    :param element: BeautifulSoup4 element.
    :type element: bs4.element.Tag or bs4.element.NavigableString
    :return: xpath as string
    :rtype: str
    Usage
    -----
    >>> import bs4
    >>> html = (
    ...     '<html><head><title>title</title></head>'
    ...     '<body><p>p <i>1</i></p><p>p <i>2</i></p></body></html>'
    ...     )
    >>> soup = bs4.BeautifulSoup(html, 'html.parser')
    >>> xpath_soup(soup.html.body.p.i)
    '/html/body/p[1]/i'
    >>> import bs4
    >>> xml = '<doc><elm/><elm/></doc>'
    >>> soup = bs4.BeautifulSoup(xml, 'lxml-xml')
    >>> xpath_soup(soup.doc.elm.next_sibling)
    '/doc/elm[2]'
    """
    components = []
    child = element if element.name else element.parent
    for parent in child.parents:
        siblings = parent.find_all(child.name, recursive=False)
        components.append(child.name if len(siblings) == 1 else f"{child.name!s}[{next(i for i, s in enumerate(siblings, 1) if s is child)}]")
        child = parent
    components.reverse()
    return f"/{'/'.join(components)!s}"
