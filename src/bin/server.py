#! /usr/bin/env python

import os
import json
from typing import Optional

from fastapi import Depends, FastAPI, Request, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel


from medp.utils import SearchEngine, DescriptionType, clean_spaces
from bin.classifier import Classifier

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="html")

security = HTTPBasic()


def check_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = credentials.username == os.getenv("SERVER_USER")
    correct_password = credentials.password == os.getenv("SERVER_PASSWORD")

    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect user or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


CACHED = {
    'alimento_tipo': None,
}

for entity in CACHED:
    CACHED[entity] = SearchEngine(f"db/{entity}.json", vocab_fn='vocab.json', ignore_fn='ignore.json')


@app.get("/search/{entity}")
def search(entity: str, text: str, nbest: int = 4, summarized: bool = True, multinomial: bool = True, description_type: DescriptionType = DescriptionType.LONG, reuse_description: bool = False, fuzzy: bool = True, max_ngram: int = 5, username: str = Depends(check_credentials)):
    if entity in CACHED:
        searcher = CACHED[entity]
    else:
        searcher = SearchEngine(f"db/{entity}.json", vocab_fn='vocab.json', ignore_fn='ignore.json')
        CACHED[entity] = searcher

    res = searcher.search(text, nbest, summarized=summarized, multinomial=multinomial, description_type=description_type, reuse_description=reuse_description, fuzzy=fuzzy, max_ngram=max_ngram)

    print(f"RES: {res}")

    return res


@app.get("/sentiment", response_class=HTMLResponse)
async def sentiment_html(request: Request):
    return templates.TemplateResponse("sentiment.html", {"request": request})


@app.get("/mood", response_class=HTMLResponse)
async def sentiment_html(request: Request):
    return templates.TemplateResponse("mood.html", {"request": request})


class Sentence(BaseModel):
    text: str
    label: Optional[str]


models = {
    'sentiment': {
        'model': None,
        'model_name': 'train.dir',
        'database_fn': 'sentiment-database.json',
    },
    'mood': {
        'model': None,
        'model_name': 'train2.dir',
        'database_fn': 'mood-database.json',
    },
}


def load_model(model_name: str):
    data = models.get(model_name, None)
    if not data:
        return None

    if data.get('model', None) is None:
        data['model'] = Classifier(**data)

    return models[model_name].get('model', None)


def classify(model_name: str, sentence: Sentence):
    sentence.text = clean_spaces(sentence.text)
    classifier = load_model(model_name)

    result = [(c, s) for s, c in classifier.classify(sentence.text)]

    return {
        "text": sentence.text,
        "result": result,
    }


def fix(model_name: str, sentence: Sentence):
    sentence.text = clean_spaces(sentence.text)
    classifier = load_model(model_name)
    classifier.fix(sentence.text, sentence.label)

    return {
        "text": sentence.text,
        "label": sentence.label,
    }


@app.post("/sentiment", response_class=JSONResponse)
async def classify_sentiment(sentence: Sentence):
    return classify('sentiment', sentence)


@app.post("/sentiment-fix", response_class=JSONResponse)
async def fix_sentiment(sentence: Sentence):
    return fix('sentiment', sentence)


@app.post("/mood", response_class=JSONResponse)
async def classify_mood(sentence: Sentence):
    return classify('mood', sentence)


@app.post("/mood-fix", response_class=JSONResponse)
async def fix_mood(sentence: Sentence):
    return fix('mood', sentence)
