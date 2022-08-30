#! /usr/bin/env python

import os
import asyncio
from typing import Optional
from collections import defaultdict

from fastapi import Depends, FastAPI, Request, HTTPException, status
from fastapi.encoders import jsonable_encoder
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel  # pylint: disable=no-name-in-module


from medp.utils import SearchEngine, DescriptionType, clean_spaces, Database
from bin.classifier import Classifier, Tagger

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

CACHED['food'] = Tagger('./train-ma.dir')


@app.get("/search/{entity}")
def search(
    entity: str, text: str, nbest: int = 4, summarized: bool = True,
    multinomial: bool = True, description_type: DescriptionType = DescriptionType.LONG,
    reuse_description: bool = False, fuzzy: bool = True, max_ngram: int = 5,
    username: str = Depends(check_credentials)
):
    if entity in CACHED:
        searcher = CACHED[entity]
    else:
        searcher = SearchEngine(f"db/{entity}.json", vocab_fn='vocab.json', ignore_fn='ignore.json')
        CACHED[entity] = searcher

    res = searcher.search(
        text, nbest, summarized=summarized, multinomial=multinomial,
        description_type=description_type, reuse_description=reuse_description,
        fuzzy=fuzzy, max_ngram=max_ngram
    )

    SEARCH_DB[text] = {
        'text': text,
        'result': res,
    }

    print(f"RES: {res}")

    return res


class FoodFix(BaseModel):
    food: str
    entity: str
    sign: str


@app.post("/food-fix", response_class=JSONResponse)
async def food_fix(food_fix: FoodFix):
    key = f"{food_fix.food}~{food_fix.entity}"
    FOOD_FIX_DB[key] = jsonable_encoder(food_fix)
    print(f"FIX: {key} = {FOOD_FIX_DB[key]}")
    return {'message': 'OK'}


@app.get("/parse-food")
async def parse_food(
    text: str, nbest: int = 4, summarized: bool = True,
    multinomial: bool = True, description_type: DescriptionType = DescriptionType.LONG,
    reuse_description: bool = False, fuzzy: bool = True, max_ngram: int = 5,
    username: str = Depends(check_credentials)
):
    tagger = CACHED['food']
    searcher = CACHED['alimento_tipo']

    res = tagger.tag(text)

    for food in res.get('foods', []):
        food.search = searcher.search(
            food.food, nbest, summarized=summarized, multinomial=multinomial,
            description_type=description_type, reuse_description=reuse_description,
            fuzzy=fuzzy, max_ngram=max_ngram
        )
        for sr in food.search.get('nbests',[]):
            key = f"{food.food}~{sr['entity']}"
            sr['sign'] = FOOD_FIX_DB.get(key, {}).get('sign', '~')
            print(f"SR: {sr}; KEY: {key}")

    TAGGER_DB[text] = {
        'text': text,
        'result': jsonable_encoder(res),
    }

    print(f"RES: {res}")

    return res


FOOD_FIX_DB = Database('food-fix.json')
SEARCH_DB = Database('food-search.json')
TAGGER_DB = Database('food-tagger.json')
REQUEST_DB = Database('request.json')


@app.get("/request")
def request(text: str, username: str = Depends(check_credentials)):
    if text in REQUEST_DB:
        REQUEST_DB[text] = REQUEST_DB[text] + 1
    else:
        REQUEST_DB[text] = 1

    print(f"REQUEST: {text}")
    return {'request': text}


@app.get("/sentiment", response_class=HTMLResponse)
async def sentiment_html(request: Request):
    return templates.TemplateResponse("sentiment.html", {"request": request})


@app.get("/mood", response_class=HTMLResponse)
async def mood_html(request: Request):
    return templates.TemplateResponse("mood.html", {"request": request})


@app.get("/food", response_class=HTMLResponse)
async def food_html(request: Request):
    return templates.TemplateResponse("food.html", {"request": request})


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


async def classify(model_name: str, sentence: Sentence):
    sentence.text = clean_spaces(sentence.text)
    classifier = load_model(model_name)

    result = [(c, s) for s, c in classifier.classify(sentence.text)]

    return {
        "text": sentence.text,
        "result": result,
    }


async def fix(model_name: str, sentence: Sentence):
    sentence.text = clean_spaces(sentence.text)
    classifier = load_model(model_name)
    classifier.fix(sentence.text, sentence.label)

    return {
        "text": sentence.text,
        "label": sentence.label,
    }


@app.post("/sentiment", response_class=JSONResponse)
async def classify_sentiment(sentence: Sentence):
    return await classify('sentiment', sentence)


@app.post("/sentiment-fix", response_class=JSONResponse)
async def fix_sentiment(sentence: Sentence):
    return await fix('sentiment', sentence)


@app.post("/mood", response_class=JSONResponse)
async def classify_mood(sentence: Sentence):
    return await classify('mood', sentence)


@app.post("/mood-fix", response_class=JSONResponse)
async def fix_mood(sentence: Sentence):
    return await fix('mood', sentence)


@app.get("/estado_animo", response_class=JSONResponse)
async def estado_animo(text: str):
    sentence = Sentence(text=text)
    tasks = [classify('sentiment', sentence), classify('mood', sentence)]
    sentiment, mood = await asyncio.gather(*tasks)
    sentiment = {m.replace('estado_animo_', ''): c for m, c in sentiment['result']}
    mood = {m.replace('estado_animo_', ''): c for m, c in mood['result']}

    results = {
        'neutral': sentiment.get('neutral', .0),
        'emociones_positivas': sentiment.get('positive', .0) + sentiment.get('negative', .0) * mood.get('emociones_positivas', .0),
    }
    results.update({m: s * sentiment.get('negative', .0) for m, s in mood.items() if m != 'emociones_positivas'})
    print(f"SENTIMENT: {sentiment}")
    print(f"MOOD: {mood}")
    print(f"RESULTS: {results}")

    nbests = [
        {
            'entity': m.replace('_', ' '),
            'score': c,
        }
        for c, m in sorted([(c, m) for m, c in results.items()], reverse=True) if c > 0.01
    ]

    res = {
        "text": sentence.text,
        "nbests": nbests,
    }

    print(f"RES: {text} = {res}")

    return res
