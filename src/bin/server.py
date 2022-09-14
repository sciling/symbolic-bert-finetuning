#! /usr/bin/env python

import os
import csv
import math
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

default_units = None
if os.path.exists('./train-ma.dir'):
    CACHED['food'] = Tagger('./train-ma.dir', normalization_fn="./multialimento-normalization.json", token_fixes_fn="./multialimento-token-fixes.json", strict_fields={'unit'})
    default_units = {}
    with open('alimentos_unidad_defecto.csv') as file:
        for row in csv.reader(file, delimiter=',', quotechar='"'):
            default_units[row[0]] = row[1]


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
    print(f"PARSEFOOD: {text}: {res}")

    for food in res.get('foods', [])[:]:
        food.search = searcher.search(
            food.food, nbest, summarized=summarized, multinomial=multinomial,
            description_type=description_type, reuse_description=reuse_description,
            fuzzy=fuzzy, max_ngram=max_ngram
        )
        # NOTE: backend only admits ints
        food.quantity = math.ceil(food.quantity) if food.quantity else None
        if 'error' in food.search or not food.search.get('nbests', []):
            food.food = None
            food.search = None
        else:
            if len(food.search.get('nbests', [])) >= 1 and not food.unit:
                entities = [unit['entity'] for unit in food.search['nbests']]
                units = list({default_units[entity] for entity in entities if entity in default_units})
                print("UNITS", entities, units, {entity: default_units.get(entity, None) for entity in entities if entity in default_units})
                if len(units) == 1:
                    food.unit = units[0]

            for sr in food.search.get('nbests', []):
                key = f"{food.food}~{sr['entity']}"
                sr['sign'] = FOOD_FIX_DB.get(key, {}).get('sign', '~')
                print(f"SR: {sr}; KEY: {key}")

        if not food.search and not food.unit and not food.quantity:
            res.get('foods', []).remove(food)

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


@app.get("/multimood", response_class=HTMLResponse)
async def multi_mood_html(request: Request):
    return templates.TemplateResponse("multimood.html", {"request": request})


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
    'multimood': {
        'model': None,
        'model_name': 'train-ml.dir',
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


@app.get("/unsupervised-mood", response_class=JSONResponse)
async def unsupervised_mood():
    unsupervised = set()
    for model_name in ['sentiment', 'mood']:
        classifier = load_model(model_name)
        unsupervised |= classifier.get_unsupervised()

    return sorted(unsupervised)


@app.post("/multimood", response_class=JSONResponse)
async def classify_multimood(sentence: Sentence):
    return await classify('multimood', sentence)


@app.post("/multimood-fix", response_class=JSONResponse)
async def fix_multimood(sentence: Sentence):
    return await fix('multimood', sentence)


@app.get("/unsupervised-multimood", response_class=JSONResponse)
async def unsupervised_multimood():
    unsupervised = set()
    for model_name in ['sentiment', 'multimood']:
        classifier = load_model(model_name)
        unsupervised |= classifier.get_unsupervised()

    return sorted(unsupervised)


@app.get("/estado_animo", response_class=JSONResponse)
async def estado_animo(text: str, nbest: int = 4, threshold: float = 0.01):
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
        for c, m in sorted([(c, m) for m, c in results.items() if c >= threshold], reverse=True)[:nbest]
    ]

    res = {
        "text": sentence.text,
        "nbests": nbests,
    }

    print(f"RES: {text} = {res}")

    return res


@app.get("/estado_animo_multiple", response_class=JSONResponse)
async def estado_animo_multiple(text: str, nbest: int = 4, threshold: float = 0.01):
    sentence = Sentence(text=text)
    tasks = [classify('sentiment', sentence), classify('multimood', sentence)]
    sentiment, mood = await asyncio.gather(*tasks)
    sentiment = {m.replace('estado_animo_', ''): c for m, c in sentiment['result']}
    mood = {m.replace('estado_animo_', ''): c for m, c in mood['result']}

    results = {
        'neutral': sentiment.get('neutral', .0),
        'emociones_positivas': sentiment.get('positive', .0) + sentiment.get('negative', .0) * mood.get('emociones_positivas', .0),
    }
    negative_emotions = {m: s * sentiment.get('negative', .0) for m, s in mood.items() if m != 'emociones_positivas'}
    print(f"NEGATIVE[{sentiment.get('negative', .0)}][{all([s < threshold for s in negative_emotions.values()])}]: {negative_emotions}")
    if sentiment.get('negative', .0) >= threshold and all([s < threshold for s in negative_emotions.values()]):
        negative_emotions = {'emociones_negativas': sentiment.get('negative', .0)}
    results.update(negative_emotions)
    print(f"SENTIMENT: {sentiment}")
    print(f"MOOD: {mood}")
    print(f"RESULTS: {results}")

    nbests = [
        {
            'entity': m.replace('_', ' '),
            'score': c,
        }
        for c, m in sorted([(c, m) for m, c in results.items() if c >= threshold], reverse=True)[:nbest]
    ]

    res = {
        "text": sentence.text,
        "nbests": nbests,
    }

    print(f"RES: {text} = {res}")

    return res
