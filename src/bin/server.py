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


class Sentence(BaseModel):
    text: str
    label: Optional[str]


classifier = Classifier('./train.dir')
try:
    with open("sentiment-database.json") as file:
        sentiment = json.load(file)
except:
    sentiment = {}


@app.post("/sentiment", response_class=JSONResponse)
async def classify_sentiment(sentence: Sentence):
    sentence.text = clean_spaces(sentence.text)

    if sentence.text in sentiment:
        part = {"estado_animo_positive": 0, "estado_animo_neutral": 0, "estado_animo_negative": 0}
        part[sentiment[sentence.text]] = 1.0
        result = [(c, s) for s, c in sorted([(s, c) for c, s in part.items()], reverse=True)]
    else:
        result = [(c, s) for s, c in classifier.classify(sentence.text)]
        sentiment[sentence.text] = result[0][0]

        with open("sentiment-database.json", "w") as file:
            json.dump(sentiment, file, indent=2, ensure_ascii=False)

    return {
        "text": sentence.text,
        "result": result,
    }


@app.post("/sentiment-fix", response_class=JSONResponse)
async def fix_sentiment(sentence: Sentence):
    sentiment[sentence.text] = sentence.label

    with open("sentiment-database.json", "w") as file:
        json.dump(sentiment, file, indent=2, ensure_ascii=False)

    return {
        "text": sentence.text,
        "label": sentence.label,
    }
