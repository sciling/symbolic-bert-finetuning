#! /usr/bin/env python

import os

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from medp.utils import SearchEngine

app = FastAPI()

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


@app.get("/search/{entity}/{text}")
def search(entity: str, text: str, nbest: int = 4, username: str = Depends(check_credentials)):
    searcher = SearchEngine(f"{entity}-db.json")
    res = searcher.search(text, nbest)

    return res
