#! /usr/bin/env python

import os

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

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


@app.get("/search/{db}")
def search(db: str, username: str = Depends(check_credentials)):
    return {}
