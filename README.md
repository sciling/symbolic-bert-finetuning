# ibm_watson_test

config.json
```json
{
    "credentials": {
        "version": "2017-05-26",
        "versionV2": "2018-11-08",
        "apikey": "XXXXXX",
        "url": "https://XXXXXX.watson.cloud.ibm.com/instances/XXXXXX"
    }
}
```

```
make virtual-environment
poetry run src/bin/watool.py tests/basic-test.yaml
```

