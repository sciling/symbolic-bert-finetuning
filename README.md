# ibm_watson_test

config.json
```json
{
    "credentials": {
        "version": "2017-05-26",
        "versionV2": "2018-11-08",
        "apikey": "XXXXXX",  # pragma: allowlist secret
        "url": "https://XXXXXX.watson.cloud.ibm.com/instances/XXXXXX"
    }
}
```

```bash
make virtual-environment
poetry run src/bin/watool.py test tests/basic-test.yaml
```

Convert template into watson assistant compatible json:
```bash
poetry run src/bin/watool.py apply chatbots/cu2-contamos-contigo/sections/faq.yml --flatten --to-json
```
