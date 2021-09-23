# QuICT Documentation

## Pre-install

```sh
pip install -U -r requirements.docs.txt
```

## Build docs

```sh
make html
```

## Local host

```sh
python -m http.server -d ./_build/html 8000 -b 0.0.0.0
```