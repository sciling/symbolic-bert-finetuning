#! /usr/bin/env python

import csv
import json
from pathlib import Path
from typing import List
import random

import typer
from ruamel.yaml import YAML
from transformers import pipeline
import torch
from torch.nn import CosineSimilarity
from transformers import AutoTokenizer, BertForSequenceClassification, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_metric, load_dataset
from spellchecker import SpellChecker
import spacy
from spacy.attrs import LOWER, POS, ENT_TYPE, IS_ALPHA, DEP, LEMMA, LOWER, IS_PUNCT, IS_DIGIT, IS_SPACE, IS_STOP
from spacy.tokens.doc import Doc
import numpy as np
from torch import nn

from medp.utils import EmbeddingsProcessor

app = typer.Typer()

cos = CosineSimilarity(dim=1, eps=1e-6)

spanish_bert = 'dccuchile/bert-base-spanish-wwm-uncased'
spanish_zeroshot = 'Recognai/zeroshot_selectra_medium'


# nlp = spacy.load('es_dep_news_trf')
nlp = spacy.load('es_core_news_lg')


def p(doc):
    print([f"{t.text}:{t.lemma_}:{t.pos_}:{t.dep_}:{t.head.text}->{[child for child in t.children]}" for t in doc])


class Classifier:
    def __init__(self, model_name, is_multilabel=False, database_fn=None, spellcheck=True, **kwargs):
        self.model_name = model_name
        self.database_fn = database_fn
        self.spellchecker = SpellChecker(language='es') if spellcheck else None

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if is_multilabel:
            print('multi_label_classification')
            self.model = BertForSequenceClassification.from_pretrained(model_name, problem_type="multi_label_classification")
        else:
            print('single_label_classification')
            self.model = BertForSequenceClassification.from_pretrained(model_name)
        self.labels = [label for _, label in sorted(self.model.config.id2label.items())]

        self.database = None
        if database_fn:
            try:
                with open(database_fn) as file:
                    self.database = json.load(file)
            except:
                self.database = {}

    def fix(self, text, label):
        if self.database is not None:
            self.database[text] = label

            with open(self.database_fn, "w") as file:
                json.dump(self.database, file, indent=2, ensure_ascii=False)

    def classify(self, text):
        # If literal match is found return from database.
        if self.database is not None and text in self.database:
            part = {k: .0 for k in self.labels}
            part[self.database[text]] = 1.0
            return sorted([(s, c) for c, s in part.items()], reverse=True)

        if self.spellchecker:
            text = ' '.join(self.spellchecker.correction(d.text) for d in nlp(text))

        # If spell corrected match is found return from database.
        if self.database is not None and text in self.database:
            part = {k: .0 for k in self.labels}
            part[self.database[text]] = 1.0
            return sorted([(s, c) for c, s in part.items()], reverse=True)

        # Otherwise, classify
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**inputs).logits.flatten()
            proba = [p.item() for p in nn.functional.softmax(logits, dim=0)]
        res = sorted(zip(proba, self.labels), reverse=True)
        self.fix(text, res[0][1])
        return res


def remove_tokens(doc, index_to_del, list_attr=[LOWER, POS, ENT_TYPE, IS_ALPHA, DEP, LEMMA, LOWER, IS_PUNCT, IS_DIGIT, IS_SPACE, IS_STOP]):
    """
    Remove tokens from a Spacy *Doc* object without losing
    associated information (PartOfSpeech, Dependance, Lemma, extensions, ...)

    Parameters
    ----------
    doc : spacy.tokens.doc.Doc
        spacy representation of the text
    index_to_del : list of integer
         positions of each token you want to delete from the document
    list_attr : list, optional
        Contains the Spacy attributes you want to keep (the default is
        [LOWER, POS, ENT_TYPE, IS_ALPHA, DEP, LEMMA, LOWER, IS_PUNCT, IS_DIGIT, IS_SPACE, IS_STOP])
    Returns
    -------
    spacy.tokens.doc.Doc
        Filtered version of doc
    """

    np_array = doc.to_array(list_attr) # Array representation of Doc

    # Creating a mask: boolean array of the indexes to delete
    mask_to_del = np.ones(len(np_array), np.bool)
    mask_to_del[index_to_del] = 0

    np_array_2 = np_array[mask_to_del]
    doc2 = Doc(doc.vocab, words=[t.text for t in doc if t.i not in index_to_del])
    doc2.from_array(list_attr, np_array_2)

    # Handling user extensions
    #  The `doc.user_data` dictionary is holding the data backing user-defined attributes.
    #  The data is based on characters offset, so a conversion is needed from the
    #  old Doc to the new one.
    #  More info here: https://github.com/explosion/spaCy/issues/2532
    arr = np.arange(len(doc))
    new_index_to_old = arr[mask_to_del]
    doc_offset_2_token = {tok.idx : tok.i  for tok in doc}  # needed for the user data
    doc2_token_2_offset = {tok.i : tok.idx  for tok in doc2}  # needed for the user data
    new_user_data = {}
    for ((prefix, ext_name, offset, x), val) in doc.user_data.items():
        old_token_index = doc_offset_2_token[offset]
        new_token_index = np.where(new_index_to_old == old_token_index)[0]
        if new_token_index.size == 0:  # Case this index was deleted
            continue
        new_char_index = doc2_token_2_offset[new_token_index[0]]
        new_user_data[(prefix, ext_name, new_char_index, x)] = val
    doc2.user_data = new_user_data

    return doc2


@app.command()
def entities_to_dataset(entities_fn: Path, train_fn: Path, dev_fn: Path, test_fn: Path, dev: float = 0.05, test: float = 0.05, max_examples: int = 0, entities: str = typer.Option(None), randomize: bool = True):
    if entities:
        entities = set(entities.split(','))
        print(f"Filter entities: {entities}")

    data = []
    with open(entities_fn) as fileread:
        csvreader = csv.reader(fileread, delimiter=',', quotechar='"')
        idx = 0
        for row in csvreader:
            if not entities or row[1] in entities:
                intent_name = f"{row[0]}_{row[1]}".replace(' ', '_')
                for sentence in row[2:]:
                    data.append((sentence, intent_name))
                    idx += 1

    if randomize:
        random.shuffle(data)

    if max_examples:
        data = data[:max_examples]

    n_dev = int(len(data) * dev)
    n_test = int(len(data) * test)
    n_train = len(data) - n_dev - n_test
    print(f"Size train: {n_train}, dev: {n_dev}, test: {n_test}")

    with open(train_fn, "w") as filewrite:
        csvwriter = csv.writer(filewrite, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
        csvwriter.writerow(['sentence', 'label'])
        for row in data[:n_train]:
            csvwriter.writerow(row)

    with open(dev_fn, "w") as filewrite:
        csvwriter = csv.writer(filewrite, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
        csvwriter.writerow(['sentence', 'label'])
        for row in data[n_train:n_train + n_dev]:
            csvwriter.writerow(row)

    with open(test_fn, "w") as filewrite:
        csvwriter = csv.writer(filewrite, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
        csvwriter.writerow(['sentence', 'label'])
        for row in data[n_train + n_dev:]:
            csvwriter.writerow(row)


@app.command()
def get_embeddings(sentences: List[str]):
    EmbeddingsProcessor.get_model(spanish_bert)
    embeddings = EmbeddingsProcessor.pages_to_embeddings(sentences)
    scores = cos(embeddings[0], embeddings)
    print(list(zip(sentences, embeddings)), scores)


@app.command()
def few_shot(entities_fn: Path, new_samples: List[str] = typer.Argument(None), interactive: bool = False):
    typer.echo(f"Processing terms {entities_fn}' ...")

    if entities_fn.suffix in ('.yaml', '.yml'):
        yaml = YAML(typ="safe")  # default, if not specfied, is 'rt' (round-trip)
        with open(entities_fn) as file:
            entities = yaml.load(file)

    else:
        entities = {}
        with open(entities_fn) as file:
            csvreader = csv.reader(file, delimiter=',', quotechar='"')
            for row in csvreader:
                entities[row[1]] = list(set(row[2:]))

    samples = [sample for value in entities.values() for sample in value]
    labels = [label for label, value in entities.items() for sample in value]

    n_good = 0
    X = EmbeddingsProcessor.pages_to_embeddings(samples)
    for i, (label, sample) in enumerate(zip(labels, samples)):
        scores = cos(X[i], X)
        index_sorted = torch.argsort(scores)
        output = [(labels[i], scores[i], samples[i]) for i in index_sorted if samples[i] != sample]
        print(f"'{sample}', '{label}', {list(reversed(output))[0:3]}")
        if label == labels[index_sorted[-2]]:
            n_good += 1

    total = len(samples)
    print(f"{n_good} / {total} = {n_good/total}")

    if new_samples:
        n_good = 0
        new_X = EmbeddingsProcessor.pages_to_embeddings(new_samples)
        for i, sample in enumerate(new_samples):
            scores = cos(new_X[i], X)
            index_sorted = torch.argsort(scores)
            output = [(labels[i], scores[i], samples[i]) for i in index_sorted if samples[i] != sample]
            print(sample, list(reversed(output))[0:3])


@app.command()
def zero_shot(entities_fn: Path, multi_label: bool = False):
    typer.echo(f"Processing terms {entities_fn}' ...")

    yaml = YAML(typ="safe")  # default, if not specfied, is 'rt' (round-trip)
    with open(entities_fn) as file:
        entities = yaml.load(file)

    candidate_labels = list(entities.keys())
    samples = [sample for value in entities.values() for sample in value]
    labels = [label for label, value in entities.items() for sample in value]

    classifier = pipeline("zero-shot-classification", model=spanish_zeroshot, device=0)

    res = classifier(
        samples,
        candidate_labels=candidate_labels,
        hypothesis_template="Este ejemplo es {}.",
        multi_class=multi_label
    )

    output = [r['labels'][0] for r in res]
    scores = [r['scores'][0] for r in res]
    print(json.dumps(list(zip(labels, output, scores, samples)), indent=2, ensure_ascii=False))


@app.command()
def train(train_fn: Path, dev_fn: Path, output_dir: Path = 'train.dir', model_name: str = spanish_bert, cache_dir: Path = 'cache.dir', max_seq_length: int = 128, num_train_epochs: int = 3, learning_rate: float = 2e-5, problem_type: str = 'single_label_classification'):

    data_files = {
        'train': str(train_fn),
        'validation': str(dev_fn),
    }

    # Loading a dataset from local csv files
    dataset = load_dataset(
        "csv",
        data_files=data_files,
        cache_dir=str(cache_dir)
    )

    label_list = dataset["train"].unique("label")
    label_list.sort()  # Let's sort it for determinism
    num_labels = len(label_list)
    label_to_id = {v: i for i, v in enumerate(label_list)}

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    max_seq_length = min(max_seq_length, tokenizer.model_max_length)

    def tokenize_function(examples):
        result = tokenizer(examples["sentence"], padding="max_length", max_length=max_seq_length, truncation=True)
        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[label] if label != -1 else -1) for label in examples["label"]]

        return result

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        problem_type=problem_type,
        num_labels=num_labels,
        label2id=label_to_id,
        id2label={id: label for label, id in label_to_id.items()},
        ignore_mismatched_sizes=True
    )
    model.classifier = nn.Linear(model.config.hidden_size, num_labels)
    print(model)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        per_device_train_batch_size=16,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs
    )

    metric = load_metric("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        compute_metrics=compute_metrics,
    )

    train_result = trainer.train()

    metrics = train_result.metrics
    trainer.save_model()  # Saves the tokenizer too for easy upload

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    metrics = trainer.evaluate(eval_dataset=tokenized_datasets['validation'])
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    app()
