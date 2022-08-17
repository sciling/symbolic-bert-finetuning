#! /usr/bin/env python
# Based on https://raw.githubusercontent.com/data-henrik/watson-conversation-tool/master/watoolV2.py
# Copyright 2017-2018 IBM Corp. All Rights Reserved.
# See LICENSE for details.
#
# Author: Henrik Loeser
#
# Converse with your assistant based on IBM Watson Assistant service on IBM Cloud.
# See the README for documentation.
#

import datetime
import json
import os
import sys

from collections import OrderedDict
from collections import defaultdict
from graphlib import TopologicalSorter
from pathlib import Path
from textwrap import indent
from typing import List
from typing import Optional
from uuid import uuid4

import requests
import ruamel.yaml
import typer
import unidecode

from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson import AssistantV2
from jinja2 import Environment
from jinja2 import FileSystemLoader
from jinja2 import select_autoescape
from ruamel.yaml import YAML
from ruamel.yaml.representer import RoundTripRepresenter


app = typer.Typer()


# https://stackoverflow.com/a/53875283
class MyRepresenter(RoundTripRepresenter):
    pass


ruamel.yaml.add_representer(OrderedDict, MyRepresenter.represent_dict, representer=MyRepresenter)


def get_assistant_service(config_fn="config.json"):
    # Credentials are read from a file
    with open(config_fn) as file:
        config = json.load(file)
        config_wa = config["credentials"]

    # Initialize the Watson Assistant client, use API V2
    if "apikey" not in config_wa:
        raise ValueError("Expected apikey in credentials.")

    # Authentication via IAM
    authenticator = IAMAuthenticator(config_wa["apikey"])
    print(authenticator)

    assistant_service = AssistantV2(authenticator=authenticator, version=config_wa["versionV2"])
    assistant_service.set_service_url(config_wa["url"])
    return assistant_service, config


def not_match_response(pattern, output, context):
    match_exit_status = ""
    match_error_info = ""
    matches = True
    if not pattern:
        return matches

    correctResponeOutput = get_isCorrectResponse(output, pattern.get('anything_else',False), pattern.get('equal_last_response',False))

    intent = output["intents"][0]
    correct = True
    intentConfidence = intent["confidence"]
    intentname = intent["intent"] 
    if intent["confidence"] < pattern["confidence"]:
        if output["entities"]:
            intent = {
                "intent": "entity_recognition",
                "confidence": 1,
            }
        else:
            intent = {
                "intent": None,
                "confidence": 1 - intent["confidence"],
            }
    print(f"<CORRECT>: {correctResponeOutput} \n")
    intent["correct"] = correctResponeOutput
    
    print(f"<MATCH>: {intent} <-> {pattern}")

    #pattern["correct"] = pattern.get("correct",True)
    if "correct" in pattern and matches:
        matches = matches and pattern["correct"] == correctResponeOutput
        if not matches and not match_exit_status:
            match_exit_status = "\"correct\" doesn't match"
        elif not pattern["correct"] and matches:
            return match_exit_status, match_error_info
    elif not correctResponeOutput:
        matches = False
        match_exit_status = "Incorrect response"
        match_error_info = "Expected a correct output and got correct = False"
        return match_exit_status, match_error_info

    if "confidence" in pattern and matches:
        matches = matches and intent["confidence"] >= pattern["confidence"]
        if not matches and not match_exit_status:
            match_exit_status = "the confidence of the intent \"" + str(intent["intent"]) + "\" is too low"
            match_error_info = "Got " + str(intent["confidence"]) + " and expected >= " + str(pattern["confidence"])

    if "intent" in pattern and matches:
        matches = matches and str(intent["intent"]) == str(pattern["intent"])
        if not matches and not match_exit_status:
            match_exit_status = "\"intent\" doesn't match"
            match_error_info = "Got: " + str(intent["intent"]) + " and expected " + str(pattern["intent"])
            if str(intent["intent"]) == "entity_recognition":
                match_error_info+=f"\nIs recomended to check the CONFIDENCE! Greatest confidence is {intentConfidence} < {pattern['confidence']} for the intent {intentname}"
                match_error_info+="\nso it was transformed into entity_recognition"

    if "entities" in pattern and matches:
        entities = {e["entity"]: e for e in output["entities"]}
        for pat in pattern["entities"]:
            if pat["name"] in entities:
                if "value" in pat:
                    matches = matches and entities[pat["name"]]["value"] == pat["value"]
                    if not matches and not match_exit_status:
                        match_exit_status = "\"entities\" values don't match"
                        match_error_info = "Got " + str(entities[pat["name"]]["value"]) + " and expected " + str(pat["value"])
                if "confidence" in pat:
                    matches = matches and entities[pat["name"]]["confidence"] >= pat["confidence"]
                    if not matches and not match_exit_status:
                        match_exit_status = "\"entities\" confidence is too low"
                        match_error_info = "Got " + str(entities[pat["name"]]["confidence"]) + " and expected >= " + str(pat["confidence"])
            else:
                matches = False
                if not matches and not match_exit_status:
                        match_exit_status = "Entity not found"
                        match_error_info = f"{pat['name']} is not in {entities}"
    matches = matches and context is not None
    if not matches and not match_exit_status:
        match_exit_status = "\"context\" doesn't match, should be None"
    return match_exit_status, match_error_info


def get_text(output):
    generic = get_generic(output)
    if generic != None and len(generic) > 0:
        title = generic[0].get("title", "")
        return "\n".join([title] + [response_to_str(o) for o in get_responses(output)])
    return ""


def get_generic(output):
    generic = output["generic"]

    if len(generic) > 0 and "suggestions" in output["generic"][0]:
        generic = output["generic"][0]["suggestions"][0]["output"]["generic"]

    return generic


def get_responses(output):
    generic = output["generic"][0]

    if "suggestions" in generic:
        return generic["suggestions"][0]["output"]["generic"]

    return [generic]

lastResponse = None
def get_isCorrectResponse(output:dict,anything_else=False, equalLastResp=False):
    actualResponse = None
    for i in range(len(output["generic"])):
        if "text" in output["generic"][len(output["generic"])-i-1]:
            actualResponse = output["generic"][len(output["generic"])-i-1]["text"]
            break
        elif "title" in output["generic"][len(output["generic"])-i-1]:
            actualResponse = output["generic"][len(output["generic"])-i-1]["title"]
            break

    #confidence = output["intents"][0]["confidence"] > 0.5 or output["entities"] != []
    confidence = True
    correct = (equalLastResp or lastResponse != actualResponse)
    correct = correct and 'nodes_visited' in output['debug']
    correct = correct and (anything_else or "anything_else" not in str(output['debug']['nodes_visited'][-1]['conditions']) and 'alternate_responses' not in output)
    
    
    print("Correct answer test status: ", lastResponse, actualResponse , lastResponse != actualResponse,'nodes_visited' in output['debug'] and"anything_else" not in str(output['debug']['nodes_visited'][-1]['conditions']), 'alternate_responses' not in output)
    print(output.keys())
    globals()["lastResponse"] = actualResponse
    
    return correct

def response_to_str(response):
    if "text" in response:
        return response["text"]

    if "options" in response:
        return f"<OPTIONS>: {', '.join(o['label'] for o in response['options'])}"

    return "<UNK>"


# Start a dialog and converse with Watson
def run_entity_test(assistant_service, assistant_id, uuid, entity, test):
    # create a new session
    response = assistant_service.create_session(assistant_id=assistant_id).get_result()
    session_id = response["session_id"]
    print(f"Session created ({session_id})!\n")

    log_dn = f"./logs/{uuid}/{session_id}"
    os.makedirs(log_dn, exist_ok=False)

    good = 0
    total = 0

    error = None
    # Now loop to chat
    for step_n, (label, texts) in enumerate(test.items()):
        print(f"entity {entity} -> {label}: {texts}\n")
        total += len(texts)

        for minput in texts:
            # if we catch a "bye" then exit after deleting the session
            if minput == "bye":
                response = assistant_service.delete_session(assistant_id=assistant_id, session_id=session_id).get_result()
                print("Session deleted. Bye...")
                break

            # send the input to Watson Assistant
            # Set alternate_intents to False for less output
            resp = assistant_service.message(
                assistant_id=assistant_id,
                session_id=session_id,
                input={"text": minput, "options": {"alternate_intents": True, "return_context": True, "debug": True}},
            ).get_result()

            # Dump the returned answer
            with open(f"{log_dn}/{step_n:03d}-response.json", "w") as file:
                print(f"logging step response into {file.name} ...")
                json.dump(resp, file, indent=2, ensure_ascii=False)

            output = resp["output"]
            print(f"<OUTPUT ENTITIES>: {', '.join([e['entity'] + ':' + e['value'] + ':' + str(e['confidence']) for e in output['entities']])}")

            has_intent = output["intents"][0]["confidence"] >= 0.5
            entity_value = next((e for e in output["entities"] if e["entity"] == entity), None)
            print(f"{entity}: {label}: {minput} -> {entity_value['value'] if entity_value else None}")
            if entity_value and entity_value["value"] == label and not has_intent:
                good += 1
    # response = assistant_service.delete_session(assistant_id=assistant_id, session_id=session_id).get_result()

    if total == 0:
        return None

    if good != total:
        error = f"Accuracy for @{entity}: {good} / {total} = {100*good/total:.1f}%"
    return error


# Start a dialog and converse with Watson
def run_test(assistant_service, assistant_id, uuid, test):

    globals()["lastResponse"] = None
    context = test.get("initial_context", {}).copy()

    # create a new session
    response = assistant_service.create_session(assistant_id=assistant_id).get_result()
    session_id = response["session_id"]
    print("Session created!\n")
    error = None

    log_dn = f"./logs/{uuid}/{session_id}"
    os.makedirs(log_dn, exist_ok=False)

    # Now loop to chat
    for step_n, step in enumerate(test["steps"]):
        print(f"step {step_n}: {step}\n")
        # get some input
        minput = step["query"]

        # send the input to Watson Assistant
        # Set alternate_intents to False for less output
        resp = assistant_service.message(
            assistant_id=assistant_id,
            session_id=session_id,
            input={"text": minput, "options": {"alternate_intents": True, "return_context": True, "debug": True}},
        ).get_result()

        # Save returned context for next round of conversation
        if "context" in resp:
            context = resp["context"]

        # Dump the returned answer
        with open(f"{log_dn}/{step_n:03d}-response.json", "w") as file:
            print(f"logging step response into {file.name} ...")
            json.dump(resp, file, indent=2, ensure_ascii=False)

        # Dump the returned answer
        with open(f"{log_dn}/{step_n:03d}-context.json", "w") as file:
            print(f"logging step context into {file.name} ...\n")
            json.dump(context, file, indent=2, ensure_ascii=False)

        output = resp["output"]
        # Client actions from original script
        # if "actions" in output and len(output["actions"]) and output["actions"][0]["type"] == "client":
        #     # Dump the returned answer
        #     if not output_only:
        #         print("")
        #         print("Full response object of intermediate step:")
        #         print("------------------------------------------")
        #         print(json.dumps(resp, indent=2, ensure_ascii=False))

        #     if hca is not None:
        #         context_new = hca.handleClientActions(context, output["actions"], resp)

        #         # call Watson Assistant with result from client action(s)
        #         resp = assistant_service.message(
        #             assistant_id=assistant_id,
        #             session_id=session_id,
        #             input={"text": minput, "options": {"alternate_intents": True, "return_context": True, "debug": True}},
        #             intents=output["intents"],
        #             context=context_new,
        #         ).get_result()
        #         context = resp["context"]
        #         output = resp["output"]
        #     else:
        #         print("\n\nplease use -actionmodule to define module to handle client actions")
        #         break

        
        text = get_text(output)
        print(f"> watson: {indent(text, prefix='          ').strip()}")
        print(f"> user:   {minput}\n")
        print(f"<INTENTS>: {', '.join([i['intent'] + ':' + str(i['confidence']) for i in output['intents']])}")
        print(f"<ENTITIES>: {', '.join([e['entity'] + ':' + e['value'] + ':' + str(e['confidence']) for e in output['entities']])}")

        print("\n")
        not_match_result, error_msg = not_match_response(step.get("response", None), output, context)
        if bool(not_match_result):
            print("\n")
            error = f"Error in step {step_n} with query '{minput}':\n{not_match_result}. {error_msg}."
            break
        print("---------------------------------------------------------------------\n")

    # if we catch a "bye" then exit after deleting the session
    response = assistant_service.delete_session(assistant_id=assistant_id, session_id=session_id).get_result()
    print("Session deleted. Bye...")
    return error

@app.command(name="test")
def test_cmd(test_fn: Path, enabled_tests: Optional[List[str]] = typer.Argument(None)):
    typer.echo(f"Processing {test_fn} -> {enabled_tests}")

    # load configuration and initialize Watson
    assistant_service, _ = get_assistant_service()

    yaml = YAML(typ="safe")  # default, if not specfied, is 'rt' (round-trip)
    with open(test_fn) as file:
        tests = yaml.load(file)

    uuid = uuid4()

    results = []

    for entity, test in tests["entities"].items():
        if enabled_tests and entity not in enabled_tests:
            continue
        reason = run_entity_test(assistant_service, tests["config"]["assistant_id"], uuid, entity, test)
        if reason:
            formatted = typer.style(f"✗ test for entity '@{entity}' failed: {reason}", fg=typer.colors.RED, bold=True)
        else:
            formatted = typer.style(f"✔ test for entity '@{entity}' succeeded.", fg=typer.colors.GREEN, bold=True)
        results.append(formatted)
        typer.echo(formatted)

    for test in tests["dialogs"]:
        if enabled_tests and test["name"] not in enabled_tests:
            continue
        reason = run_test(assistant_service, tests["config"]["assistant_id"], uuid, test)
        if reason:
            formatted = typer.style(f"✗ test for dialog '{test['name']}' failed: {reason}", fg=typer.colors.RED, bold=True)
        else:
            formatted = typer.style(f"✔ test for dialog '{test['name']}' succeeded.", fg=typer.colors.GREEN, bold=True)
        results.append(formatted)
        typer.echo(formatted)

    typer.echo(typer.style("\nSummary:", fg=typer.colors.WHITE, bold=True))
    for test in results:
        typer.echo(test)


def simplify_entities(entities):
    simplified = {}
    for entity in entities:
        simple = {
            "values": {v["value"]: v["synonyms"] if "synonyms" in v else [] for v in entity["values"]},
        }
        if "fuzzy_match" in entity:
            simple["fuzzy_match"] = entity["fuzzy_match"]

        simplified[entity["entity"]] = simple
    return simplified


def prepare_entities(entities):
    wa_entities = []
    for entity, config in entities.items():
        entity = {
            "entity": entity,
            "fuzzy_match": config.get("fuzzy_match", False),
            "values": [
                {
                    "type": "synonyms",
                    "value": entity,
                    "synonyms": synonyms,
                }
                for entity, synonyms in config.get("values", {}).items()
            ],
        }
        wa_entities.append(entity)
    return wa_entities


def simplify_intents(intents):
    simplified = {v["intent"]: {"description": v.get("description", None), "examples": [e["text"] for e in v.get("examples", [])]} for v in intents}
    return simplified


def prepare_intents(intents):
    wa_entities = []
    for intent, examples in intents.items():
        wa_entities.append(
            {
                "intent": intent,
                "examples": [{"text": text} for text in examples],
            }
        )
    return wa_entities


def sort_nodes(nodes):
    ids = {node["dialog_node"]: node for node in nodes}
    prevs = {node["previous_sibling"] for node in nodes if "previous_sibling" in node}

    # print(prevs)
    sorted_nodes = []
    while True:
        last = next((node for node in nodes if node["dialog_node"] not in prevs and node["dialog_node"] in ids))
        sorted_nodes.append(last)
        if "previous_sibling" not in last:
            break

        # print(f"{last['dialog_node']} <- {last['previous_sibling']}")
        prevs.remove(last["previous_sibling"])
        del ids[last["dialog_node"]]
        last = ids[last["previous_sibling"]]

    sorted_nodes.reverse()
    # print(f"SORTED: {[n['dialog_node'] for n in sorted_nodes]}")
    return sorted_nodes


# https://stackoverflow.com/a/20254842
def replace_recursively(search_dict, field, replacer):
    """
    Takes a dict with nested lists and dicts,
    and searches all dicts for a key of the field
    provided.
    """
    if isinstance(search_dict, (list, tuple)):
        for item in search_dict:
            if isinstance(item, dict):
                replace_recursively(item, field, replacer)

    elif isinstance(search_dict, dict):
        for key, value in search_dict.items():

            if key == field:
                search_dict[key] = replacer(value)

            elif isinstance(value, dict):
                replace_recursively(value, field, replacer)

            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, dict):
                        replace_recursively(item, field, replacer)


def title_to_dialog_node(title):
    return "".join(ch for ch in unidecode.unidecode(title.lower().replace(" ", "_")) if ch.isalnum() or ch == "_")


def do_rename_nodes(nodes):
    readable_nodes = {}
    for node in nodes:
        if "title" in node:
            readable_nodes[node["dialog_node"]] = title_to_dialog_node(node["title"])

    replace_recursively(nodes, "dialog_node", lambda x: readable_nodes.get(x, x))
    replace_recursively(nodes, "previous_sibling", lambda x: readable_nodes.get(x, x))

    print(f"READABLE: {readable_nodes}")


def plain_to_tree(nodes):
    tree = []
    children = defaultdict(list)

    for node in nodes:
        if "parent" in node:
            children[node["parent"]].append(node)
            del node["parent"]
        else:
            tree.append(node)

    for node in nodes:
        if node["dialog_node"] in children:
            node["children"] = sort_nodes(children[node["dialog_node"]])

    return tree


def get_graph(nodes, graph, ids):
    if isinstance(nodes, dict):
        nodes = nodes.get("children", [])

    for node in nodes:
        ids[node["dialog_node"]] = node
        graph[node["dialog_node"]] = [subnode["dialog_node"] for subnode in node.get("children", [])]
        get_graph(node, graph, ids)


def tree_to_plain(dialog_nodes):
    ids = {}
    graph = {}
    get_graph(dialog_nodes, graph, ids)
    sorter = TopologicalSorter(graph)
    order = tuple(sorter.static_order())
    # print(order, graph)

    wa_dialog_nodes = [ids[n] for n in order]
    for node in wa_dialog_nodes:
        if "children" in node:
            for child in node["children"]:
                child["parent"] = node["dialog_node"]
            del node["children"]

    return wa_dialog_nodes


def simplify_dialog_nodes(nodes):
    stats = defaultdict(lambda: defaultdict(int))

    for node in nodes:
        stats["type"][node["type"]] += 1
        stats["title"][node["title"] if "title" in node else None] += 1

    print(stats)
    if stats["title"][None] > 0:
        print(f"Warning: There are {stats['title'][None]} dialog nodes without explicit title.")
    for title, num in stats["title"].items():
        if title is not None and num > 1:
            print(f"Warning: Title '{title}' appears {num} times. It should be unique.")

    return nodes


def expand_question(node):
    expanded = node
    return expanded


expansions = {
    "question": expand_question,
}


def expand_node(node):
    if node["type"] in expansions:
        return expansions[node["type"]](node)

    return node


def prepare_dialog_nodes(dialog_nodes):
    return dialog_nodes


def simplify_counterexamples(counterexamples):
    return counterexamples


def prepare_counterexamples(counterexamples):
    return counterexamples


@app.command()
def parse(dialog_fn: str, simplify: bool = False, rename_nodes: bool = False, make_tree: bool = True):
    yaml = YAML(typ="safe")  # default, if not specfied, is 'rt' (round-trip)

    with open(dialog_fn) as file:
        wa_dialog = yaml.load(file)

    dialog = OrderedDict([(k, wa_dialog[k]) for k in ("name", "description", "entities", "intents", "dialog_nodes")])
    if simplify:
        dialog["simplified"] = True
        dialog["entities"] = simplify_entities(dialog["entities"])
        dialog["intents"] = simplify_intents(dialog["intents"])
        dialog["dialog_nodes"] = simplify_dialog_nodes(dialog["dialog_nodes"])
        dialog["counterexamples"] = simplify_counterexamples(dialog.get("counterexamples", []))

    if rename_nodes:
        do_rename_nodes(dialog["dialog_nodes"])

    if make_tree:
        dialog["dialog_nodes_as_tree"] = True
        dialog["dialog_nodes"] = plain_to_tree(dialog["dialog_nodes"])

    # Keep the order of the keys in ordereddict
    yaml = YAML()
    yaml.Representer = MyRepresenter
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.dump(dialog, sys.stdout)


@app.command()
def prepare(template_fn: str, dialog_fn: str):
    yaml = YAML(typ="safe")  # default, if not specfied, is 'rt' (round-trip)

    with open(template_fn) as file:
        wa_dialog = yaml.load(file)

    with open(dialog_fn) as file:
        dialog = yaml.load(file)

    for k in ("name", "description", "entities", "intents", "dialog_nodes"):
        wa_dialog[k] = dialog[k]

    if dialog.get("simplified", False):
        wa_dialog["entities"] = prepare_entities(dialog["entities"])
        wa_dialog["intents"] = prepare_intents(dialog["intents"])
        wa_dialog["dialog_nodes"] = prepare_dialog_nodes(dialog["dialog_nodes"])
        wa_dialog["counterexamples"] = prepare_counterexamples(dialog.get("counterexamples", []))

    if dialog.get("dialog_nodes_as_tree", False):
        wa_dialog["dialog_nodes"] = tree_to_plain(dialog["dialog_nodes"])

    # Keep the order of the keys in ordereddict
    json.dump(wa_dialog, sys.stdout, indent=2, ensure_ascii=False)


@app.command()
def apply(config_fn: str, to_json: bool = False, flatten: bool = False):
    yaml = YAML(typ="safe")  # default, if not specfied, is 'rt' (round-trip)

    env = Environment(
        extensions=[
            "jinja2.ext.do",
            "jinja2_ansible_filters.AnsibleCoreFiltersExtension",
        ],
        loader=FileSystemLoader("templates"),
        autoescape=select_autoescape(),
    )
    env.policies["json.dumps_kwargs"] = {"ensure_ascii": False, "sort_keys": True}

    with open(config_fn) as file:
        config = yaml.load(file)

    template = env.get_template(config.get("template"))

    result = template.render(**config)
    result = yaml.load(result)

    if flatten:
        result = tree_to_plain([result])

    if to_json:
        json.dump(result, sys.stdout, indent=2, ensure_ascii=False)
    else:
        # Keep the order of the keys in ordereddict
        yaml = YAML()
        yaml.Representer = MyRepresenter
        yaml.indent(mapping=2, sequence=4, offset=2)
        yaml.dump(result, sys.stdout)


def call_ibm_cloud_api(assistant_sid, url, data=None, headers=None, params=None):
    # place to set default headers
    _headers = {}
    if headers:
        _headers.update(headers)

    # place to set default params
    _params = {}

    if params:
        _params.update(params)

    cookies = {
        "assistant.sid": assistant_sid,
    }

    if data is not None:
        # print('POST', url, data, _headers, _params, cookies)
        res = requests.post(url, json=data, headers=_headers, params=_params, cookies=cookies)
    else:
        # print('GET', url, _headers, _params, cookies)
        res = requests.get(url, headers=headers, params=params, cookies=cookies)

    return res.json()


@app.command()
def logs(chatbot_id: str):
    # Credentials are read from a file
    with open("config.json") as file:
        config = json.load(file)
        assistant_sid = config["credentials"]["assistant.sid"]
        assistant_id = config["chatbots"][chatbot_id]["assistant_id"]

    data = {
        "type": "conversationLogs",
        "options": {
            "language": "es",
            "assistant_id": assistant_id,
            "version": "v2",
            "start_time": "2022-03-01T00:00:00.000Z",
            "end_time": datetime.datetime.now().isoformat(),
            "interval": "1d",
            "time_zone": "+02:00",
            "order": "desc",
            "intents": [],
            "sample_size": 300,
            "topics": [],
            "conversational_skill_state": "coexistence",
        },
    }
    url = f"https://eu-de.assistant.watson.cloud.ibm.com/rest/v2/tooling/assistants/{assistant_id}/reports"
    response = call_ibm_cloud_api(assistant_sid, url, data=data)
    print(json.dumps(response, indent=2))


@app.command()
def update_dialogs():

    # Credentials are read from a file
    with open("config.json") as file:
        config = json.load(file)
        assistant_sid = config["credentials"]["assistant.sid"]
        chatbots = config["chatbots"]

    params = {
        "include_audit": "true",
        "verbose": "false",
        "export": "true",
        "sort": "stable",
    }

    for name, chatbot in chatbots.items():
        url = f'https://eu-de.assistant.watson.cloud.ibm.com/rest/v2/skills/{chatbot["skill_id"]}'
        filename = f"dialogs/{name}.json"
        print(f"Downloading chatbot {name} from {url} into {filename}")
        response = call_ibm_cloud_api(assistant_sid, url, params=params)
        with open(filename, "w") as file:
            json.dump(response, file, indent=2)


def create_auto_test(name, chatbot):
    filename = f"dialogs/{name}.json"
    with open(filename) as file:
        dialog = json.load(file)

    entities = {}
    for entity in dialog["workspace"]["entities"]:
        if entity["entity"].startswith("sys-"):
            continue

        ent_test = {v["value"]: v["synonyms"][:10] if "synonyms" in v else [] for v in entity["values"]}
        entities[entity["entity"]] = ent_test

    test = {
        "id": name,
        "name": name,
        "config": {"assistant_id": chatbot["assistant_id"]},
        "entities": entities,
    }

    return test


@app.command()
def update_auto_tests():

    update_dialogs()

    # Credentials are read from a file
    with open("config.json") as file:
        config = json.load(file)
        chatbots = config["chatbots"]

    for name, chatbot in chatbots.items():
        test = create_auto_test(name, chatbot)
        test_fn = f"tests/auto-{name}.yaml"
        with open(test_fn, "w") as file:
            yaml = YAML()
            yaml.Representer = MyRepresenter
            yaml.indent(mapping=2, sequence=4, offset=2)
            yaml.dump(test, file)


if __name__ == "__main__":
    app()