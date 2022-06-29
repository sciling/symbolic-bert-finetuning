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

import json
import os

from textwrap import indent
from uuid import uuid4

import typer

from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson import AssistantV2
from ruamel.yaml import YAML


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

    assistant_service = AssistantV2(authenticator=authenticator, version=config_wa["versionV2"])
    assistant_service.set_service_url(config_wa["url"])
    return assistant_service


def match_response(pattern, output, context):
    matches = True
    if not pattern:
        return matches

    intent = output["intents"][0]
    if intent["confidence"] < 0.5:
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

    print(f"<MATCH>: {intent} <-> {pattern}")
    if "intent" in pattern:
        matches = matches and intent["intent"] == pattern["intent"]

    if "confidence" in pattern:
        matches = matches and intent["confidence"] >= pattern["confidence"]

    if "entities" in pattern:
        entities = {e["entity"]: e for e in output["entities"]}
        for pat in pattern["entities"]:
            if pat["name"] in entities:
                if "value" in pat:
                    matches = matches and entities[pat["name"]]["value"] == pat["value"]
                if "confidence" in pat:
                    matches = matches and entities[pat["name"]]["confidence"] >= pat["confidence"]
            else:
                matches = False

    matches = matches and context is not None

    return matches


def get_text(output):
    generic = get_generic(output)[0]
    title = generic.get("title", "")
    return "\n".join([title] + [response_to_str(o) for o in get_responses(output)])


def get_generic(output):
    generic = output["generic"]

    if "suggestions" in generic[0]:
        generic = generic["suggestions"][0]["output"]["generic"]

    return generic


def get_responses(output):
    generic = output["generic"][0]

    if "suggestions" in generic:
        return generic["suggestions"][0]["output"]["generic"]

    return [generic]


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
    print("Session created!\n")

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
            print(f"<ENTITIES>: {', '.join([e['entity'] + ':' + e['value'] + ':' + str(e['confidence']) for e in output['entities']])}")

            has_intent = output["intents"][0]["confidence"] >= 0.5
            entity_value = next((e for e in output["entities"] if e["entity"] == entity), None)
            print(f"{entity}: {label}: {minput} -> {entity_value['value'] if entity_value else None}")
            if entity_value and entity_value["value"] == label and not has_intent:
                good += 1
    response = assistant_service.delete_session(assistant_id=assistant_id, session_id=session_id).get_result()

    if good != test:
        error = f"Accuracy for @{entity}: {good} / {total} = {100*good/total:.1f}%"
    return error


# Start a dialog and converse with Watson
def run_test(assistant_service, assistant_id, uuid, test):
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

        print(f"> user:   {minput}")
        text = get_text(output)
        print(f"> watson: {indent(text, prefix='          ').strip()}\n")
        print(f"<INTENTS>: {', '.join([i['intent'] + ':' + str(i['confidence']) for i in output['intents']])}")
        print(f"<ENTITIES>: {', '.join([e['entity'] + ':' + e['value'] + ':' + str(e['confidence']) for e in output['entities']])}")

        print("\n")
        if not match_response(step.get("response", None), output, context):
            print("\n")
            error = f"error in step {step_n} with query '{minput}'"
            break
        print("\n")

    # if we catch a "bye" then exit after deleting the session
    response = assistant_service.delete_session(assistant_id=assistant_id, session_id=session_id).get_result()
    print("Session deleted. Bye...")
    return error


def main(test_fn: str):
    typer.echo(f"Processing {test_fn}")

    # load configuration and initialize Watson
    assistant_service = get_assistant_service()

    yaml = YAML(typ="safe")  # default, if not specfied, is 'rt' (round-trip)
    with open(test_fn) as file:
        tests = yaml.load(file)

    uuid = uuid4()

    results = []

    for entity, test in tests["entities"].items():
        reason = run_entity_test(assistant_service, tests["config"]["assistant_id"], uuid, entity, test)
        if reason:
            formatted = typer.style(f"✗ test for entity '@{entity}' failed: {reason}", fg=typer.colors.RED, bold=True)
        else:
            formatted = typer.style(f"✔ test for entity '@{entity}' succeeded.", fg=typer.colors.GREEN, bold=True)
        results.append(formatted)
        typer.echo(formatted)

    for test in tests["dialogs"]:
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


if __name__ == "__main__":
    typer.run(main)
