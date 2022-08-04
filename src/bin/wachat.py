from csv import excel
import watool as WA
from ruamel.yaml import YAML
from textwrap import indent


def get_text(output, id):
    generic = WA.get_generic(output)[id]
    title = generic.get("title", "")
    return "\n".join([title] + [WA.response_to_str(o) for o in WA.get_responses(output)])

def print_responses(output):
    for i in output["generic"]:

        text = "\n" + i["text"]
        print(f"> watson: {indent(text, prefix='          ').strip()}")

    print("")
    print(f"<OUTPUT INTENTS>: {', '.join([i['intent'] + ':' + str(i['confidence']) for i in output['intents']])}")
    print(f"<OUTPUT ENTITIES>: {', '.join([e['entity'] + ':' + e['value'] + ':' + str(e['confidence']) for e in output['entities']])}\n")


def run_chat():
    assistant, _  = WA.get_assistant_service()

    good_CU = False

    while not good_CU:
        cu_fn = input("Inserta fichero de caso de uso: ")
        if cu_fn == "bye":
            exit(1)
        try:
            cu_int = int(cu_fn)
            cu_fn = "tests/CU" + str(cu_int) + ".yaml"
            yaml = YAML(typ="safe")
            with open(cu_fn) as file:
                cu_yaml = yaml.load(file)
                cu_id = cu_yaml["config"]["assistant_id"]
            good_CU = True
        except:
            pass



    response = assistant.create_session(assistant_id=cu_id).get_result()
    session_id = response["session_id"]
    print(f"Session created ({session_id})!\n")

    while True:

        minput = input("> ")
                # if we catch a "bye" then exit after deleting the session
        if minput == "bye":
            response = assistant.delete_session(assistant_id=cu_id, session_id=session_id).get_result()
            print("Session deleted. Bye...")
            break

        # send the input to Watson Assistant
        # Set alternate_intents to False for less output
        resp = assistant.message(
            assistant_id=cu_id,
            session_id=session_id,
            input={"text": minput, "options": {"alternate_intents": True, "return_context": True, "debug": True}},
        ).get_result()

        output = resp["output"]
        print_responses(output)




if __name__ == "__main__":
    run_chat()
    
