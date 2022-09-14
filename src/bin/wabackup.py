#! /usr/bin/env python

import os
import json
from urllib.parse import unquote

import requests
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from xvfbwrapper import Xvfb



def get_assistant_sid(export_dn=None):
    is_chrome = True

    with Xvfb(width=1280, height=740, colordepth=16) as xvfb:

        if is_chrome:
            import chromedriver_autoinstaller
            from selenium.webdriver.chrome.options import Options

            chromedriver_autoinstaller.install()  # Check if the current version of chromedriver exists
                                                  # and if it doesn't exist, download it automatically,
                                                  # then add chromedriver to path

            options = Options()
            # undetectable chrome driver https://stackoverflow.com/a/70709308
            options.add_argument("start-maximized")
            options.add_argument("--window-size=1920,1200")
            options.add_argument("--no-sandbox")
            # options.add_argument("--headless")
            options.add_argument("--disable-gpu")
            options.add_experimental_option("excludeSwitches", ["enable-automation"])
            options.add_experimental_option('excludeSwitches', ['enable-logging'])
            options.add_experimental_option('useAutomationExtension', False)
            options.add_argument('--disable-blink-features=AutomationControlled')

            driver = webdriver.Chrome(options=options)

            driver.execute_cdp_cmd('Network.setUserAgentOverride', {
                "userAgent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/85.0.4183.102 Safari/537.36'
            })

        else:
            from selenium.webdriver.firefox.options import Options
            options = Options()
            driver = webdriver.Firefox(options=options)

        # https://piprogramming.org/articles/How-to-make-Selenium-undetectable-and-stealth--7-Ways-to-hide-your-Bot-Automation-from-Detection-0000000017.html
        # Remove navigator.webdriver Flag using JavaScript
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

        driver.get("https://cloud.ibm.com/")

        def take_screenshot():
            if not hasattr(take_screenshot, 'num_screenshot'):
                take_screenshot.num_screenshot = 0
            else:
                take_screenshot.num_screenshot += 1

            if export_dn:
                print(f"taking screenshot: {export_dn}/{take_screenshot.num_screenshot}.png")
                driver.save_screenshot(f"{export_dn}/{take_screenshot.num_screenshot}.png")

        def get(name, condition=EC.element_to_be_clickable, timeout=10, retry=3):
            take_screenshot()
            element = None

            while retry > 0:
                try:
                    print(f"GET BY CSS ({retry} remaining retries): {name}")
                    element = WebDriverWait(driver, timeout).until(
                        condition((By.CSS_SELECTOR, name))
                    )
                    retry = 0
                except TimeoutException:
                    print(f"Timed out")
                    retry -= 1
                take_screenshot()

            if not element:
                raise Exception(f"Could not get {name}")

            print(f"ELEMENT: {element} {element.get_attribute('outerHTML')[:100]} ...")
            return element

        get('input#userid').send_keys(os.getenv('WATSON_USER'))
        get('button[name="login"]').click()

        get('input#password').send_keys(os.getenv('WATSON_PASSWORD'))
        submit = get('button[name="login"]', EC.presence_of_element_located)
        driver.execute_script("arguments[0].click();", submit)

        get('#main-content')
        driver.get("https://eu-de.assistant.watson.cloud.ibm.com/crn%3Av1%3Abluemix%3Apublic%3Aconversation%3Aeu-de%3Aa%2Fcc77cd568abe4954913135fed5d4d917%3A59a05c17-afcb-4374-afc6-caba0754ee4c%3A%3A/assistants/2fa458ff-8c40-43be-b69c-bbb78ff05dbd/home")

        get('div[role="main"]')

        sid = next((unquote(cookie['value']) for cookie in driver.get_cookies() if cookie['name'] == 'assistant.sid'))

        print(f"SID: {sid}")
        driver.quit()

    return sid


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


def update_dialogs():
    export_dn = './backup'
    os.makedirs(export_dn, exist_ok=True)

    assistant_sid = os.getenv('WATSON_SID', None)
    if assistant_sid is None:
        assistant_sid = get_assistant_sid(export_dn)

    chatbots = {
        "cu1": {
            "assistant_id": "75ee9c37-df5d-47fc-bf2e-123b8aad578d",
            "skill_id": "f0c03c43-7345-49ea-b915-0d04ac3f55d5"
        },
        "cu2": {
            "assistant_id": "fd3e711e-767d-4c31-969b-490f4534ea5b",
            "skill_id": "121c38e0-03d1-43f9-8945-7228dda24260"
        },
        "cu3": {
            "assistant_id": "3b98249b-30c0-4aa0-a784-8c889f77372d",
            "skill_id": "78f31774-0591-4112-96be-5d418a415953"
        },
        "cu4.1": {
            "assistant_id": "4cca232f-80ff-40f6-91b5-f3c9acec4897",
            "skill_id": "61d88831-0573-4bf4-acb3-4735d8798574"
        },
        "cu4.2": {
            "assistant_id": "da990cc8-091b-4964-8af3-b8246e475ecc",
            "skill_id": "3b5384b4-1c02-4e26-80af-15cde28e63ad"
        },
        "cu4.3": {
            "assistant_id": "a7a5904e-b74b-46e0-9984-444daf9e36f7",
            "skill_id": "314b74b7-8d7c-48c1-9460-9bbe63ec2142"
        },
        "cu4.4": {
            "assistant_id": "88c34e3d-6da9-4cef-affe-a1319816ba38",
            "skill_id": "11c1a77d-faec-40e7-a0a5-17df26e2e1ea"
        },
        "cu5": {
            "assistant_id": "f8923f44-6edb-45bb-ba8f-605e4d537826",
            "skill_id": "5da0199d-11c8-48ae-9b79-393bd7d2618b"
        }
    }

    params = {
        "include_audit": "true",
        "verbose": "false",
        "export": "true",
        "sort": "stable",
    }

    for name, chatbot in chatbots.items():
        url = f'https://eu-de.assistant.watson.cloud.ibm.com/rest/v2/skills/{chatbot["skill_id"]}'
        filename = f"{export_dn}/{name}.json"
        print(f"Downloading chatbot {name} from {url} into {filename}")
        response = call_ibm_cloud_api(assistant_sid, url, params=params)
        with open(filename, "w") as file:
            json.dump(response, file, indent=2)


if __name__ == "__main__":
    update_dialogs()
