# imports
from bs4 import BeautifulSoup
import json
import random
import re
import requests
import string
from urllib.parse import urlparse, parse_qs

from utils import extract_balanced_json

# constants
SPORK_URL = 'https://spork-repertorium-bd227a1477a1.herokuapp.com'
LOGIN_URL = SPORK_URL + '/auth/sign-in'
CSRF_URL = SPORK_URL + '/api/auth/csrf'
CREDS_URL = SPORK_URL + '/api/auth/callback/credentials'
MANUS_URL = SPORK_URL + '/manuscript'
CHANT_URL = SPORK_URL + '/api/chant-record'
CANTU_API = SPORK_URL + '/api/cantus-index/lookup?cantusId='
INDEX_API = SPORK_URL + '/api/image'


class SporkWorkspaceSession:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers['User-Agent'] = 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:126.0) Gecko/20100101 Firefox/126.0'

    def login_platform(self, user, password):
        """
        Log in to the REPERTORIUM workspace using provided password.
        Parameters:
            user (str): user email
            password (str): password
        Returns:
            bool: True if login is successful, False otherwise
            str: message describing the outcome of login attempt
        """
        page = self.session.get(CSRF_URL)
        csrfToken = page.json()["csrfToken"]
        payload = {
            'redirect': 'false',
            'email': user,
            'password': password,
            'csrfToken': csrfToken,
            'callbackUrl': LOGIN_URL,
            'json': 'true'
        }
        response = self.session.post(CREDS_URL, data=payload)
        result = response.json()
        if 'url' in result and 'error=' in result['url']:
            parsed = urlparse(result['url'])
            error_msg = parse_qs(parsed.query).get('error', ['Unknown error'])[0]
            error_msg = f"Login failed: {error_msg}"
            print(error_msg)
            return False, error_msg
        else:
            error_msg = 'Login succesful'
            return True, error_msg

    def get_manuscript_pagelist(self, manuscriptID):
        """
        Get list of pages of a given manuscript.
        Parameters:
            manuscriptID (str): manuscript identifier
        Returns:
            list: list of JSON objects with pages (can be empty)
        """
        pagenum = 0
        data = []
        while True:
            page = self.session.get(f"{MANUS_URL}/{manuscriptID}?page={pagenum}")
            text = page.text
            match = re.search(r'(\\"images\\":\s*\[.*?\])' , text)
            if not match:
                break
            # json_str = match.group(1)
            json_str = extract_balanced_json(' {' + text[match.start():])
            json_str = json_str.replace('\\"', '"')
            # json_str = '{' + json_str + '}'
            try:
                datapage = json.loads(json_str)
                datapage = datapage['images']
                data.extend(datapage)
            except json.JSONDecodeError as e:
                print("Error when parsing JSON:", e)
                break
            if len(datapage) == 0:
                break
            pagenum += 1
        return data

    def get_annotations_from_page(self, manuscriptID, imageID):
        """
        Get annotations from a specific page of a manuscript.
        Parameters:
            manuscriptID (str): manuscript identifier
            imageID (str): image identifier
        Returns:
            dict: JSON object with annotations (can be empty)
        """
        page = self.session.get(f"{MANUS_URL}/{manuscriptID}/{imageID}")
        html = page.text
        parser = BeautifulSoup(html, "html.parser")
        scripts = parser.find_all("script")

        # find the data
        data = {}
        for script in scripts:
            if not script.string or not "self.__next_f.push" in script.string:
                continue
            # get annotations
            match = re.search(r'({\\"image\\":{)', script.string)
            if not match:
                continue
            json_str = extract_balanced_json(script.string[match.start():])
            # json_str = json_str.replace('\\"', '"')
            json_str = json_str.encode('utf-8').decode('unicode_escape')
            json_str = json_str.replace('\\"', '')
            json_str = json_str + '}'
            try:
                data = json.loads(json_str)
                data = data['image']
            except json.JSONDecodeError as e:
                print("Error al parsear JSON:", e)
            break
        return data

    def get_chant_info(self, cantusid):
        """
        Get information about a chant using its Cantus ID.
        Parameters:
            cantusid (str): Cantus ID of the chant
        Returns:
            dict: JSON object with chant information
        """
        try:
            chantinfo = self.session.get(CANTU_API + cantusid).json()
        except Exception as e:
            return []
        return chantinfo

    def add_new_chant_record(self, cantusid):
        """
        Add a new chant record to the workspace.
        Parameters:
            cantusid (str): Cantus ID of the chant to be added
        Returns:
            str: ID of the newly created chant record
        """
        # chant info
        #chantinfo = self.session.get(CANTU_API + cantusid).json()
        chantinfo = self.get_chant_info(cantusid)
        # create chant
        payload = {
            "genre": chantinfo['info']['field_genre'] if chantinfo!=[] and chantinfo['info'] is not None else "",
            "feast": chantinfo['info']['field_feast'] if chantinfo!=[] and chantinfo['info'] is not None else "",
            "office": "",
            "mode": "",
            "cantusId": cantusid,
            "isNew": False
        }
        random_seq = ''.join(random.choices(string.ascii_letters, k=8))
        response = self.session.patch(f"{CHANT_URL}/{random_seq}", json=payload)
        chantRecordId = response.json()['chantRecord']['chantRecord']['id']
        return chantRecordId

    def add_new_chant_annotation(self, dataorig):
        """
        Add a new chant annotation to the workspace.
        Parameters:
            dataorig (dict): JSON object with annotation data
        Returns:
            dict: updated JSON object returned by the server
        """
        response = self.session.patch(INDEX_API + '/' + str(dataorig['id']), json=dataorig)
        return response
