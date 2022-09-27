import os
import datetime
import json
import jwt
import requests

from .utils import get_config
from .encrypt_manager import EncryptManager


class EncryptedRequest:
    def __init__(self):
        self._encrypt = EncryptManager()

    def get_current_user(self):
        current_login_status = get_config()
        return current_login_status['username']

    def get(self, url: str):
        aes_key = os.urandom(16)
        header = self._generate_header(aes_key)

        # Get response.
        response = requests.get(
            url=url,
            headers=json.dumps(header)
        )

        # decrepted response
        decrpted_response = self._decrepted_response(response)
        return json.loads(decrpted_response)

    def post(self, url: str, json_dict: dict = None):
        aes_key = os.urandom(16)
        header = self._generate_header(aes_key)
        if json_dict is not None:
            json_dict = json.dumps(json_dict)

        response = requests.post(
            url=url,
            headers=json.dumps(header),
            data=self._encrypt.encryptedmsg(json_dict, aes_key)
        )

        # decrepted response
        decrpted_response = self._decrepted_response(response)
        return json.loads(decrpted_response)

    def delete(self, url: str, json_dict: dict = None):
        aes_key = os.urandom(16)
        header = self._generate_header(aes_key)
        if json_dict is not None:
            json_dict = json.dumps(json_dict)

        response = requests.delete(
            url=url,
            headers=json.dumps(header),
            data=self._encrypt.encryptedmsg(json_dict, aes_key)
        )

        # decrepted response
        decrpted_response = self._decrepted_response(response)
        return json.loads(decrpted_response)

    def _generate_header(self, aes_key: bytes):
        username = self.get_current_user()
        payload = {
            'username': username,
            'aes_key': self._encrypt.encryptedmsg(aes_key, username).decode('ascii'),
            'exp': (datetime.datetime.utcnow() + datetime.timedelta(minutes=15)).timestamp()
        }

        jwt_token = jwt.encode(
            payload=payload,
            key="TestForQuICT",
            algorithm="HS256"
        )

        return {"Authorization": f"Bearer {jwt_token}"}

    def _decrepted_response(self, response: requests.Response):
        encrypted_aes_key = response.headers.get('aes_key')
        content = response.content

        aes_key = self._encrypt.decryptedmsg(encrypted_aes_key, self.get_current_user(), True)
        return self._encrypt.decryptedmsg(content, aes_key)
