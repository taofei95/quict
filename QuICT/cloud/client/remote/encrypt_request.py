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
        self.__SALT = "TestForQuICT"

    def get_current_userinfo(self):
        current_login_status = get_config()
        return current_login_status

    def get(self, url: str):
        aes_key = os.urandom(16)
        header = self._generate_header(aes_key)

        # Get response.
        response = requests.get(
            url=url,
            headers=header
        )

        # decrepted response
        return self._decrepted_response(response)

    def post(self, url: str, json_dict: dict = None):
        aes_key = os.urandom(16)
        header = self._generate_header(aes_key)
        if json_dict is not None:
            json_dict = json.dumps(json_dict)

        response = requests.post(
            url=url,
            headers=header,
            data=self._encrypt.encryptedmsg(json_dict, aes_key) if json_dict is not None else None
        )

        # decrepted response
        return self._decrepted_response(response)

    def delete(self, url: str, json_dict: dict = None):
        aes_key = os.urandom(16)
        header = self._generate_header(aes_key)
        if json_dict is not None:
            json_dict = json.dumps(json_dict)

        response = requests.delete(
            url=url,
            headers=header,
            data=self._encrypt.encryptedmsg(json_dict, aes_key) if json_dict is not None else None
        )

        # decrepted response
        return self._decrepted_response(response)

    def _generate_header(self, aes_key: bytes):
        userinfo = self.get_current_userinfo()
        payload = {
            'username': userinfo['username'],
            'aes_key': self._encrypt.encryptedmsg(aes_key, userinfo['password'][:16]).decode('ascii'),
            'exp': datetime.datetime.utcnow() + datetime.timedelta(minutes=15)
        }

        jwt_token = jwt.encode(
            payload=payload,
            key=self.__SALT,
            algorithm="HS256"
        )

        return {"Authorization": f"Bearer {jwt_token}"}

    def _decrepted_response(self, response: requests.Response):
        jwt_token = response.headers.get('Authorization')
        if jwt_token and jwt_token.startswith('Bearer '):
            payload = jwt.decode(jwt_token[7:], self.__SALT, ["HS256"])
        else:
            payload = None

        username = payload.get("username", None)
        encrypted_aes_key = payload.get('aes_key')

        userinfo = self.get_current_userinfo()
        assert username == userinfo['username']

        content = response.content

        if not content:
            return None

        aes_key = self._encrypt.decryptedmsg(encrypted_aes_key, userinfo['password'][:16], True)
        decrypted_data = json.loads(self._encrypt.decryptedmsg(content, aes_key))

        if isinstance(decrypted_data, dict) and 'error' in decrypted_data.keys():
            raise KeyError(f"{decrypted_data['error']}")

        return decrypted_data
