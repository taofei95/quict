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

    def post(self, url: str, json_dict: dict = None, authorized: bool = False):
        aes_key = os.urandom(16)
        if authorized:
            authorized = json_dict['username']

        header = self._generate_header(aes_key, authorized)
        if json_dict is not None:
            json_dict = json.dumps(json_dict)

        response = requests.post(
            url=url,
            headers=header,
            data=self._encrypt.encryptedmsg(json_dict, aes_key) if json_dict is not None else None
        )

        # decrepted response
        if authorized:
            return self._decrepted_response(response, json.loads(json_dict))

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

    def _generate_header(self, aes_key: bytes, authorized: bool = False):
        if not authorized:
            user_info = get_config()
            username, password = user_info['username'], user_info['password'][:16]
        else:
            username = authorized
            password = self.__SALT

        payload = {
            'username': username,
            'aes_key': self._encrypt.encryptedmsg(aes_key, password).decode('ascii'),
            'exp': datetime.datetime.utcnow() + datetime.timedelta(minutes=15)
        }

        jwt_token = jwt.encode(
            payload=payload,
            key=self.__SALT,
            algorithm="HS256"
        )

        return {"Authorization": f"Bearer {jwt_token}"}

    def _decrepted_response(self, response: requests.Response, json_dict: dict = None):
        jwt_token = response.headers.get('Authorization')
        if jwt_token and jwt_token.startswith('Bearer '):
            payload = jwt.decode(jwt_token[7:], self.__SALT, ["HS256"])
        else:
            payload = None

        username = payload.get("username", None)
        encrypted_aes_key = payload.get('aes_key')

        if json_dict is None:
            user_info = get_config()
            local_user, password = user_info['username'], user_info['password'][:16]
        else:
            local_user, password = json_dict['username'], self.__SALT

        assert username == local_user

        content = response.content
        if not content:
            return None

        aes_key = self._encrypt.decryptedmsg(encrypted_aes_key, password, True)
        decrypted_data = json.loads(self._encrypt.decryptedmsg(content, aes_key))

        if isinstance(decrypted_data, dict) and 'error' in decrypted_data.keys():
            raise KeyError(f"{decrypted_data['error']}")

        return decrypted_data
