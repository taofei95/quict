import os
import base64
import json
import requests
from Crypto.Cipher import AES


class EncryptedRequest:
    def __init__(self, username: str):
        self._user = username
        self._user_password = None      # Get userpassword by username

    def get(self, url: str):
        aes_key = os.urandom(16)
        header = self._generate_header(aes_key)

        # Get response.
        response = requests.get(
            url=url,
            headers=json.dumps(header),
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
            data=json_dict
        )

        # decrepted response
        decrpted_response = self._decrepted_response(response)
        return json.loads(decrpted_response)

    def delete(self, url: str , json_dict: dict = None):
        aes_key = os.urandom(16)
        header = self._generate_header(aes_key)
        if json_dict is not None:
            json_dict = json.dumps(json_dict)

        response = requests.delete(
            url=url,
            headers=json.dumps(header),
            data=json_dict
        )

        # decrepted response
        decrpted_response = self._decrepted_response(response)
        return json.loads(decrpted_response)

    def _padding(self, value: str, block_size: int = 16) -> bytes:
        if len(value) % block_size != 0:
            value += '\0' * (block_size - len(value) % block_size)

        return str.encode(value)

    def _generate_header(self, aes_key: bytes):
        return {
            'user_name': self._user,
            'aes_key': self.encryptedmsg(aes_key, self._user_password)
        }

    def _decrepted_response(self, response: requests.Response):
        encrypted_aes_key = response.headers.get('aes_key')
        content = response.content

        aes_key = self.decryptedmsg(encrypted_aes_key, self._user_password)
        return self.decryptedmsg(content, aes_key)

    def encryptedmsg(self, msg: str, key: str) -> bytes:
        aes = AES.new(self._padding(key), mode=AES.MODE_ECB)
        aes_message = aes.encrypt(self._padding(msg))
        encrypted_text = str(base64.encodebytes(aes_message), encoding='utf-8')

        return encrypted_text

    def decryptedmsg(self, msg: bytes, key: str):
        aes = AES.new(self._padding(key), mode=AES.MODE_ECB)
        base64_decryptedmsg = base64.decodebytes(msg.encode(encoding='utf-8'))
        encrypted_msg = str(aes.decrypt(base64_decryptedmsg), encoding='utf-8').replace('\0', '')

        return encrypted_msg
