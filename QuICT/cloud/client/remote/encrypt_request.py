import os
import base64
import json
import requests
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad, pad

from .utils import get_config


class EncryptedRequest:
    def __init__(self):
        pass

    def get_current_user(self):
        current_login_status = get_config()
        # return current_login_status['username']
        return "testestetste"

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
            data=self.encryptedmsg(json_dict, aes_key)
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
            data=self.encryptedmsg(json_dict, aes_key)
        )

        # decrepted response
        decrpted_response = self._decrepted_response(response)
        return json.loads(decrpted_response)

    def _generate_header(self, aes_key: bytes):
        username = self.get_current_user()
        return {
            'username': username,
            'aes_key': self.encryptedmsg(aes_key, username)
        }

    def _decrepted_response(self, response: requests.Response):
        encrypted_aes_key = response.headers.get('aes_key')
        content = response.content

        aes_key = self.decryptedmsg(encrypted_aes_key, self.get_current_user(), True)
        return self.decryptedmsg(content, aes_key)

    def encryptedmsg(self, msg: str, key: bytes) -> bytes:
        if isinstance(msg, str):
            msg = msg.encode()

        if isinstance(key, str):
            key = key.encode()
    
        aes = AES.new(pad(key, 16), mode=AES.MODE_ECB)
        aes_message = aes.encrypt(pad(msg, 16))
        encrypted_text = base64.encodebytes(aes_message)

        return encrypted_text

    def decryptedmsg(self, msg: bytes, key: str, output_byte: bool = False):
        if isinstance(msg, str):
            msg = msg.encode()

        if isinstance(key, str):
            key = key.encode()

        aes = AES.new(pad(key, 16), mode=AES.MODE_ECB)
        base64_decryptedmsg = base64.decodebytes(msg)
        encrypted_msg = unpad(aes.decrypt(base64_decryptedmsg), 16)
        if not output_byte:
            encrypted_msg = str(unpad(aes.decrypt(base64_decryptedmsg), 16), encoding='utf-8')

        return encrypted_msg
