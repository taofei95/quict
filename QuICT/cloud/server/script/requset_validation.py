import json
import jwt
import os
import functools
from flask import request, Response

from ...client.remote.encrypt_manager import EncryptManager


def request_validation(func):
    """Check JWT validity and do data decryption before getting into the actual logistic.
    Args:
        func:
    Returns:
        None.
    """

    @functools.wraps(func)
    def with_valid(*args, **kwargs):
        # Get jwt_token and its payload
        encrypt = EncryptManager()
        jwt_token = request.headers.get('Authorization')
        payload = jwt.decode(jwt_token, "TestForQuICT", ["HS256"])
        username = payload.get("username", None)
        aes_key = payload.get('aes_key')

        # TODO: check user available

        data = request.data
        if data != b'':
            decrypted_aeskey = encrypt.decryptedmsg(aes_key, username, True)
            decrypted_data = encrypt.decryptedmsg(data, decrypted_aeskey)
            kwargs['json_dict'] = json.loads(decrypted_data)

        return_data = func(*args, **kwargs)

        # create header for response
        return create_response(username, return_data)

    return with_valid


def create_response(username: str, json_dict: dict):
    encrypt = EncryptManager()
    aes_key = os.urandom(16)

    payload = {
        'username': username,
        'aes_key': encrypt.encryptedmsg(aes_key, username).decode('ascii')
    }

    jwt_token = jwt.encode(
        payload=payload,
        key="TestForQuICT",
        algorithm="HS256"
    )

    if json_dict is not None:
        json_dict = encrypt.encryptedmsg(json.dumps(json_dict), aes_key)

    response = Response(response=json_dict)
    response.headers = {"Authorization": f"Bearer {jwt_token}"}

    return response
