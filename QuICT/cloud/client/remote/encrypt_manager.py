import base64
import hmac

from hashlib import sha256
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad, pad


class EncryptManager:
    def __init__(self):
        self._default_encrypted_key = "QuICTOpenSource2022"

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

    def encrypted_passwd(self, passwd: str) -> str:
        key = self._default_encrypted_key.encode('utf-8')
        if isinstance(passwd, str):
            passwd = passwd.encode('utf-8')

        encrypted_passwd = hmac.new(key, passwd, digestmod=sha256).digest()
        sign = base64.b64encode(encrypted_passwd).decode()

        return sign
