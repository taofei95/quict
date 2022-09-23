from hashlib import sha256
import hmac
import base64


DEFAULT_ENCRYPTED_KEY = "QuICTOpenSource2022"


def encrypted_passwd(passwd: str):
    key = DEFAULT_ENCRYPTED_KEY.encode('utf-8')
    passwd = passwd.encode('utf-8')
    encrypted_passwd = hmac.new(key, passwd, digestmod=sha256).digest()
    sign = base64.b64encode(encrypted_passwd).decode()

    return sign


d = "test for eudislienfsef03>/dsf"
print(encrypted_passwd(d))
