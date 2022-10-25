from flask import Blueprint

from script.requset_validation import request_validation
from script.redis_controller import RedisController
from script.sql_controller import SQLManger


env_blueprint = Blueprint(name="env", import_name=__name__)
URL_PREFIX = "/quict/env"


@env_blueprint.route(f"{URL_PREFIX}/login", methods=["POST"])
@request_validation(login=True)
def login(**kwargs):
    """ Login user
    kwargs = {
        'json_dict': dict = {
            'username': str,
            'password': str
            }
        }
    """
    json_dict = kwargs['json_dict']
    username = json_dict['username']
    password = json_dict['password']

    login_state = SQLManger().validation_password(username, password)
    if login_state:
        RedisController().update_user_dynamic_info(username)

    return login_state


@env_blueprint.route(f"{URL_PREFIX}/register", methods=["POST"])
@request_validation(login=True)
def register(**kwargs):
    """ Register new user.
    kwargs = {
        'json_dict': dict = {
            'username': str,
            'password': str,
            'email': str,
            'level': int,
            }
        }
    """
    json_dict = kwargs['json_dict']
    username = json_dict['username']

    SQLManger().add_user(json_dict)
    RedisController().update_user_dynamic_info(username, json_dict)


@env_blueprint.route(f"{URL_PREFIX}/unsubscribe", methods=["POST"])
@request_validation()
def unsubscribe(username):
    """ Delete an user. """
    SQLManger().delete_user(username)
    RedisController().delete_user(username)


@env_blueprint.route(f"{URL_PREFIX}/update_password", methods=["POST"])
@request_validation()
def update_password(username, new_password):
    """ Update user's password. """
    SQLManger().update_password(username, new_password)


@env_blueprint.route(f"{URL_PREFIX}/update_user_info", methods=["POST"])
@request_validation()
def update_user_info(username, new_email: str = None, new_level: int = None):
    """ Update user's information. """
    if new_email is not None:
        SQLManger().update_user_email(username, new_email)

    if new_level is not None:
        SQLManger().update_user_level(username, new_level)


@env_blueprint.route(f"{URL_PREFIX}/forget_password", methods=["POST"])
@request_validation()
def forget_password(username: str):
    """ Send email for user for activate new password. """
    user_info = SQLManger().get_user_info(username)
    user_email = user_info[1]

    # TODO: Send email to user
    pass
