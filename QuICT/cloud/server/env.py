from flask import Blueprint

from script.requset_validation import request_validation
from script.redis_controller import RedisController
from script.sql_controller import SQLManger


env_blueprint = Blueprint(name="env", import_name=__name__)
URL_PREFIX = "/quict/env"


@env_blueprint.route(f"{URL_PREFIX}/login", methods=["POST"])
@request_validation
def login(**kwargs):
    """start a job. """
    json_dict = kwargs['json_dict']
    username = json_dict['username']
    pwd = json_dict["password"]

    return SQLManger().validation_password(username, pwd)


@env_blueprint.route(f"{URL_PREFIX}/register", methods=["POST"])
@request_validation
def register(**kwargs):
    json_dict = kwargs['json_dict']
    username = json_dict['username']
    pwd = json_dict["password"]

    SQLManger().add_user(username, pwd, json_dict['user_info'])
    RedisController().update_user_dynamic_info(username, json_dict['user_info'])


@env_blueprint.route(f"{URL_PREFIX}/register", methods=["POST"])
@request_validation
def unsubscribe(**kwargs):
    username = kwargs['username']

    SQLManger().delete_user(username)
    RedisController().delete_user(username)


@env_blueprint.route(f"{URL_PREFIX}/status", methods=["GET"])
@request_validation
def status_cluster():
    """ Return cluster state. """
    return RedisController().get_cluster_status()
