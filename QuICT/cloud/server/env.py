from flask import Blueprint

from script.requset_validation import request_validation
from script.redis_controller import RedisController


env_blueprint = Blueprint(name="env", import_name=__name__)
URL_PREFIX = "/quict/env"


@env_blueprint.route(f"{URL_PREFIX}/login", methods=["POST"])
@request_validation
def login(**kwargs):
    """start a job. """
    json_dict = kwargs['json_dict']
    username = json_dict['username']
    pwd = json_dict["password"]

    return RedisController().validation(username, pwd)


@env_blueprint.route(f"{URL_PREFIX}/status", methods=["GET"])
@request_validation
def status_cluster():
    """ Return cluster state. """
    return RedisController().get_cluster_status()
