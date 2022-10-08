from flask import Blueprint

from script.requset_validation import request_validation


env_blueprint = Blueprint(name="env", import_name=__name__)
URL_PREFIX = "/quict/env"


@env_blueprint.route(f"{URL_PREFIX}/login", methods=["POST"])
@request_validation
def login(**kwargs):
    """start a job. """
    json_dict = kwargs['json_dict']
    # TODO: check passwd in database

    return [4, 5, 6]


@env_blueprint.route(f"{URL_PREFIX}/status", methods=["GET"])
@request_validation
def status_cluster():
    """start a job. """
    return [4, 5, 6]
