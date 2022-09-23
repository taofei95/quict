from flask import Blueprint


env_blueprint = Blueprint(name="env", import_name=__name__)
URL_PREFIX = "/quict/env"


@cluster_blueprint.route(f"{URL_PREFIX}/login", methods=["GET"])
def login():
    """start a job. """
    return [4, 5, 6]


@cluster_blueprint.route(f"{URL_PREFIX}/logout", methods=["POST"])
def logout():
    """start a job. """
    return [4, 5, 6]
