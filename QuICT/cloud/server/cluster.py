import requests
from flask import Blueprint


cluster_blueprint = Blueprint(name="cluster", import_name=__name__)
URL_PREFIX = "/v1/cluster"


@cluster_blueprint.route(f"{URL_PREFIX}/status", methods=["GET"])
def status_cluster():
    """start a job. """
    return [4, 5, 6]
