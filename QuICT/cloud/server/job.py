import requests
from flask import Blueprint


job_blueprint = Blueprint(name="jobs", import_name=__name__)
URL_PREFIX = "/quict/jobs"


@job_blueprint.route(f"{URL_PREFIX}/start", methods=["POST"])
def start_job(name: str):
    """start a job. """
    return [name, "start"]


@job_blueprint.route(f"{URL_PREFIX}/<name>:stop", methods=["POST"])
def stop_job(name: str):
    """ Stop a job. """
    return [name, "stop"]


@job_blueprint.route(f"{URL_PREFIX}/<name>:restart", methods=["POST"])
def restart_job(name: str):
    """ restart a job. """
    return [name, "restart"]


@job_blueprint.route(f"{URL_PREFIX}/<name>:delete", methods=["DELETE"])
def delete_job(name: str):
    """ delete a job. """
    return [name, "delete"]


@job_blueprint.route(f"{URL_PREFIX}/list", methods=["GET"])
def list_jobs():
    return [1, 2, 3]


@job_blueprint.route(f"{URL_PREFIX}/<name>:status", methods=["GET"])
def status_jobs(name: str):
    return [name, "status"]
