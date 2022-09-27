import requests
from flask import Blueprint

from .script.requset_validation import request_validation


job_blueprint = Blueprint(name="jobs", import_name=__name__)
URL_PREFIX = "/quict/jobs"


@job_blueprint.route(f"{URL_PREFIX}/start", methods=["POST"])
@request_validation
def start_job(**kwargs):
    """start a job. """
    job_dict = kwargs['json_dict']

    # start job by redis controller
    return ["start"]


@job_blueprint.route(f"{URL_PREFIX}/<name>:stop", methods=["POST"])
@request_validation
def stop_job(name: str):
    """ Stop a job. """
    return [name, "stop"]


@job_blueprint.route(f"{URL_PREFIX}/<name>:restart", methods=["POST"])
@request_validation
def restart_job(name: str):
    """ restart a job. """
    return [name, "restart"]


@job_blueprint.route(f"{URL_PREFIX}/<name>:delete", methods=["DELETE"])
@request_validation
def delete_job(name: str):
    """ delete a job. """
    return [name, "delete"]


@job_blueprint.route(f"{URL_PREFIX}/list", methods=["GET"])
@request_validation
def list_jobs():
    return [1, 2, 3]


@job_blueprint.route(f"{URL_PREFIX}/<name>:status", methods=["GET"])
@request_validation
def status_jobs(name: str):
    return [name, "status"]
