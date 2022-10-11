from flask import Blueprint

from script.requset_validation import request_validation
from script.redis_controller import RedisController
from utils.data_structure import JobOperatorType


job_blueprint = Blueprint(name="jobs", import_name=__name__)
URL_PREFIX = "/quict/jobs"


@job_blueprint.route(f"{URL_PREFIX}/start", methods=["POST"])
@request_validation
def start_job(**kwargs):
    """start a job. """
    job_dict = kwargs['json_dict']
    job_dict['username'] = kwargs['username']

    # start job by redis controller
    RedisController().add_pending_job(job_dict)


@job_blueprint.route(f"{URL_PREFIX}/<name>:stop", methods=["POST"])
@request_validation
def stop_job(name: str, username: str):
    """ Stop a job. """
    RedisController().add_operator(f"{username}:{name}", JobOperatorType.stop)


@job_blueprint.route(f"{URL_PREFIX}/<name>:restart", methods=["POST"])
@request_validation
def restart_job(name: str, username: str):
    """ restart a job. """
    RedisController().add_operator(f"{username}:{name}", JobOperatorType.restart)


@job_blueprint.route(f"{URL_PREFIX}/<name>:delete", methods=["DELETE"])
@request_validation
def delete_job(name: str, username: str):
    """ delete a job. """
    RedisController().add_operator(f"{username}:{name}", JobOperatorType.delete)


@job_blueprint.route(f"{URL_PREFIX}/list", methods=["GET"])
@request_validation
def list_jobs(username: str):
    RedisController().list_jobs(username)


@job_blueprint.route(f"{URL_PREFIX}/<name>:status", methods=["GET"])
@request_validation
def status_jobs(name: str, username: str):
    RedisController().get_job_info(f"{username}:{name}")
