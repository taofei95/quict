import json
import redis

from ..utils.data_structure import JobOperatorType, JobState


class RedisController:
    def __init__(self):
        try:
            self._redis_pool = redis.ConnectionPool(host='127.0.0.1', port=6379, db=0)
            self._redis_connection = redis.Redis(connection_pool=self._redis_pool, decode_responses=True)
        except Exception as e:
            raise KeyError(f"Failure to connect to Redis, due to {e}. please check master node redis connection.")

    ####################################################################
    ############               User DB Function             ############
    ####################################################################
    def get_user_dynamic_info(self, user_name: str):
        return self._redis_connection.hgetall(f"User_Dynamic_Info:{user_name}")

    def update_user_dynamic_info(self, user_name: str, job_resource: dict):
        self._redis_connection.hmset(f"User_Dynamic_Info:{user_name}", job_resource)

    def delete_user(self, user_name: str):
        self._redis_connection.delete(f"User_Dynamic_Info:{user_name}")

    ####################################################################
    ############             Cluster DB Function            ############
    ####################################################################
    def get_cluster_status(self):
        # May using k8s CLI to replace
        return self._redis_connection.get("cluster")

    def update_cluster_status(self, clauster_status: dict):
        clauster_name = clauster_status['name']
        self._redis_connection.hset(f"clauster:{clauster_name}", clauster_status)

    ####################################################################
    ############              Jobs DB Function              ############
    ####################################################################
    def get_pending_jobs_queue(self):
        return self._redis_connection.lrange("pending_jobs", 0, -1)

    def get_running_jobs_queue(self):
        return self._redis_connection.lrange("running_jobs", 0, -1)

    def get_finish_jobs_queue(self):
        return self._redis_connection.lrange("finish_jobs", 0, -1)

    def get_operator_queue(self):
        return self._redis_connection.lrange("operator_queue", 0, -1)

    def get_job_info(self, job_name):
        return self._redis_connection.hgetall(f"Job_Info:{job_name}")

    def list_jobs(self, username: str):
        keys = self._redis_connection.keys(f"Job_Info:{username}*")
        job_infos_str = ""
        for k in keys:
            table = self._redis_connection.hgetall(k.decode('utf-8'))
            job_infos_str += str(table)

        return job_infos_str

    def add_pending_job(self, job_dict: dict):
        job_name = job_dict['job_name']
        username = job_dict['username']
        if self._redis_connection.exists(f"Job_Info:{username}:{job_name}"):
            raise KeyError("repeated name.")

        self._redis_connection.rpush("pending_jobs", f"{username}:{job_name}")

        # Add job info into JobInfo Table
        job_dict['state'] = JobState.Pending
        self._redis_connection.hmset(f"Job_Info:{username}:{job_name}", job_dict)

    def add_running_job_from_pending_jobs(self, job_name: str):
        self._redis_connection.rpush("running_jobs", job_name)
        self._redis_connection.lrem("pending_jobs", 0, value=job_name)
        self._redis_connection.hset(f"Job_Info:{job_name}", 'state', JobState.Running)

    def add_finish_job_from_running_jobs(self, job_name: str):
        self._redis_connection.rpush("finish_jobs", job_name)
        self._redis_connection.lrem("running_jobs", 0, value=job_name)
        self._redis_connection.hset(f"Job_Info:{job_name}", 'state', JobState.Finish)

    def add_operator(self, job_name: str, operator: JobOperatorType):
        self._redis_connection.rpush("operator_queue", f"{job_name}:{operator.value}")

    def remove_job(self, job_name: str):
        if not self._redis_connection.exists(f"Job_Info:{job_name}"):
            raise KeyError("job not in database.")

        job_state = self._redis_connection.hget(f"Job_Info:{job_name}", "state")
        if job_state == JobState.Error:
            job_state = JobState.Finish
        elif job_state == JobState.Stop:
            job_state = JobState.Running

        self._redis_connection.lrem(f"{job_state.value}_jobs", 0, job_name)
        self._redis_connection.delete(f"Job_Info:{job_name}")

    def remove_operator(self, op: str):
        self._redis_connection.lrem('operator_queue', 0, op)

    def change_job_state(self, job_name: str, state: JobState):
        self._redis_connection.hset(f"Job_Info:{job_name}", 'state', state)

    ####################################################################
    ############              DB Utils Function             ############
    ####################################################################
    def _locked(self):
        pass

    def _launch_redis_server(self):
        pass

    def _unlocked(self):
        pass
