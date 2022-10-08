import json
import redis


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
    def get_user_information(self, user_name: str):
        return self._redis_connection.hgetall(f"user:{user_name}")            

    def get_user_password(self, user_name: str):
        return self._redis_connection.hget("user_password_mapping", user_name)

    def add_user(self, user_name: str, user_info: dict):
        encode_passwd = user_info['password']
        # Update user-passwd mapping
        self._redis_connection.hset("user_password_mapping", user_name, encode_passwd)

        # Add user info
        del user_info['password']
        self._redis_connection.hset(f"user:{user_name}", user_info)

    def delete_user(self, user_name: str):
        self._redis_connection.hdel("user_password_mapping", user_name)
        self._redis_connection.delete(f"user:{user_name}")

    ####################################################################
    ############             Cluster DB Function            ############
    ####################################################################
    def get_cluster_status(self):
        return self._redis_connection.get("cluster")

    def update_cluster_status(self, clauster_status: dict):
        clauster_name = clauster_status['name']
        self._redis_connection.hset(f"clauster:{clauster_name}", clauster_status)

    ####################################################################
    ############              Jobs DB Function              ############
    ####################################################################
    def get_pending_jobs_queue(self):
        return self._redis_connection.get("pending_jobs")

    def get_running_jobs_queue(self):
        return self._redis_connection.get("running_jobs")

    def get_finish_jobs_queue(self):
        return self._redis_connection.get("finish_jobs")

    def get_operator_jobs_queue(self):
        return self._redis_connection.get("operator_queue")

    def add_job(self, job_dict: dict):
        job_name = job_dict['job_name']
        self._redis_connection.rpush(f"pending_jobs", job_name)

        # Add job info into JobInfo Table
        self._redis_connection.hmset(f"Job_Info:{job_name}", job_dict)

    def add_operator(self, job_name: str, operator: str):
        self._redis_connection.rpush("operator_queue", json.dumps({job_name: operator}))

    def remove_job(self, job_name: str):
        if not self._redis_connection.exists(f"Job_Info:{job_name}"):
            raise KeyError("job not in database.")

        job_status = self._redis_connection.hget(f"Job_Info:{job_name}", "Status")
        job_name = self._redis_connection.hget(f"Job_Info:{job_name}", "job_name")
        self._redis_connection.lrem(f"{job_status}_jobs", 1, job_name)

        self._redis_connection.delete(f"Job_Info:{job_name}")

    ####################################################################
    ############              DB Utils Function             ############
    ####################################################################
    def _locked(self):
        pass

    def _launch_redis_server(self):
        pass

    def _unlocked(self):
        pass
