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
        pass

    def add_user(self, user_name: str, user_info: dict):
        pass

    def delete_user(self, user_name: str):
        pass

    ####################################################################
    ############             Cluster DB Function            ############
    ####################################################################
    def get_cluster_status(self):
        pass
    
    def update_cluster_status(self, clauster_status: dict):
        pass

    ####################################################################
    ############              Jobs DB Function              ############
    ####################################################################
    def get_pending_jobs_queue(self):
        pass

    def add_pending_job(self, job_dict: dict):
        pass

    def get_running_jobs_queue(self):
        pass

    def add_running_job(self, job_dict: dict):
        pass

    def get_finish_jobs_queue(self):
        pass

    def add_finish_job(self, job_dict: dict):
        pass

    def get_killed_jobs_queue(self):
        pass

    def add_killed_job(self, job_dict: dict):
        pass

    def remove_job(self, job_name: str):
        pass

    ####################################################################
    ############              DB Utils Function             ############
    ####################################################################
    def _locked(self):
        pass

    def _launch_redis_server(self):
        pass

    def _unlocked(self):
        pass
