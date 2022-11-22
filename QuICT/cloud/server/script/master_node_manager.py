import time
import multiprocessing
import random

from redis_controller import RedisController
from QuICT.cloud.server.utils import (
    delete_job_folder, JobOperatorType, delete_user_folder,
    JobState, ResourceOp, user_resource_op, user_stop_jobs_op
)


class MasterNodeManager:
    def __init__(self):
        self.check_interval = 15
        self.redis_connection = RedisController()

    def start(self) -> None:
        """Start agents."""
        pending_job_agent = PendingJobProcessor(
            redis_connection=self.redis_connection,
            check_interval=self.check_interval,
        )
        pending_job_agent.start()

        running_job_agent = RunningJobProcessor(
            redis_connection=self.redis_connection,
            check_interval=self.check_interval,
        )
        running_job_agent.start()

        operator_queue_agent = OperatorQueueProcessor(
            redis_connection=self.redis_connection,
            check_interval=self.check_interval,
        )
        operator_queue_agent.start()


class PendingJobProcessor(multiprocessing.Process):
    def __init__(self, redis_connection: RedisController, check_interval: int = 10):
        super().__init__()
        self.redis_connection = redis_connection
        self.check_interval = check_interval

    def run(self):
        while True:
            self._check_pending_jobs()
            time.sleep(self.check_interval)

    def _check_pending_jobs(self):
        # Get current pending jobs
        pending_jobs = self.redis_connection.get_pending_jobs_queue()

        for job_name in pending_jobs:
            job_detail = self.redis_connection.get_job_info(job_name)

            # User's limitation
            user_info = self.redis_connection.get_user_dynamic_info(job_detail['username'])
            is_user_satisfied, updated_user_info = user_resource_op(
                user_info, int(job_detail['circuit_info']['width']), job_detail["device"], ResourceOp.Allocation
            )
            if not is_user_satisfied:
                continue

            # Start job
            self._start_job(job_detail)
            self.redis_connection.add_running_job_from_pending_jobs(job_name)
            # Update resource
            self.redis_connection.update_user_dynamic_info(job_detail['username'], updated_user_info)

    def _start_job(self, job_detail: dict):
        pass


class RunningJobProcessor(multiprocessing.Process):
    def __init__(self, redis_connection: RedisController, check_interval: int = 10):
        super().__init__()
        self.redis_connection = redis_connection
        self.check_interval = check_interval

    def run(self):
        while True:
            self._check_running_jobs()
            time.sleep(self.check_interval)

    def _check_running_jobs(self):
        # Get running jobs
        running_jobs = self.redis_connection.get_running_jobs_queue()

        for job_name in running_jobs:
            # TODO: check running job status
            job_detail = self.redis_connection.get_job_info(job_name)
            user_info = self.redis_connection.get_user_dynamic_info(job_detail['username'])
            if job_detail['state'] == JobState.Stop.value:
                continue

            job_state = self._check_job_state(job_name)
            is_finish = job_state in [JobState.Finish, JobState.Error]
            if not is_finish:
                continue

            self.redis_connection.add_finish_job_from_running_jobs(job_name, job_state)
            # Release User's resource
            _, updated_user_info = user_resource_op(
                user_info, int(job_detail['circuit_info']['width']), job_detail["device"], ResourceOp.Release
            )
            self.redis_connection.update_user_dynamic_info(job_detail['username'], updated_user_info)

    def _check_job_state(self, job_name: str):
        # FIXME: temporary fake code, only test pipeline work.
        temp = random.random()
        if temp >= 0.5:
            return JobState.Finish

        if temp <= 0.1:
            return JobState.Error

        return JobState.Running


class OperatorQueueProcessor(multiprocessing.Process):
    def __init__(self, redis_connection: RedisController, check_interval: int = 10):
        super().__init__()
        self.redis_connection = redis_connection
        self.check_interval = check_interval

    def run(self):
        while True:
            self._check_operator_queue()
            time.sleep(self.check_interval)

    def _check_operator_queue(self):
        # Get Operator Queue
        operator_queue = self.redis_connection.get_operator_queue()

        for op in operator_queue:
            job_name, operator = op[:-4], op[-3:]
            job_detail = self.redis_connection.get_job_info(job_name)
            if operator == JobOperatorType.delete.value:
                is_success = self._delete_related_job(job_name, job_detail)
            elif operator == JobOperatorType.stop.value:
                is_success = self._stop_related_job(job_name, job_detail)
            elif operator == JobOperatorType.restart.value:
                is_success = self._restart_related_job(job_name, job_detail)
            else:
                is_success = self._delete_user_info(job_name)

            if is_success:
                self.redis_connection.remove_operator(op)

    def _stop_related_job(self, job_name: str, job_detail: dict) -> bool:
        if job_detail['state'] != JobState.Running.value:
            print("no need to stop an non-running job.")
            return True

        running_jobs = self.redis_connection.get_running_jobs_queue()
        if job_name not in running_jobs:
            print("No running jobs need to stop.")
            return False

        # Update user's related info
        username = job_detail['username']
        user_info = self.redis_connection.get_user_dynamic_info(username)
        # Release User resource
        is_stopped, updated_user_info = user_stop_jobs_op(
            user_info, ResourceOp.Allocation
        )

        # TODO: Stop given job through k8s CLI
        self.redis_connection.change_job_state(job_name, JobState.Stop)
        self.redis_connection.update_user_dynamic_info(username, updated_user_info)

        # TODO: return warning about failure to stop target jobs
        # print("Failure to stop target job.")
        return is_stopped

    def _restart_related_job(self, job_name: str, job_detail: dict) -> bool:
        username = job_detail['username']
        if not job_detail['state']  == JobState.Stop.value:
            print("No stoped job need to be restart.")
            return True

        # Update user's related info
        user_info = self.redis_connection.get_user_dynamic_info(username)
        # Release User resource
        is_restart, updated_user_info = user_stop_jobs_op(
            user_info, ResourceOp.Release
        )

        # TODO: Restart given job through k8s CLI

        if is_restart:
            self.redis_connection.update_user_dynamic_info(username, updated_user_info)
            # Update job's state
            self.redis_connection.change_job_state(job_name, JobState.Running)

        return True

    def _delete_related_job(self, job_name: str, job_detail: dict):
        username = job_detail['username']
        job_state = job_detail['state']

        if job_state in [JobState.Running.value, JobState.Stop.value]:
            # TODO: kill running job

            # Get User Info
            user_info = self.redis_connection.get_user_dynamic_info(username)
            # Release User resource
            if job_state == JobState.Stop:
                _, updated_user_resource = user_stop_jobs_op(
                    user_info, ResourceOp.Release
                )
            else:
                _, updated_user_resource = user_resource_op(
                    user_info, int(job_detail['circuit_info']['width']), job_detail["device"], ResourceOp.Release
                )

            self.redis_connection.update_user_dynamic_info(username, updated_user_resource)

        self.redis_connection.remove_job(job_name, job_state)
        delete_job_folder(username, job_detail['job_name'])

        return True

    def _delete_user_info(self, user_name: str) -> bool:
        user_info = self.redis_connection.get_user_dynamic_info(user_name)
        if user_info["number_of_running_jobs"] == 0:
            self.redis_connection.delete_user_dynamic_info(user_name)
            delete_user_folder(user_name)

            return True

        return False


if __name__ == "__main__":
    manager = MasterNodeManager()
    manager.start()
