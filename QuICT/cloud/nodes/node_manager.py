import subprocess


class ResourceNodeManager:
    def __init__(self):
        pass

    def start_job(self, job_dict: dict):
        # Start Job within Container

        # Update Container Information
        pass

    def stop_job(self, job_name: str):
        # Get job information from Redis

        # Stop related container

        # Update job status
        pass

    def restart_job(self, job_name: str):
        # Get job information from Redis

        # Restart related container

        # Update job status
        pass

    def delete_job(self, job_name: str):
        # Get job information from Redis
        # Killed related container
        # Clear Related data
        # Update job status
        pass
