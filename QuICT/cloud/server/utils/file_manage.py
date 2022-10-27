import os


# Default User Path
USER_ROOT_PATH = "/Cluster/User"


def create_user_folder(user_name):
    user_path = os.path.join(USER_ROOT_PATH, user_name)
    if not os.path.exists(os.path.join):
        os.makedirs(user_path)
    else:
        raise KeyError("user name already exists.")


def create_job_folder(user_name, job_name):
    job_path = os.path.join(USER_ROOT_PATH, user_name, job_name)
    if not os.path.exists(os.path.join):
        os.makedirs(job_path)
    else:
        raise KeyError("job name already exists.")


def delete_user_folder(user_name):
    user_path = os.path.join(USER_ROOT_PATH, user_name)
    if os.path.exists(os.path.join):
        os.remove(user_path)
    else:
        raise KeyError("user name already exists.")


def delete_job_folder(user_name, job_name):
    job_path = os.path.join(USER_ROOT_PATH, user_name, job_name)
    if os.path.exists(job_path):
        os.remove(job_path)
    else:
        raise KeyError("job name do not exists.")
