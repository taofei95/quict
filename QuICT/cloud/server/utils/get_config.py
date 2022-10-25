import os
import yaml

from QuICT.cloud.server.script.sql_controller import SQLManger


DEFAULT_USER_YML_PATH = os.path.join(
    os.path.dirname(__file__),
    "../config/user_info.yml"
)


def get_default_user_config(username: str):
    with open(DEFAULT_USER_YML_PATH, encoding='utf-8') as yml:
        yaml_dict = yaml.load(yml)

    # Get user info from SQL database
    user_info = SQLManger().get_user_info(username)
    yaml_dict['username'] = username
    yaml_dict['maximum_parallel_level'] = user_info[3]
    yaml_dict['maximum_stop_level'] = user_info[4]
    yaml_dict['GPU_allowence'] = user_info[5]

    return yaml_dict
