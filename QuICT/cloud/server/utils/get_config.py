import os
import yaml


DEFAULT_USER_YML_PATH = os.path.join(
    os.path.dirname(__file__),
    "../config/user_info.yml"
)


def get_default_user_config():
    with open(os.path.abspath(DEFAULT_USER_YML_PATH), encoding='utf-8') as yml:
        yaml_dict = yaml.load(yml)

    return yaml_dict
