import os
import json


LOCAL_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__),
    "../../config/runtime_config.json"
)


def get_config():
    with open(LOCAL_CONFIG_PATH, "r", encoding="utf-8") as config_file:
        local_status = json.load(config_file)

    return local_status


def update_config(updated_status):
    with open(LOCAL_CONFIG_PATH, "w") as config_file:
        config_file.write(json.dumps(updated_status))
