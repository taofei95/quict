import os
import shutil


def get_template(type: str, output_path: str):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    tempfile_path = os.path.join(
        os.path.dirname(__file__),
        "../template"
    )
    type_list = ["simulation", "qcda"] if type == "all" else [type]
    for t in type_list:
        file_name = f"job_{t}.yml"
        temp_file_path = os.path.join(
            tempfile_path,
            file_name
        )
        shutil.copy(temp_file_path, output_path)
