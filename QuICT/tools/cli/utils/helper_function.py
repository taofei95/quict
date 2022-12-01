import os

from QuICT.cloud.cli.utils.validation import JobValidation


def path_check(func):
    """ create the output path, if not exist. """
    def wraps(*args, **kwargs):
        output_path = args[-1] if args else kwargs["output_path"]
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        func(*args, **kwargs)

    return wraps


def yaml_decompostion(func):
    """ The decorator for regularize job's yaml file.
    """

    def wraps(*args, **kwargs):
        job_file_path = args[0] if args else kwargs["file"]
        # Validation job file
        job_info = JobValidation().job_validation(job_file_path)

        # step 3: run
        func(file=job_info)

    return wraps
