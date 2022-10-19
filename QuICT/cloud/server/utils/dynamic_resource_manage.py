from typing import Tuple
from .data_structure import ResourceOp


def user_resource_op(user_info: dict, resource_info: dict, op: ResourceOp) -> Tuple[bool, dict]:
    signal = -1 if op == ResourceOp.Release else 1
    target_resource = resource_info['number_of_qubits']
    number_of_running_jobs = user_info['number_of_running_jobs'] + signal
    # exceed limit job number
    if number_of_running_jobs > user_info['max_running_jobs']:
        return False, user_info

    # Deal with running job's resource
    if resource_info['type'] == "CPU":
        cur_user_running_qubits = _qubits_operator(
            user_info['running_qubits_in_cpu'], target_resource, signal
        )
        qubits_limitation = user_info['max_qubits_in_cpu']
    else:
        cur_user_running_qubits = _qubits_operator(
            user_info['running_qubits_in_gpu'], target_resource, signal
        )
        qubits_limitation = user_info['max_qubits_in_gpu']

    # exceed limit qubits number
    if cur_user_running_qubits > qubits_limitation:
        return False, user_info

    # Update user's infomation
    user_info['number_of_running_jobs'] = number_of_running_jobs
    if resource_info['type'] == "CPU":
        user_info['running_qubits_in_cpu'] = cur_user_running_qubits
    else:
        user_info['running_qubits_in_gpu'] = cur_user_running_qubits

    return True, user_info


def _qubits_operator(qa, qb, signal):
    q_max, q_min = max(qa, qb), min(qa, qb)
    return q_max + signal * (2 ** -(q_max - q_min))


def user_stop_jobs_op(user_info: dict, op: ResourceOp) -> Tuple[bool, dict]:
    signal = -1 if op == ResourceOp.Release else 1
    max_stopped_job_number = user_info['maximum_stop_level']
    current_stopped_job_number = user_info['number_of_stop_jobs']

    if (
        current_stopped_job_number + signal > max_stopped_job_number or
        user_info['number_of_running_jobs'] >= user_info['max_running_jobs']
    ):
        return False, user_info

    user_info['number_of_stop_jobs'] += signal
    user_info['number_of_running_jobs'] -= signal
    return True, user_info
