from typing import Tuple
import numpy as np

from .data_structure import ResourceOp


def user_resource_op(user_info: dict, circuit_width: int, device: str, op: ResourceOp) -> Tuple[bool, dict]:
    signal = -1 if op == ResourceOp.Release else 1
    number_of_running_jobs = user_info['number_of_running_jobs'] + signal
    # exceed limit job number
    if number_of_running_jobs > user_info['maximum_parallel_level']:
        return False, user_info

    # Deal with running job's resource
    if device == "CPU":
        cur_user_running_qubits = _qubits_operator(
            user_info['running_qubits_in_cpu'], circuit_width, signal
        )
        qubits_limitation = user_info['max_qubits_in_cpu']
    else:
        cur_user_running_qubits = _qubits_operator(
            user_info['running_qubits_in_gpu'], circuit_width, signal
        )
        qubits_limitation = user_info['max_qubits_in_gpu']

    # exceed limit qubits number
    if cur_user_running_qubits > qubits_limitation:
        return False, user_info

    # Update user's infomation
    user_info['number_of_running_jobs'] = number_of_running_jobs
    if device == "CPU":
        user_info['running_qubits_in_cpu'] = cur_user_running_qubits
    else:
        user_info['running_qubits_in_gpu'] = cur_user_running_qubits

    return True, user_info


def _qubits_operator(qa, qb: int, signal):
    assert qb > 0
    if signal == -1:
        assert np.isclose(qa, qb) or qa > qb

    if qa == 0:
        return qb

    result = 2 ** qa + signal * 2 ** qb
    return np.log2(result) if not np.isclose(result, 0) else 0


def user_stop_jobs_op(user_info: dict, op: ResourceOp) -> Tuple[bool, dict]:
    signal = -1 if op == ResourceOp.Release else 1
    max_stopped_job_number = user_info['maximum_stop_level']
    current_stopped_job_number = user_info['number_of_stop_jobs']

    if (
        current_stopped_job_number + signal > max_stopped_job_number or
        user_info['number_of_running_jobs'] >= user_info['maximum_parallel_level']
    ):
        return False, user_info

    user_info['number_of_stop_jobs'] += signal
    user_info['number_of_running_jobs'] -= signal
    return True, user_info
