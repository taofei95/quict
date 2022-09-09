# from QuICT.cloud.client.remote import QuICTManageNodeClient


def status_cluster():
    requests.get(
        url=f"{job_prefix}/status"
    )
