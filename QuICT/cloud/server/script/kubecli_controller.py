import subprocess

from kubenerte import client


def kube_pod_yaml(job_info, user_info):
    """
    apiVersion: v1 #必选, 版本号, 例如v1
    kind: Pod #必选, Pod 
    metadata: #必选，元数据 
    name: nginx #必选, Pod名称 
    labels: #自定义标签 
        app: nginx #自定义标签名字 
    spec: #必选, Pod中容器的详细定义 
        containers: #必选, Pod中容器列表, 一个pod里会有多个容器 
            - name: nginx #必选，容器名称 
            image: nginx #必选，容器的镜像名称 
            imagePullPolicy: IfNotPresent # [Always | Never | IfNotPresent] #获取镜像的策略 Alawys表示下载镜像 IfnotPresent表示优先使用本地镜像, 否则下载镜像, Nerver表示仅使用本地镜像 
            ports: #需要暴露的端口库号列表 
            - containerPort: 80 #容器需要监听的端口号 
        restartPolicy: Always # [Always | Never | OnFailure]#Pod的重启策略, Always表示一旦不管以何种方式终止运行, kubelet都将重启, OnFailure表示只有Pod以非0退出码退出才重启, Nerver表示不再重启该Pod 
    """
    pass


def kube_server_yaml():
    """apiVersion: v1
    kind: Service
    metadata:
    name: service-hello
    labels:
    name: service-hello
    spec:
    type: NodePort      #这里代表是NodePort类型的,另外还有ingress,LoadBalancer
    ports:
    - port: 80          #这里的端口和clusterIP(kubectl describe service service-hello中的IP的port)对应, 即在集群中所有机器上curl 10.98.166.242:80可访问发布的应用服务。
        targetPort: 8080  #端口一定要和container暴露出来的端口对应, nodejs暴露出来的端口是8081, 所以这里也应是8081
        protocol: TCP
        nodePort: 31111   # 所有的节点都会开放此端口30000--32767, 此端口供外部调用。
    selector:
        run: hello         #这里选择器一定要选择容器的标签, 之前写name:kube-node是错的。
    """
    pass


def start_job(file_path):
    """ kubectl apply -f path"""
    # Standardize start job deployment.
    job_details = kube_pod_yaml(start_job_deployment=start_job_deployment)

    # Save details
    K8sDetailsWriter.save_job_details(job_details=job_details)

    # Create and apply k8s config
    k8s_job = self._create_k8s_job(job_details=job_details)
    client.BatchV1Api().create_namespaced_job(body=k8s_job, namespace="default")


def job_operator(job_name, user_name, ops):
    """ 

    Args:
        job_name (_type_): _description_
        user_name (_type_): _description_
        ops (_type_): _description_
    """
    if ops == "delete":
        client.BatchV1Api().delete_namespaced_job(name=job_name, namespace=user_name)

    if ops == "stop":
        # TODO: using docker container stop
        pass

    if ops == "restart":
        # TODO: using docker container restart
        pass


def job_info(job_name, user_name):
    """ Return the job's running information

    Args:
        job_name (_type_): _description_
        user_name (_type_): _description_
    """
    # Get resources
    job_list = client.BatchV1Api().list_namespaced_job(namespace=user_name, pod_name=job_name).to_dict()["items"]

    return job_list


def list_jobs(username):
    job_list = client.BatchV1Api().list_namespaced_job(namespace="default").to_dict()["items"]

    return job_list
