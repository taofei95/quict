from argparse import ArgumentParser, RawTextHelpFormatter


QUICT_DESCRIBE = """
   ___            ___    ____   _____
  / _ \   _   _  |_ _|  / ___| |_   _|
 | | | | | | | |  | |  | |       | |
 | |_| | | |_| |  | |  | |___    | |
  \__\_\  \___/  |___|  \____|   |_|


Welcome to QuICT Command Line Interface.

Using 'quict --help' to get the documents about QuICT CLI.
Using 'quict --version' to get the version of QuICT.

"""


def cli_construct():
    parser = ArgumentParser(
        prog="quict",
        description=QUICT_DESCRIBE,
        formatter_class=RawTextHelpFormatter
    )
    parser.add_argument('--version', action='version', version='QuICT CLI 1.0', help="Show the current version.")
    subparsers = parser.add_subparsers()

    # Login
    login_sp = subparsers.add_parser(
        name="login",
        description="Login to QuICT Cloud System.",
        help="Login to QuICT Cloud System.",
    )
    login_sp.add_argument("-n", "--name", type=str, nargs=1)
    login_sp.add_argument("-p", "--password", type=str, nargs=1)

    # Logout
    _ = subparsers.add_parser(
        name="logout",
        description="Logout from the QuICT.",
        help="Logout from the QuICT."
    )

    # Circuit
    circuit_sp = subparsers.add_parser(
        name="circuit",
        description="Manage the circuit.",
        help="get circuit template."
    )
    circuit_cli_construct(circuit_sp)

    # Job
    job_sp = subparsers.add_parser(
        name="job",
        description="QuICT Jobs related.",
        help="QuICT job's related tools."
    )
    job_cli_construct(job_sp)

    # Cluster
    cluster_sp = subparsers.add_parser(
        name="cluster",
        description="cluster status related.",
        help="get cluster information."
    )
    cluster_cli_construct(cluster_sp)

    # Environment
    env_sp = subparsers.add_parser(
        name="env",
        description="Environment related tools",
        help="Manage environment"
    )
    env_cli_construct(env_sp)

    # Benchmark
    benchmark_sp = subparsers.add_parser(
        name="benchmark",
        description="benchmark related.",
        help="QuICT benchmark"
    )
    benchmark_cli_construct(benchmark_sp)

    return parser.parse_args()


def circuit_cli_construct(circuit_sp: ArgumentParser):
    subparser = circuit_sp.add_subparsers()
    # quict circuit get_random
    get_random = subparser.add_parser(
        name="get_random",
        description="Get random circuit.",
        help="get random circuit"
    )
    get_random.add_argument(
        "-i", "--instruction_set",
        choices=["USTC", "Google", "IBMQ", "IONQ"], nargs="?",
        help="Choice the instrcution set for random circuit."
    )
    get_random.add_argument(
        "-q", "--qubits",
        nargs="+", type=int, default=5,
        help="The number of qubits' number."
    )
    get_random.add_argument(
        "-s", "--size",
        nargs="+", type=int, default=25,
        help="The number of quantum gates."
    )
    get_random.add_argument(
        "-o", "--output",
        nargs="?", default=".",
        help="The output path, default to be current path."
    )

    # quict circuit get_algorithm
    get_algorithm = subparser.add_parser(
        name="get_algorithm",
        description="Get quantum algorithm's circuit.",
        help="get quantum algorithm's circuit"
    )
    get_algorithm.add_argument(
        "-a", "--alg",
        choices=["QFT", "Grover", "Shor", "VQE"], default="QFT",
        help="The quantum algorithm."
    )
    get_algorithm.add_argument(
        "-q", "--qubits",
        nargs="+", type=int, default=5,
        help="The number of qubits' number."
    )
    get_algorithm.add_argument(
        "-o", "--output",
        nargs="?", default=".",
        help="The output path, default to be current path."
    )

    # quict circuit add
    add = subparser.add_parser(
        name="add",
        description="Store quantum circuit.",
        help="store quantum circuit"
    )
    add.add_argument(
        "-n", "--name",
        nargs=1,
        help="The name of quantum circuit."
    )
    add.add_argument(
        "-f", "--file",
        nargs=1, default=".",
        help="The path of qasm file."
    )

    # quict circuit delete
    delete = subparser.add_parser(
        name="delete",
        description="delete quantum circuit.",
        help="delete quantum circuit"
    )
    delete.add_argument(
        "-n", "--name",
        nargs=1,
        help="The name of quantum circuit."
    )

    # quict circuit list
    _ = subparser.add_parser(
        name="list",
        description="list all the quantum circuit.",
        help="list all the quantum circuit."
    )


def job_cli_construct(job_sp: ArgumentParser):
    from QuICT.cloud.client.job import (
        start_job, stop_job, restart_job, delete_job, status_job, list_jobs
    )

    subparser = job_sp.add_subparsers()
    get_template = subparser.add_parser(
        name="get_template",
        description="Get job template.",
        help="Get job's template."
    )
    get_template.add_argument(
        "-t", "--type",
        choices=["simulation", "qcda", "all"], default="all",
        help="Get the template about simulation jobs or QCDA jobs."
    )
    get_template.add_argument(
        "-o", "--output",
        nargs="?",
        default=".",
        help="The output path, default to be current path."
    )

    # start job
    start = subparser.add_parser(
        name="start",
        description="Start the job.",
        help="start the job."
    )
    start.add_argument(
        "-f", "--file",
        nargs="+",
        help="The path of jobs file, could be a directory or some file path.",
    )
    start.set_defaults(func=start_job)

    # check job status
    status = subparser.add_parser(
        name="status",
        description="check the job's status.",
        help="check the job."
    )
    status.add_argument(
        "-n", "--name",
        nargs=1,
        help="The name of target job."
    )
    status.set_defaults(func=status_job)

    # stop
    stop = subparser.add_parser(
        name="stop",
        description="stop a job.",
        help="stop a job."
    )
    stop.add_argument(
        "-n", "--name",
        nargs=1,
        help="The name of target job."
    )
    stop.set_defaults(func=stop_job)

    # restart
    restart = subparser.add_parser(
        name="restart",
        description="restart the job.",
        help="restart the job."
    )
    restart.add_argument(
        "-n", "--name",
        nargs=1,
        help="The name of target job."
    )
    restart.set_defaults(func=restart_job)

    # delete
    delete = subparser.add_parser(
        name="delete",
        description="delete the job.",
        help="delete the job."
    )
    delete.add_argument(
        "-n", "--name",
        nargs=1,
        help="The name of target job."
    )
    delete.set_defaults(func=delete_job)

    # list
    list_job = subparser.add_parser(
        name="list",
        description="list the job.",
        help="list the job."
    )
    list_job.set_defaults(func=list_jobs)


def cluster_cli_construct(cluster_sp: ArgumentParser):
    from QuICT.cloud.client.cluster import status_cluster
    subparser = cluster_sp.add_subparsers()
    # quict cluster status
    status = subparser.add_parser(
        name="status",
        description="Show cluster running status.",
        help="Show cluster running status.",
    )
    status.set_defaults(func=status_cluster)


def env_cli_construct(env_sp: ArgumentParser):
    subparser = env_sp.add_subparsers()
    # quict env build
    build = subparser.add_parser(
        name="build",
        description="Build docker as running environment in distributed system.",
        help="build dockers",
    )
    build.add_argument(
        "-p", "--path",
        nargs=1,
        help="The path of Docker build file."
    )

    # quict env deploy
    deploy = subparser.add_parser(
        name="deploy",
        description="Deploy the docker into cluster",
        help="deploy the docker into cluster",
    )
    deploy.add_argument(
        "-n", "--name",
        nargs=1, type=str,
        help="The docker's name."
    )
    deploy.add_argument(
        "-d", "--device",
        choices=["CPU", "GPU"], default="CPU",
        help="The device of docker environment, default to be CPU."
    )

    # quict env list
    list = subparser.add_parser(
        name="list",
        description="List all docker environments in cluster create by user.",
        help="list all available docker environment."
    )
    list.add_argument(
        "-d", "--device", nargs="?",
        choices=["CPU", "GPU"],
        help="List docker environments of given devices."
    )

    # quict env delete
    delete = subparser.add_parser(
        name="delete",
        description="Delete docker environment.",
        help="Delete docker environment."
    )
    delete.add_argument(
        "-n", "--name",
        nargs=1, type=str,
        help="The docker's name which to delete."
    )


def benchmark_cli_construct(benchmark_sp: ArgumentParser):
    subparser = benchmark_sp.add_subparsers()
    # quict benchmark qcda
    qcda = subparser.add_parser(
        name="qcda",
        description="show the benchmarks about QCDA.",
        help="show the benchmarks about QCDA."
    )
    qcda.add_argument(
        "-i", "--instruction_set",
        choices=["USTC", "Google", "IBMQ", "IONQ"], nargs="?",
        help="Using given instruction set to do QCDA benchmark."
    )
    qcda.add_argument(
        "-m", "--mapping",
        action="store_true",
        help="show QCDA's mapping benchmark or not."
    )
    qcda.add_argument(
        "-o", "--optimization",
        action="store_true",
        help="show QCDA's optimization benchmark or not"
    )

    # quict benchmark simulation
    simulation = subparser.add_parser(
        name="simulation",
        description="show the benchmarks about Simulation.",
        help="show the benchmarks about Simulation."
    )
    simulation.add_argument(
        "-d", "--device", nargs="?",
        choices=["CPU", "GPU"], default="CPU",
        help="Select CPU/GPU in simulation."
    )
    simulation.add_argument(
        "-s", "--size",
        choices=["small", "medium", "large", "all"], default="all",
        help="Choice the size of simulation, default to all."
    )


if __name__ == "__main__":
    test = cli_construct()
    test.func(test)
