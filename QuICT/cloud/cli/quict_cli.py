from argparse import ArgumentParser, RawTextHelpFormatter
from pydoc import describe
from re import M


QUICT_DESCRIBE = \
"""
   ___            ___    ____   _____ 
  / _ \   _   _  |_ _|  / ___| |_   _|
 | | | | | | | |  | |  | |       | |  
 | |_| | | |_| |  | |  | |___    | |  
  \__\_\  \___/  |___|  \____|   |_|  \n


Welcome to QuICT Command Line Interface. \n \n

Using 'quict --help' to get the documents about QuICT CLI.
Using 'quict --version' to get the version of QuICT.

"""

def cli_construct():
    parser = ArgumentParser(
        prog="quict",
        description=QUICT_DESCRIBE,
        formatter_class=RawTextHelpFormatter
    )
    parser.add_argument('--version',action='version',version='QuICT CLI 1.0', help="Show the current version.")

    subparsers = parser.add_subparsers()

    # Login
    login_sp = subparsers.add_parser(
        name="login",
        description="Login to QuICT Cloud System."
    )
    login_sp.add_argument("-n", "--name", type=str, nargs=1)
    login_sp.add_argument("-p", "--password", type=str, nargs=1)
    
    # Logout
    logout_sp = subparsers.add_parser(
        name="logout",
        description="Logout from the QuICT."
    )

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
        description="cluster status related."
    )
    
    # Environment
    env_sp = subparsers.add_parser(
        name="env",
        description="Environment related tools"
    )
    
    # Benchmark
    benchmark_sp = subparsers.add_parser(
        name="benchmark",
        description="benchmark related."
    )

    return parser.parse_args()


def job_cli_construct(job_sp: ArgumentParser):
    subparser = job_sp.add_subparsers()
    get_template = subparser.add_parser(
        name="get_template",
        description="Get job template.",
        help="Get job's template."
    )
    get_template.add_argument(
        "-t", "--type",
        choices=["simulation", "qcda"],
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
        help="The path of jobs file, could be a directory or some file path."
    )
    
    # check job status
    check = subparser.add_parser(
        name="check",
        description="check the job's status.",
        help="check the job."
    )
    check.add_argument(
        "-n", "--name",
        nargs=1,
        help="The name of target job."
    )

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
    
    # list
    list = subparser.add_parser(
        name="list",
        description="list the job.",
        help="list the job."
    )
    

if __name__ == "__main__":
    test = cli_construct()
