from os import path, getcwd
from glob import glob

from setuptools import find_packages, setup


# Detect if I'm in `root` or `root/build`
PY_FILE_PATH = path.dirname(path.abspath(__file__))
PRJ_ROOT_RELATIVE = "." if getcwd() == PY_FILE_PATH else ".."


# static file
file_data = [
    ("QuICT/lib/qasm/libs", [f"{PRJ_ROOT_RELATIVE}/QuICT/lib/qasm/libs/qelib1.inc"]),
    ("QuICT/simulation/utils", [f"{PRJ_ROOT_RELATIVE}/QuICT/simulation/utils/simulator_parameters.json"]),
    ("QuICT/tools/cli/template", [f"{PRJ_ROOT_RELATIVE}/QuICT/tools/cli/template/quict_job.yml"]),
    ("QuICT/lib/circuitlib", [f"{PRJ_ROOT_RELATIVE}/QuICT/lib/circuitlib/circuit_library.db"]),
    ("QuICT/lib/circuitlib/template", glob(f"{PRJ_ROOT_RELATIVE}/QuICT/lib/circuitlib/template/*")),
    ("QuICT/lib/circuitlib/algorithm/adder", glob(f"{PRJ_ROOT_RELATIVE}/QuICT/lib/circuitlib/algorithm/adder/*")),
    ("QuICT/lib/circuitlib/algorithm/clifford", glob(f"{PRJ_ROOT_RELATIVE}/QuICT/lib/circuitlib/algorithm/clifford/*")),
    ("QuICT/lib/circuitlib/algorithm/cnf", glob(f"{PRJ_ROOT_RELATIVE}/QuICT/lib/circuitlib/algorithm/cnf/*")),
    ("QuICT/lib/circuitlib/algorithm/grover", glob(f"{PRJ_ROOT_RELATIVE}/QuICT/lib/circuitlib/algorithm/grover/*")),
    ("QuICT/lib/circuitlib/algorithm/maxcut", glob(f"{PRJ_ROOT_RELATIVE}/QuICT/lib/circuitlib/algorithm/maxcut/*")),
    ("QuICT/lib/circuitlib/algorithm/qft", glob(f"{PRJ_ROOT_RELATIVE}/QuICT/lib/circuitlib/algorithm/qft/*")),
    ("QuICT/lib/circuitlib/algorithm/qnn", glob(f"{PRJ_ROOT_RELATIVE}/QuICT/lib/circuitlib/algorithm/qnn/*")),
    ("QuICT/lib/circuitlib/algorithm/quantum_walk", glob(f"{PRJ_ROOT_RELATIVE}/QuICT/lib/circuitlib/algorithm/quantum_walk/*")),
    ("QuICT/lib/circuitlib/algorithm/vqe", glob(f"{PRJ_ROOT_RELATIVE}/QuICT/lib/circuitlib/algorithm/vqe/*")),
    ("QuICT/lib/circuitlib/random/aspen-4", glob(f"{PRJ_ROOT_RELATIVE}/QuICT/lib/circuitlib/random/aspen-4/*")),
    ("QuICT/lib/circuitlib/random/ourense", glob(f"{PRJ_ROOT_RELATIVE}/QuICT/lib/circuitlib/random/ourense/*")),
    ("QuICT/lib/circuitlib/random/rochester", glob(f"{PRJ_ROOT_RELATIVE}/QuICT/lib/circuitlib/random/rochester/*")),
    ("QuICT/lib/circuitlib/random/sycamore", glob(f"{PRJ_ROOT_RELATIVE}/QuICT/lib/circuitlib/random/sycamore/*")),
    ("QuICT/lib/circuitlib/random/tokyo", glob(f"{PRJ_ROOT_RELATIVE}/QuICT/lib/circuitlib/random/tokyo/*")),
]


setup(
    name="quict",
    version="0.9.3",
    description="Quantum Compute Platform of Institute of Computing Technology",
    author="Library for Quantum Computation and Theoretical Computer Science, ICT, CAS",
    author_email="quact@ict.ac.cn",
    license="Apache License 2.0",
    platforms=["Windows", "Linux", "macOS"],
    url="https://e.gitee.com/quictucas/repos/quictucas/quict",
    package_dir={"QuICT": f"{PRJ_ROOT_RELATIVE}/QuICT"},
    entry_points={
        "console_scripts": [
            "quict = QuICT.tools.cli.quict:main",
        ],
    },
    install_requires=[
        "matplotlib>=3.6.1",
        "networkx>=2.8.7",
        "numba>=0.56.3",
        "numpy>=1.23.4",
        "ply>=3.11",
        "psutil>=5.9.4",
        "pylatexenc>=2.10"
        "scipy==1.9.2",
        "pyyaml==6.0"
    ],
    packages=find_packages(where=PRJ_ROOT_RELATIVE),
    data_files=file_data,
    include_package_data=True,
    python_requires=">=3.8",
    zip_safe=False,
)
