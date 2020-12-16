NAME=$(cat /etc/*release | grep ^NAME= | sed s/NAME=// | sed s/\"//g)

case $NAME in

    "Ubuntu")
    echo "Ubuntu"
    apt install build-essential python3-setuptools python3-numpy python3-scipy libtbb2 libtbb-dev
    ;;

    "Fedora")
    echo "Fedora"
    dnf install  make gcc gcc-c++ kernel-devel linux-headers tbb tbb-devel python3-setuptools python3-numpy python3-scipy
    ;;

    *)
    echo "Unknown Platform"

esac