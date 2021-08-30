NAME=$(cat /etc/*release | grep ^NAME= | sed s/NAME=// | sed s'/GNU\/Linux//' | tr -d " \t\r " | sed s/\"//g)

case $NAME in

    "Ubuntu"|"Debian")
    echo "Ubuntu|Debian"
    apt install build-essential python3-setuptools python3-numpy python3-scipy libtbb2 libtbb-dev -y
    ;;

    "Fedora")
    echo "Fedora"
    dnf install  make gcc gcc-c++ kernel-devel linux-headers tbb tbb-devel python3-setuptools python3-numpy python3-scipy
    ;;

    *)
    echo "Unknown Platform:"
    echo $NAME

esac