NAME=$(cat /etc/*release | grep ^NAME= | sed s/NAME=// | sed s'/GNU\/Linux//' | tr -d " \t\r " | sed s/\"//g)

case $NAME in

    "Ubuntu"|"Debian")
    echo "Ubuntu|Debian"
    apt install  build-essential libtbb2 libtbb-dev clang llvm \
    python3 python3-setuptools python3-numpy python3-scipy -y
    ;;

    *)
    echo "Unknown Platform:"
    echo $NAME

esac