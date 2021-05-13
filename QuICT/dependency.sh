sudo apt install curl
sudo apt install python3.7
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
sudo apt install python3-distutils
sudo python3.7 get-pip.py
sudo pip install launchpadlib
pip3.7 install flask -i https://mirrors.aliyun.com/pypi/simple
pip3.7 install numpy -i https://mirrors.aliyun.com/pypi/simple

apt install build-essential python3-setuptools python3-numpy python3-scipy libtbb2 libtbb-dev
cd ./backend/algorithm/QuICT/backends
sudo g++  -o quick_operator_cdll.so dll.cpp -std=c++11  -fPIC -shared -ltbb
