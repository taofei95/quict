cd ./tbb-2020.1
sudo rm release_path.sh
make
. ./release_path.sh
a=`uname  -a`
b="Darwin"
if [[ $a =~ $b ]];then
    sudo cp libtbb.dylib /usr/local/lib
else
    sudo cp *.so /usr/lib
    sudo cp *.so.2 /usr/lib
fi
cd ../../../QuICT/backends
g++ -std=c++11 dll.cpp -fPIC -shared -o quick_operator_cdll.so -I . -ltbb
cd ../synthesis/initial_state_preparation
g++ -std=c++11 _initial_state_preparation.cpp -fPIC -shared -o initial_state_preparation_cdll.so -I ../../backends -ltbb
cd ../../../
sudo python3 setup.py install
