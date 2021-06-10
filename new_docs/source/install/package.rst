
Package
-----------------------------------------------------------------------

Jump into https://github.com/vision-oihyxc/QuICT/releases and download the
lastets release source code and unzip.

The source code now can only use on Linux and OS X.

Install Dependency
'''''''''''''''''''''''''''

You can try ``sudo ./dependency.sh`` to install dependencies
automatically(only Ubuntu and Fedora are supported currently).
If you prefer install python packages using ``pip``, just skip
setuptools, numpy and scipy in following commands.

To install dependencies on Ubuntu:

  sudo apt install build-essential libtbb2 libtbb-dev

  python3 python3-setuptools python3-numpy python3-scipy

> Or just install ``sudo apt install build-essential libtbb2 libtbb-dev``
> if you handle python parts in another way.

To install dependencies on Fedora:

  sudo dnf install make gcc gcc-c++ kernel-devel linux-headers tbb tbb-devel

  python3 python3-setuptools python3-numpy python3-scipy

> Or just ``sudo dnf install make gcc gcc-c++ kernel-devel linux-headers tbb tbb-devel``
> if you handle python parts in another way.

> Our helper scripts would use ``which``, ``uname``, ``grep``  and ``sed``. Install them if they are not equipped.

Build & Install QuICT
''''''''''''''''''''''''''''''''''''''''''''''''''''''

Following commands would build QuICT and install it system-wide.
If you are going to install it into some python virtual environment, do it without any `sudo`.

  ./build.sh
  
  sudo ./install.sh
