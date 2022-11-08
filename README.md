# QuICT

## In case that you have a favor over Docker

Though releasing images on Docker Hub is in our agenda, currently
docker users might need to build docker image from sources.
With the docker file we provide in the repository, one can easily
build a docker image with only a little performance loss. 

```
docker build -t quict .
```

## Build & Install QuICT

**Make sure to upgrade pip wheel and setuptools before building!**

> For Windows users, please see the special notes for Building on Windows.

Following commands would build QuICT and install it system-wide.
You might need "sudo" privileges to install QuICT into system python package path.

> Due to some missing features in low version GCC (<=11), current QuICT is recommended to be built with clang.
> In future versions, GCC will be supported.

```bash
export CC=clang && export CXX=clang++ && ./build.sh && ./install.sh
```

If you are encountered with permission issues during installing, try

```bash
export CC=clang && export CXX=clang++ && ./build.sh && ./install.sh --user
```

### Building on Windows

It is recommended to use `clang-cl.exe`, which is the clang compiler with MSVC CLI. Other compilers may work but not tested. Open "Developer PowerShell for VS", changing the working directory to QuICT repository root. Then build with following commands:

```powershell
$ENV:CC="clang-cl.exe"
$ENV:CXX="clang-cpp.exe"

```
