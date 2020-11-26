.PHONY: tbb quict install clean default install_tbb install_quict

TBBLIB :=
TBBDST :=

ifeq ($(OS),Windows_NT)
	@echo "Do not support win32 platform currently"
	exit
else
	UNAME_S := $(shell uname -s)
	ifeq ($(UNAME_S),Linux)
		TBBLIB += libtbb.so libtbb.so.2 libtbbmalloc.so libtbbmalloc.so.2 libtbbmalloc_proxy.so libtbbmalloc_proxy.so.2
		TBBDST += /usr/lib
	endif
	ifeq ($(UNAME_S),Darwin)
		TBBLIB += libtbb.dylib
		TBBDST += /usr/local/lib
	endif
endif

PYTHON :=
ifeq ($(VIRTUAL_ENV), )
	PYTHON += python3
else
	PYTHON += $(VIRTUAL_ENV)/bin/python3
endif

tbb:
	${MAKE} -C tbb-2020.1

clean:
	${MAKE} -C tbb-2020.1 clean
	$(RM) tbb-2020.1/release_path.sh
	$(RM) QuICT/backends/quick_operator_cdll.so
	$(RM) QuICT/synthesis/initial_state_preparation/initial_state_preparation_cdll.so

quict: install_tbb
	cd ./QuICT/backends && \
	$(CXX) -std=c++11 dll.cpp -fPIC -shared -o quick_operator_cdll.so -I . -ltbb && \
	cd ../synthesis/initial_state_preparation && \
	$(CXX) -std=c++11 _initial_state_preparation.cpp -fPIC -shared -o initial_state_preparation_cdll.so -I ../../backends -ltbb
	$(PYTHON) setup.py build

# Do NOT add tbb here as a dependency, otherwise releasr_path.sh would be wrong
install_tbb:
	cd tbb-2020.1 && \
	. ./release_path.sh && \
	cp -t $(TBBDST) $(TBBLIB)

install_quict: quict
	$(PYTHON) setup.py install

# sudo
install:  install_quict install_tbb

# sudo
uninstall:
	cd $(TBBDST) && \
	$(RM) $(TBBLIB)
	$(PYTHON) -m pip uninstall QuICT

default: tbb quict
