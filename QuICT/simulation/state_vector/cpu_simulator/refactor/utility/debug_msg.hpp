#ifndef QUICT_SIM_BACKEND_UTILITY_DEBUG_MSG_H
#define QUICT_SIM_BACKEND_UTILITY_DEBUG_MSG_H

#include <iostream>

#ifdef NDEBUG
#define DEBUG_MSG(x) (void(x))
#else
// Only worked when compiling with `Debug`.
#define DEBUG_MSG(x) std::cout << (x) << std::endl
#endif

#endif
