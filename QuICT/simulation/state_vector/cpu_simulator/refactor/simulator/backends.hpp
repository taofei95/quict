#ifndef QUICT_SIM_BACKEND_BACKENDS_H
#define QUICT_SIM_BACKEND_BACKENDS_H

namespace sim {
enum BackendTag { AUTO = 0, NAIVE, SSE, AVX, AVX512 };
}

#endif
