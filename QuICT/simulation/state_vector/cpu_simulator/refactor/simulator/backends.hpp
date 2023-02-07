#ifndef QUICT_SIM_BACKEND_BACKENDS_H
#define QUICT_SIM_BACKEND_BACKENDS_H

namespace sim {
enum BackendTag { AUTO, NAIVE, SSE2, AVX, AVX512 };
}

#endif
