//
// Created by Ci Lei on 2021-06-23.
//

#ifndef SIMULATION_BACKENDS_STATE_VECTOR_H
#define SIMULATION_BACKENDS_STATE_VECTOR_H

#include <vector>
#include <complex>
#include <cstdint>

namespace QuICT {

    template<typename precision_t=double>
    class StateVector {
    public:
        //* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        // Data Type conventions
        //* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

        using complex_t = std::complex<precision_t>;
        using vector_t = std::vector<complex_t>;


        //* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        // Constructors & Destructors
        //* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

        StateVector(uint64_t qubit_num) {
            state_vector_ = vector_t(qubit_num, static_cast<complex_t>(0));
        }

        StateVector(complex_t *init_state, uint64_t qubit_num) {
            state_vector_ = vector_t(qubit_num);
            std::copy(init_state, init_state + qubit_num * sizeof(complex_t),
                      state_vector_.begin());
        }

        StateVector(const vector_t &state_vector) {
            std::copy(state_vector.begin(), state_vector.end(), state_vector_.begin());
        }

        //* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        // Data accessor
        //* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

        inline auto &operator[](uint64_t idx) {
            return this->state_vector_[idx];
        }

    protected:
        vector_t state_vector_;
    };

}

#endif //SIMULATION_BACKENDS_STATE_VECTOR_H
