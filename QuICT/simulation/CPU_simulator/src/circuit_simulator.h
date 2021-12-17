//
// Created by LinHe on 2021-09-24.
//

#ifndef SIM_BACK_CIRCUIT_SIMULATOR_H
#define SIM_BACK_CIRCUIT_SIMULATOR_H

#include <string>
#include <vector>
#include <map>
#include <complex>

#include "gate.h"
#include "utility.h"
#include "matricks_simulator.h"
#include "tiny_simulator.h"

namespace QuICT {
    template<typename Precision>
    class CircuitSimulator {
    public:
        explicit CircuitSimulator(uint64_t qubit_num) : qubit_num_(qubit_num) {
            // Build name
            using namespace std;
            static_assert(is_same_v<Precision, double> || is_same_v<Precision, float>,
                          "MaTricksSimulator only supports double/float precision.");
            name_ = "CircuitSimulator";
            if constexpr(std::is_same_v<Precision, double>) {
                name_ += "[double, ";
            } else if (std::is_same_v<Precision, float>) {
                name_ += " [float, ";
            }
            name_ += std::to_string(qubit_num) + " bit(s)]";
        }

        const std::string &name() {
            return name_;
        }

        inline std::complex<Precision> *
        run(const std::vector<GateDescription<Precision>> &gate_desc_vec, bool keep_state);

    protected:
        std::string name_;
        uint64_t qubit_num_;
        Precision *real_ = nullptr, *imag_ = nullptr;
        inline static TinySimulator<Precision> tiny_sim_;
        inline static MaTricksSimulator<Precision> matricks_sim_;
    };

    template<typename Precision>
    std::complex<Precision> *CircuitSimulator<Precision>::run(
            const std::vector<GateDescription<Precision>> &gate_desc_vec,
            bool keep_state
    ) {
        if (!keep_state || (real_ == nullptr && imag_ == nullptr)) {
            // Initialize state vector
            uint64_t len = 1LL << qubit_num_;
            real_ = new Precision[len];
            imag_ = new Precision[len];
            std::fill(real_, real_ + len, 0);
            std::fill(imag_, imag_ + len, 0);
            real_[0] = 1.0;
        }
        if (qubit_num_ > 4) { // Can use matricks simulator
            for (const auto &gate_desc: gate_desc_vec) {
                matricks_sim_.apply_gate(qubit_num_, gate_desc, real_, imag_);
            }
        } else { // Only can use plain simulator
            for (const auto &gate_desc: gate_desc_vec) {
                tiny_sim_.apply_gate(qubit_num_, gate_desc, real_, imag_);
            }
        }
        auto res = new std::complex<Precision>[1 << qubit_num_];
        combine_complex(qubit_num_, real_, imag_, res);
        return res;
    }
};

#endif //SIM_BACK_CIRCUIT_SIMULATOR_H
