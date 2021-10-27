//
// Created by Ci Lei on 2021-10-22.
//

#ifndef SIM_BACK_Q_STATE_SET_H
#define SIM_BACK_Q_STATE_SET_H

#include <vector>
#include <map>
#include <complex>

#include "q_state.h"

namespace QuICT {
    template<typename Precision>
    class QStateSet {
    public:
        QStateSet(uint64_t qubit_num)
                : fa_(qubit_num), rank_(qubit_num, 1), qubit_num_(qubit_num) {
            for (uint64_t i = 0; i < qubit_num; ++i) {
                fa_[i] = i;
                states_.push_back(new QState<Precision>(i));
            }
        }

        ~QStateSet() {
            for (auto it: states_) {
                if (it) {
                    delete it;
                }
            }
        }

        inline QState <Precision> *get_q_state(uint64_t qubit_id);

        inline QState <Precision> *merge_q_state(uint64_t qubit_id_a, uint64_t qubit_id_b);

        inline QState <Precision> *merge_all();

    protected:
        std::vector<uint64_t> fa_;
        std::vector<uint64_t> rank_;
        std::vector<QState < Precision> *>
        states_;
        uint64_t qubit_num_;

        inline uint64_t find(uint64_t id);

        QState <Precision> *merge_range(uint64_t l, uint64_t r);
    };

    template<typename Precision>
    QState <Precision> *QStateSet<Precision>::get_q_state(uint64_t qubit_id) {
        uint64_t id = find(qubit_id);
        return states_[id];
    }

    template<typename Precision>
    uint64_t QStateSet<Precision>::find(uint64_t id) {
        while (fa_[id] != id) {
            auto fa_fa = fa_[fa_[id]];
            fa_[id] = fa_fa;
            id = fa_fa;
        }
        return id;
    }

    template<typename Precision>
    QState <Precision> *QStateSet<Precision>::merge_q_state(uint64_t qubit_id_a, uint64_t qubit_id_b) {
        uint64_t rt_a = find(qubit_id_a);
        uint64_t rt_b = find(qubit_id_b);
        if (rt_a == rt_b) {
            return states_[rt_a];
        }

        if (rank_[rt_a] < rank_[rt_b]) {
            std::swap(rt_a, rt_b);
        }

        if (rank_[rt_a] == rank_[rt_b]) {
            rank_[rt_a]++;
        }
        fa_[rt_b] = rt_a;
        auto s = QState<Precision>::merge_q_state(states_[rt_a], states_[rt_b], rt_a);
        delete_and_set_null(states_[rt_a]);
        delete_and_set_null(states_[rt_b]);
        states_[rt_a] = s;
        return states_[rt_a];
    }

    template<typename Precision>
    QState <Precision> *QStateSet<Precision>::merge_all() {
        return merge_range(0, qubit_num_);
    }

    template<typename Precision>
    QState <Precision> *QStateSet<Precision>::merge_range(uint64_t l, uint64_t r) {
        switch (r - l) {
            case 1: {
                return states_[find(l)];
            }
            case 2: {
                return merge_q_state(l, l + 1);
            }
            case 3: {
                merge_q_state(l, l + 1);
                return merge_q_state(l, l + 2);
            }
            default: {
                auto m = (l + r) >> 1;
                merge_range(l, m);
                merge_range(m, r);
                return merge_q_state(l, m);
            }
        }
    }
}

#endif //SIM_BACK_Q_STATE_SET_H
