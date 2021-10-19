//
// Created by Ci Lei on 2021-10-14.
//

#ifndef SIM_BACK_Q_STATE_H
#define SIM_BACK_Q_STATE_H

#include <vector>
#include <map>
#include <complex>
#include <cstdint>

#include "utility.h"

namespace QuICT {
    template<typename Precision>
    class QState {
    public:
        uint64_t id_;
        uint64_t qubit_num_;
        std::map<uint64_t, uint64_t> qubit_mapping_;
        Precision *real_;
        Precision *imag_;

        QState(uint64_t id, uint64_t qubit_num) : id_(id), qubit_num_(qubit_num) {
            real_ = new Precision[1ULL << qubit_num];
            imag_ = new Precision[1ULL << qubit_num];
            real_[0] = 1.0;
        }

        static inline QState<Precision>
        *merge_q_state(const QState<Precision> *a, const QState<Precision> *b, const uint64_t new_id);

        QState(uint64_t id) : QState(id, 1) {
            qubit_mapping_[id] = 0;
        }

        ~QState() {
            delete[] real_;
            delete[] imag_;
        }
    };


    template<typename Precision>
    class QStateSet {
    public:
        QStateSet(uint64_t qubit_num)
                : fa_(qubit_num), rank_(qubit_num, 1) {
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

        inline QState<Precision> *get_q_state(uint64_t qubit_id);

        inline QState<Precision> *merge_q_state(uint64_t qubit_id_a, uint64_t qubit_id_b);

    private:
        std::vector<uint64_t> fa_;
        std::vector<uint64_t> rank_;
        std::vector<QState<Precision> *> states_;

        inline uint64_t find(uint64_t id);
    };

    template<typename Precision>
    QState<Precision> *QStateSet<Precision>::get_q_state(uint64_t qubit_id) {
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
    QState<Precision> *QStateSet<Precision>::merge_q_state(uint64_t qubit_id_a, uint64_t qubit_id_b) {
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
    QState<Precision> *QState<Precision>::merge_q_state(
            const QState<Precision> *a,
            const QState<Precision> *b,
            const uint64_t new_id
    ) {
        auto s = new QState<Precision>(new_id, a->qubit_num_ + b->qubit_num_);
#define COMPLEX_MUL_REAL(a_r, a_i, b_r, b_i) ((a_r) * (b_r) - (a_i) * (b_i))
#define COMPLEX_MUL_IMAG(a_r, a_i, b_r, b_i) ((a_r) * (b_i) + (a_i) * (b_r))
        for (uint64_t i = 0; i < (1ULL << a->qubit_num_); i += 1) {
            auto blk_len = (1ULL << b->qubit_num_);
            if (b->qubit_num_ <= 2) {
                for (uint64_t j = 0; j < blk_len; j += 1) {
                    s->real_[i * blk_len + j] = COMPLEX_MUL_REAL(a->real_[i],
                                                                 a->imag_[i],
                                                                 b->real_[j],
                                                                 b->imag_[j]);
                    s->imag_[i * blk_len + j] = COMPLEX_MUL_IMAG(a->real_[i],
                                                                 a->imag_[i],
                                                                 b->real_[j],
                                                                 b->imag_[j]);
                }
            } else {
                __m256d ymm0 = _mm256_broadcast_sd(&a->real_[i]);
                __m256d ymm1 = _mm256_broadcast_sd(&a->imag_[i]);
                uint64_t batch_size = 4;
                for (uint64_t j = 0; j < blk_len; j += batch_size) {
                    __m256d ymm2 = _mm256_loadu_pd(&b->real_[j]);
                    __m256d ymm3 = _mm256_loadu_pd(&b->imag_[j]);
                    __m256d ymm4, ymm5; // res_r, res_i
                    COMPLEX_YMM_MUL(ymm0, ymm1, ymm2, ymm3, ymm4, ymm5);
                    _mm256_storeu_pd(&s->real_[i * blk_len + j], ymm4);
                    _mm256_storeu_pd(&s->imag_[i * blk_len + j], ymm5);
                }
            }
        }
#undef COMPLEX_MUL_REAL
#undef COMPLEX_MUL_IMAG

        for (const auto[k, v]: a->qubit_mapping_) {
            s->qubit_mapping_[k] = v;
        }
        for (const auto[k, v]: b->qubit_mapping_) {
            s->qubit_mapping_[k] = v + a->qubit_num_;
        }
        return s;
    }
}

#endif //SIM_BACK_Q_STATE_H
