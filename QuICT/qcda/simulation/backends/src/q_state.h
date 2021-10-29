//
// Created by Ci Lei on 2021-10-14.
//

#ifndef SIM_BACK_Q_STATE_H
#define SIM_BACK_Q_STATE_H

#include <vector>
#include <map>
#include <complex>
#include <memory>
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

        static inline QState<Precision> *
        merge_q_state(const QState<Precision> *a, const QState<Precision> *b, const uint64_t new_id);

        inline void mapping_back();

        QState(uint64_t id) : QState(id, 1) {
            qubit_mapping_[id] = 0;
        }

        ~QState() {
            delete[] real_;
            delete[] imag_;
        }
    };


    template<typename Precision>
    QState<Precision> *QState<Precision>::merge_q_state(
            const QState<Precision> *a,
            const QState<Precision> *b,
            const uint64_t new_id
    ) {
        auto s = new QState<Precision>(new_id, a->qubit_num_ + b->qubit_num_);
#define COMPLEX_MUL_REAL(a_r, a_i, b_r, b_i) ((a_r) * (b_r) - (a_i) * (b_i))
#define COMPLEX_MUL_IMAG(a_r, a_i, b_r, b_i) ((a_r) * (b_i) + (a_i) * (b_r))
        if (a->qubit_num_ >= 2 && b->qubit_num_ >= 2) {
            for (uint64_t i = 0; i < (1ULL << a->qubit_num_); i += 1) {
                auto blk_len = (1ULL << b->qubit_num_);
                __m256d ymm0 = _mm256_broadcast_sd(&a->real_[i]);
                __m256d ymm1 = _mm256_broadcast_sd(&a->imag_[i]);
                constexpr uint64_t batch_size = 4;
                for (uint64_t j = 0; j < blk_len; j += batch_size) {
                    __m256d ymm2 = _mm256_loadu_pd(&b->real_[j]);
                    __m256d ymm3 = _mm256_loadu_pd(&b->imag_[j]);
                    __m256d ymm4, ymm5; // res_r, res_i
                    COMPLEX_YMM_MUL(ymm0, ymm1, ymm2, ymm3, ymm4, ymm5);
                    _mm256_storeu_pd(&s->real_[i * blk_len + j], ymm4);
                    _mm256_storeu_pd(&s->imag_[i * blk_len + j], ymm5);
                }
            }
        } else if (a->qubit_num_ >= 2 && b->qubit_num_ == 1) {
            __m256d ymm0 = _mm256_loadu2_m128d(b->real_, b->real_); // b0 b1 b0 b1, real
            __m256d ymm1 = _mm256_loadu2_m128d(b->imag_, b->imag_); // b0 b1 b0 b1, imag
            constexpr uint64_t batch_size = 2;
            for (uint64_t i = 0; i < (1ULL << a->qubit_num_); i += batch_size) {
                __m256d ymm2 = _mm256_loadu2_m128d(&a->real_[i], &a->real_[i]);
                __m256d ymm3 = _mm256_loadu2_m128d(&a->imag_[i], &a->imag_[i]);
                ymm2 = _mm256_permute4x64_pd(ymm2, 0b1101'1000);
                ymm3 = _mm256_permute4x64_pd(ymm3, 0b1101'1000);
                __m256d ymm4, ymm5;
                COMPLEX_YMM_MUL(ymm0, ymm1, ymm2, ymm3, ymm4, ymm5);
                _mm256_storeu_pd(&s->real_[i * 2], ymm4);
                _mm256_storeu_pd(&s->imag_[i * 2], ymm5);
            }
        } else { // a->qubit_num_ == 1 && b->qubit_num_ == 1
            for (uint64_t i = 0; i < 2; ++i) {
                for (uint64_t j = 0; j < 2; ++j) {
                    s->real_[i * 2 + j] = COMPLEX_MUL_REAL(a->real_[i],
                                                           a->imag_[i],
                                                           b->real_[j],
                                                           b->imag_[j]);
                    s->imag_[i * 2 + j] = COMPLEX_MUL_IMAG(a->real_[i],
                                                           a->imag_[i],
                                                           b->real_[j],
                                                           b->imag_[j]);
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

    template<typename Precision>
    void QState<Precision>::mapping_back() {
        // mapping back
        std::map<uint64_t, uint64_t> r_map;
        for (const auto[k, v]: qubit_mapping_) {
            r_map[v] = k;
        }
        auto p2n = 1ULL << qubit_num_;
        auto aug_map = std::unique_ptr<uint64_t[]>(new uint64_t[p2n]);
        std::fill(aug_map.get(), aug_map.get() + p2n, 0);
        for (uint64_t i = 0; i < p2n; ++i) {
            for (int k = 0; k < qubit_num_; ++k) {
                aug_map[i] |= ((i >> (qubit_num_ - 1 - r_map[k])) & 1) << (qubit_num_ - 1 - k);
            }
        }
        auto real_next = new Precision[p2n];
        auto imag_next = new Precision[p2n];
#pragma omp parallel for
        for (uint64_t i = 0; i < p2n; ++i) {
            real_next[i] = real_[aug_map[i]];
            imag_next[i] = imag_[aug_map[i]];
        }
        delete[] this->real_;
        delete[] this->imag_;
        this->real_ = real_next;
        this->imag_ = imag_next;
        for (uint64_t i = 0; i < qubit_num_; ++i) {
            qubit_mapping_[i] = i;
        }
    }
}

#endif //SIM_BACK_Q_STATE_H
