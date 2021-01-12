#define DLLEXPORT extern "C"
#include <iostream>
#include <complex>
#include "tbb/tbb.h"
#include <random>
#include <cmath>
#include <cstdio>
using namespace std;
using namespace tbb;

DLLEXPORT void merge_operator_func(
    int qureg_length, 
    complex<double> *values, 
    int merge_length, 
    complex<double> *merge_values,
    complex<double> *merge_matrix
    ){
    long long v_l = 1 << qureg_length;
    long long m_l = 1 << merge_length;
    //complex<double> *merge_matrix = (complex<double>*)malloc(sizeof(complex<double>) * (v_l * m_l));

    parallel_for(blocked_range<size_t>(0, v_l * m_l), [
        merge_matrix,
        m_l,
        merge_length,
        values,
        merge_values
    ](blocked_range<size_t>& blk){
        for(size_t j=blk.begin();j<blk.end();j++){
            merge_matrix[j] = values[j >> merge_length] * merge_values[j & (m_l - 1)];
        }
    });
    // return merge_matrix;
}

DLLEXPORT void single_operator_func(
    int qureg_length, 
    int index, 
    complex<double> *values, 
    complex<double> *matrix
    ){
    index = qureg_length - 1 - index;
    long long v_l = 1 << qureg_length;
    parallel_for(blocked_range<size_t>(0, v_l >> 1), [
        v_l,
        index,
        values,
        matrix
    ](blocked_range<size_t>& blk){
        for(size_t j=blk.begin();j<blk.end();j++){
            long long _0 = (j & ((1 << index) - 1)) 
                            + (j >> index << (index + 1));
            long long _1 = _0 + (1 << index);
            complex<double> __0 = values[_0] * matrix[0] + values[_1] * matrix[1];
            complex<double> __1 = values[_0] * matrix[2] + values[_1] * matrix[3];
            values[_0] = __0;
            values[_1] = __1;
        }
    });
}

DLLEXPORT bool measure_operator_func(
    int qureg_length, 
    int index, 
    complex<double> *values,
    double generation,
    double *prob
    ){
    index = qureg_length - 1 - index;
    long long v_l = 1 << qureg_length;

    complex<double> *newValues = (complex<double>*)malloc(sizeof(complex<double>) * (v_l >> 1));

    double _0 = parallel_reduce(blocked_range<size_t>(0, v_l >> 1), 0.f, [
        v_l,
        index, 
        values,
        newValues
    ](blocked_range<size_t>& blk, double init = 0)->double{
        for (size_t j=blk.begin(); j<blk.end(); j++) {
            long long _0 = (j & ((1 << index) - 1)) 
                            + (j >> index << (index + 1));
            double _real = real(values[_0]);
            double _imag = imag(values[_0]);
            init += _real * _real + _imag * _imag;
        }
        return init;
    },[](double x, double y)->double{return x+y;});

    bool _1 = generation > _0;
    *prob = 1 - _0;

    if (_1){
        generation = sqrt(1 - _0);
        parallel_for(blocked_range<size_t>(0, v_l >> 1), [
            v_l,
            index,
            values,
            generation,
            newValues
        ](blocked_range<size_t>& blk){
            for(size_t j=blk.begin();j<blk.end();j++){
                long long _1 = (j & ((1 << index) - 1)) 
                                + (j >> index << (index + 1))
                                + (1 << index);
                newValues[j] = values[_1] / generation;
            }
        });
    }else{
        generation = sqrt(_0);
        parallel_for(blocked_range<size_t>(0, v_l >> 1), [
            v_l,
            index,
            values,
            generation,
            newValues
        ](blocked_range<size_t>& blk){
            for(size_t j=blk.begin();j<blk.end();j++){
                long long _0 = (j & ((1 << index) - 1)) 
                                + (j >> index << (index + 1));
                newValues[j] = values[_0] / generation;
            }
        });
    }
    parallel_for(blocked_range<size_t>(0, v_l >> 1),[
        values,
        newValues
    ](blocked_range<size_t>& blk){
        for(size_t j=blk.begin();j<blk.end();j++){
            values[j] = newValues[j];
        }
    });
    free(newValues);

    return _1;
}

DLLEXPORT void reset_operator_func(
    int qureg_length, 
    int index, 
    complex<double> *values
    ){
    index = qureg_length - 1 - index;
    long long v_l = 1 << qureg_length;      
    double _0 = parallel_reduce(blocked_range<size_t>(0, v_l >> 1), 0.f, [
        v_l,
        index, 
        values
    ](blocked_range<size_t>& blk, double init = 0)->double{
        for (size_t j=blk.begin(); j<blk.end(); j++) {
            long long _0 = (j & ((1 << index) - 1)) 
                            + (j >> index << (index + 1));
            double _real = real(values[_0]);
            double _imag = imag(values[_0]);
            init += _real * _real + _imag * _imag;
        }
        return init;
    },[](double x, double y)->double{return x+y;});

    double generation = sqrt(_0);

    if (generation < 1e-6){
        parallel_for(blocked_range<size_t>(0, v_l >> 1), [
            v_l,
            index,
            values,
            generation
        ](blocked_range<size_t>& blk){
            for(size_t j=blk.begin();j<blk.end();j++){
                long long _1 = (j & ((1 << index) - 1)) 
                                + (j >> index << (index + 1))
                                + (1 << index);
                values[j] = values[_1];
            }
        });
    }else{
        parallel_for(blocked_range<size_t>(0, v_l >> 1), [
            v_l,
            index,
            values,
            generation
        ](blocked_range<size_t>& blk){
            for(size_t j=blk.begin();j<blk.end();j++){
                long long _0 = (j & ((1 << index) - 1)) 
                                + (j >> index << (index + 1));
                values[j] = values[_0] / generation;
            }
        });
    }
}

DLLEXPORT void control_single_operator_func(
    int qureg_length, 
    int cindex, 
    int tindex, 
    complex<double> *values, 
    complex<double> *matrix
    ){
    cindex = qureg_length - 1 - cindex;
    tindex = qureg_length - 1 - tindex;
    long long v_l = 1 << qureg_length;
    if (tindex > cindex){
        parallel_for(blocked_range<size_t>(0, v_l >> 2), [
            v_l,
            cindex,
            tindex,
            values,
            matrix
        ](blocked_range<size_t>& blk){
            for(size_t j=blk.begin();j<blk.end();j++){
                long long gw = j >> cindex << (cindex + 1);
                long long _0 = (1 << cindex) + 
                    (gw & ((1 << tindex) - (1 << cindex))) + 
                    (gw >> tindex << (tindex + 1)) + 
                    (j & ((1 << cindex) - 1));
                long long _1 = _0 + (1 << tindex);
                complex<double> __0 = values[_0] * matrix[0] + values[_1] * matrix[1];
                complex<double> __1 = values[_0] * matrix[2] + values[_1] * matrix[3];
                values[_0] = __0;
                values[_1] = __1;
            }
        });
    }else{
        parallel_for(blocked_range<size_t>(0, v_l >> 2), [
            v_l,
            cindex,
            tindex,
            values,
            matrix
        ](blocked_range<size_t>& blk){
            for(size_t j=blk.begin();j<blk.end();j++){
                long long gw = j >> tindex << (tindex + 1);
                long long _0 = (1 << cindex) + 
                    (gw & ((1 << cindex) - (1 << tindex))) + 
                    (gw >> cindex << (cindex + 1)) + 
                    (j & ((1 << tindex) - 1));
                long long _1 = _0 + (1 << tindex);
                complex<double> __0 = values[_0] * matrix[0] + values[_1] * matrix[1];
                complex<double> __1 = values[_0] * matrix[2] + values[_1] * matrix[3];
                values[_0] = __0;
                values[_1] = __1;
            }
        });
    }
}

DLLEXPORT void ccx_single_operator_func(
    int qureg_length, 
    int cindex1, 
    int cindex2, 
    int tindex, 
    complex<double> *values){
    cindex1 = qureg_length - 1 - cindex1;
    cindex2 = qureg_length - 1 - cindex2;
    tindex = qureg_length - 1 - tindex;
    int indexlist[3];
    indexlist[0] = cindex1;
    indexlist[1] = cindex2;
    indexlist[2] = tindex;
    sort(indexlist, indexlist + 3);

    long long v_l = 1 << qureg_length;
    parallel_for(blocked_range<size_t>(0, v_l >> 3), [
        v_l,
        indexlist,
        cindex1,
        cindex2,
        tindex,
        values
    ](blocked_range<size_t>& blk){
        for(size_t j=blk.begin();j<blk.end();j++){
            long long gw = j >> indexlist[0] << (indexlist[0] + 1);
            long long gwg = gw >> indexlist[1] << (indexlist[1] + 1);
            long long _0 = (j & ((1 << indexlist[0]) - 1))
                        + (1 << cindex1) + (1 << cindex2)
                        + (gw & ((1 << indexlist[1]) - (1 << indexlist[0]))) 
                        + (gwg & ((1 << indexlist[2]) - (1 << indexlist[1]))) 
                        + (gwg >> indexlist[2] << (indexlist[2] + 1));
            swap(values[_0], values[_0 + (1 << tindex)]);
        }
    });
}

DLLEXPORT void unitary_operator_gate(int qureg_length, complex<double> *values, long long  *index, int index_count, complex<double> *matrix){
    int *indexlist = (int*)malloc(index_count * sizeof(int));
    for (int i = 0;i < index_count;++i)
        indexlist[i] = index[i];
    sort(indexlist, indexlist + index_count);


    long long v_l = 1 << qureg_length;
    long long xl_l = 1 << index_count;
    complex<double> *newValues = (complex<double>*)malloc(sizeof(complex<double>) * xl_l);
    
    parallel_for(blocked_range<size_t>(0, v_l >> index_count), [
        v_l,
        matrix,
        indexlist,
        index,
        xl_l,
        index_count,
        values,
        newValues
    ](blocked_range<size_t>& blk){
        for(size_t j=blk.begin();j<blk.end();j++){
            long long other = j & ((1 << indexlist[0]) - 1);
            long long gw = j >> indexlist[0] << (indexlist[0] + 1);
            for (int i = 1;i < index_count;++i){
                if (gw == 0)
                    break;
                other += gw & ((1 << indexlist[i]) - (1 << indexlist[i - 1]));
                gw = gw >> indexlist[i] << (indexlist[i] + 1);
            }
            other += gw;
            for (int i = 0;i < xl_l;++i){
                long long now = other;
                for (int k = 0;k < index_count;++k)
                    if (i & (1 << k))
                        now += 1 << index[k];
                newValues[now + other] = complex<double>(0, 0);
                for (int k = 0;k < xl_l;++k){
                    long long shift = other;
                    for (int l = 0;l < index_count;++l)
                        if (i & (1 << l))
                            shift += 1 << index[l];
                    newValues[now + other] += matrix[k] * values[shift];
                }
            }
        }
    });
    free(indexlist);
    parallel_for(blocked_range<size_t>(0, v_l),[
        values,
        newValues
    ](blocked_range<size_t>& blk){
        for(size_t j=blk.begin();j<blk.end();j++){
            values[j] = newValues[j];
        }
    });
    free(newValues);
}

DLLEXPORT void perm_operator_gate(int qureg_length, complex<double> *values, long long *index, int index_count, long long *perm){
    //for (int i = 0;i < index_count;++i)
    //    index[i] = qureg_length - 1 - index[i];
    int *indexlist = (int*)malloc(index_count * sizeof(int));
    for (int i = 0;i < index_count;++i)
        indexlist[i] = index[i];
    sort(indexlist, indexlist + index_count);

    long long v_l = 1 << qureg_length;
    long long xl_l = 1 << index_count;
    complex<double> *newValues = (complex<double>*)malloc(sizeof(complex<double>) * v_l);
    long long *perms_to = (long long *)malloc(sizeof(sizeof(long long)) * xl_l);
    parallel_for(blocked_range<size_t>(0, xl_l), [
        perms_to,
        perm,
        qureg_length,
        index,
        index_count
    ](blocked_range<size_t>& blk){
        for(size_t j=blk.begin();j<blk.end();j++){
            perms_to[j] = 0;
            long long now = perm[j];
            perm[j] = 0;
            for (int i = 0;i < index_count;++i){
                if ((1 << i) & now)
                    perm[j] += 1 << index[i];
                if ((1 << i) & j)
                    perms_to[j] += 1 << index[i];
                //printf("%d %lld %lld %lld\n", j, perm[j], perms_to[j], now);
            }
        }
    });
    parallel_for(blocked_range<size_t>(0, v_l >> index_count), [
        v_l,
        perms_to,
        perm,   
        indexlist,
        index,
        xl_l,
        index_count,
        values,
        newValues
    ](blocked_range<size_t>& blk){
        for(size_t j=blk.begin();j<blk.end();j++){
            long long other = j & ((1 << indexlist[0]) - 1);
            long long gw = j >> indexlist[0] << (indexlist[0] + 1);
            for (int i = 1;i < index_count;++i){
                if (gw == 0)
                    break;
                other += gw & ((1 << indexlist[i]) - (1 << indexlist[i - 1]));
                gw = gw >> indexlist[i] << (indexlist[i] + 1);
            }
            other += gw;
            for (int i = 0;i < xl_l;++i){
                newValues[other + perm[i]] = values[other + perms_to[i]];
            }
        }
    });
    free(indexlist);
    parallel_for(blocked_range<size_t>(0, v_l),[
        values,
        newValues
    ](blocked_range<size_t>& blk){
        for(size_t j=blk.begin();j<blk.end();j++){
            values[j] = newValues[j];
        }
    });
    free(newValues);
}

DLLEXPORT complex<double> * amplitude_cheat_operator(
    complex<double> *values, 
    long long *values_length, 
    int tangle_number,
    long long *qubit_map  
    ){
    complex<double> **segment = (complex<double> **)malloc(sizeof(complex<double> *) * tangle_number);
    long long all_length = 0;
    complex<double> *now = values;
    for(int i = 0;i < tangle_number;++i){
        all_length += values_length[i];
        segment[i] = now;
        now += (1 << values_length[i]);
    }
    long long v_l = 1 << all_length;
    complex<double> *back = (complex<double>*)malloc(sizeof(complex<double>) * v_l);
    parallel_for(blocked_range<size_t>(0, v_l), [
        all_length,
        values_length,
        tangle_number,
        segment,
        qubit_map,
        back
    ](blocked_range<size_t>& blk){
        for(size_t k=blk.begin();k<blk.end();k++){
            int now = 0;
            complex<double> ans(1, 0);
            for(int i = 0;i < tangle_number;++i){
                long long index = 0;
                for (int j = 0;j < values_length[i];++j)
                    if ((1 << (all_length - 1 - qubit_map[now + j])) & k)
                        index += 1 << (values_length[i] - 1 - j);
                ans *= segment[i][index];
                now += values_length[i];
            }
            back[k] = ans;
        }
    });
    free(segment);
    return back;
}

DLLEXPORT double* partial_prob_cheat_operator(
    complex<double> *values,
    long long *values_length,
    int tangle_number,
    int qubit_number,
    long long *qubit_map
    ){
    complex<double> **segment = (complex<double> **)malloc(sizeof(complex<double> *) * tangle_number);
    complex<double> *now = values;
    for(int i = 0;i < tangle_number;++i){
        segment[i] = now;
        now += (1 << values_length[i]);
    }
    long long v_l = 1 << qubit_number;
    double *back = (double*)malloc(sizeof(double) * v_l);
    parallel_for(blocked_range<size_t>(0, v_l), [
        qubit_number,
        values_length,
        tangle_number,
        segment,
        qubit_map,
        back
    ](blocked_range<size_t>& blk){
        for(size_t k=blk.begin();k<blk.end();k++){
            back[k] = 1;
            int tangle_iter = 0;
            for(int i = 0;i < tangle_number;++i){
                int tangle_length = values_length[i];

                int *indexlist = (int*)malloc(tangle_length * sizeof(int));
                int index_count = 0;
                long long fix_position = 0;
                for (int j = 0;j < qubit_number;++j){
                    int index = qubit_map[j] - tangle_iter;
                    if (index >= 0 && index < tangle_length){
                        if ((1 << (qubit_number - j - 1)) & k){
                            fix_position += 1 << (tangle_length - index - 1);
                        }
                        indexlist[index_count++] = tangle_length - index - 1;
                    }
                }

                if (index_count != 0){
                    sort(indexlist, indexlist + index_count);
                }

                back[k] *= parallel_reduce(blocked_range<size_t>(0, 1 << (tangle_length - index_count)), 0.f, [
                    segment,
                    i,
                    fix_position,
                    indexlist,
                    index_count,
                    k
                ](blocked_range<size_t>& blk, double init = 0)->double{
                    for (size_t j=blk.begin(); j<blk.end(); j++) {
                        long long other;
                        if (index_count != 0){
                            other = j & ((1 << indexlist[0]) - 1);
                            long long gw = j >> indexlist[0] << (indexlist[0] + 1);
                            for (int i = 1;i < index_count;++i){
                                if (gw == 0)
                                    break;
                                other += gw & ((1 << indexlist[i]) - (1 << indexlist[i - 1]));
                                gw = gw >> indexlist[i] << (indexlist[i] + 1);
                            }
                            other += gw;
                            other += fix_position;
                        }else{
                            other = j;
                        }

                        double _real = real(segment[i][other]);
                        double _imag = imag(segment[i][other]);
                        init += _real * _real + _imag * _imag;
                    }
                    return init;
                },[](double x, double y)->double{return x + y;});


                free(indexlist);

                tangle_iter += tangle_length;
            }
        }
    });
    free(segment);
    return back;
}

DLLEXPORT void control_mul_perm_operator_gate(int qureg_length, complex<double> *values, long long *index, int control, int index_count, int a, int N){
    for (int i = 0;i < index_count;++i)
        index[i] = qureg_length - 1 - index[i];
    control = qureg_length - 1 - control;
    int *indexlist = (int*)malloc(index_count * sizeof(int));
    for (int i = 0;i < index_count;++i)
        indexlist[i] = index[i];
    sort(indexlist, indexlist + index_count);

    long long v_l = 1 << qureg_length;
    long long xl_l = 1 << index_count;
    complex<double> *newValues = (complex<double>*)malloc(sizeof(complex<double>) * v_l);

    long long *perm = (long long *)malloc(sizeof(sizeof(long long)) * N);
    long long *perms_to = (long long *)malloc(sizeof(sizeof(long long)) * N);
    parallel_for(blocked_range<size_t>(1, N), [
        perms_to,
        perm,
        qureg_length,
        index,
        index_count,
        a,
        N
    ](blocked_range<size_t>& blk){
        for(size_t j=blk.begin();j<blk.end();j++){
            long long perms = 0;
            long long pos = 0;
            long long to = (long long)a * j % N;

            for (int i = 0;i < index_count;++i){
                if ((1 << i) & to)
                    perms += (long long)1 << index[i];
                if ((1 << i) & j)
                    pos += (long long)1 << index[i];
            }
            perms_to[j] = perms;
            perm[j] = pos;
        }
    });


    parallel_for(blocked_range<size_t>(0, v_l >> (index_count + 1)), [
        v_l,
        perms_to,
        perm,
        index,
        N,
        index_count,
        values,
        indexlist,
        newValues,
        control
    ](blocked_range<size_t>& blk){
        for(size_t j=blk.begin();j<blk.end();j++){
            long long other = j & ((1 << indexlist[0]) - 1);
            long long gw = j >> indexlist[0] << (indexlist[0] + 1);
            for (int i = 1;i < index_count;++i){
                if (gw == 0)
                    break;
                other += gw & ((1 << indexlist[i]) - (1 << indexlist[i - 1]));
                gw = gw >> indexlist[i] << (indexlist[i] + 1);
            }
            other += gw;
            other += (1 << control);
            for (int i = 1;i < N;++i){
                newValues[other + perms_to[i]] = values[other + perm[i]];
            }
        }
    });
    parallel_for(blocked_range<size_t>(0, v_l >> (index_count + 1)),[
        v_l,
        perms_to,
        perm,
        index,
        N,
        index_count,
        values,
        indexlist,
        newValues,
        control
    ](blocked_range<size_t>& blk){
        for(size_t j=blk.begin();j<blk.end();j++){
            long long other = j & ((1 << indexlist[0]) - 1);
            long long gw = j >> indexlist[0] << (indexlist[0] + 1);
            for (int i = 1;i < index_count;++i){
                if (gw == 0)
                    break;
                other += gw & ((1 << indexlist[i]) - (1 << indexlist[i - 1]));
                gw = gw >> indexlist[i] << (indexlist[i] + 1);
            }
            other += gw;
            other += (1 << control);
            for (int i = 1;i < N;++i){
                //printf("%d %d %d\n", other,other + perm[i], other + perms_to[i]);

                values[other + perms_to[i]] = newValues[other + perms_to[i]];
            }
        }
    });
    free(indexlist);
    free(perm);
    free(perms_to);
    free(newValues);
}

DLLEXPORT void shor_classical_initial_gate(int qureg_length, complex<double> *values, long long *index, int index_count, int x, int N, int u){
    for (int i = 0;i < index_count;++i)
        index[i] = qureg_length - 1 - index[i];

    long long v_l = 1 << qureg_length;
    long long xl_l = 1 << index_count;
    long long *mul_mod = (long long*)malloc(sizeof(long long) * index_count);
    long long aa = x;
    for (int i = 0;i < index_count;++i){
        mul_mod[index_count - 1 - i] = aa;
        aa = (long long)aa * aa % N;
    }

    int number = parallel_reduce(blocked_range<size_t>(0, xl_l), 0.f, [
        qureg_length,
        index,
        index_count,
        mul_mod,
        N,
        u
    ](blocked_range<size_t>& blk, long long init = 0)->long long{
        for (size_t j=blk.begin(); j<blk.end(); j++) {
            long long over = 1;
            for (int i = 0;i < index_count;++i){
                if ((1 << i) & j){
                    over = over * mul_mod[i] % N;
                }
            }
            if (over == u)
                ++init;
        }
        return init;

    },[](long long x, long long y)->long long{return x+y;});

    long double state = (long double)1.0 / sqrt(number);
    values[0] = 0;

    parallel_for(blocked_range<size_t>(0, xl_l), [
        qureg_length,
        index,
        index_count,
        mul_mod,
        N,
        u,
        state,
        values
    ](blocked_range<size_t>& blk){
        for(size_t j=blk.begin();j<blk.end();j++){
            long long pos = 0;
            long long over = 1;
            for (int i = 0;i < index_count;++i){
                if ((1 << i) & j){
                    pos += 1 << index[i];
                    over = over * mul_mod[i] % N;
                }
            }
            if (over == u){
                values[pos] = state;
            }
        }
    });

}