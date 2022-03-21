#define DLLEXPORT extern "C"
#include <complex>
#include "tbb/tbb.h"
#include <random>
#include <cmath>
#include <cstdio>
using namespace std;
using namespace tbb;
const long double eps = 1e-20;

DLLEXPORT int state_theta_computation(
    int state_length, 
    long double *values,
    long double *back
    ){
    int n = ceil(log2(state_length));
    long long N = (long long)1 << n;

    long double init_cof = parallel_reduce(blocked_range<size_t>(0, state_length), 0.f, [
        values
    ](blocked_range<size_t>& blk, long double init = 0)->long double{
        for (size_t j=blk.begin(); j<blk.end(); j++) {
            init += values[j];
        }
        return init;
    },[](long double x, long double y)->long double{return x+y;});
    if (abs(init_cof) < eps)
        return -1;
    for (int i = 0, now = 0;i < n;++i){
        int add = (1 << i);
        parallel_for(blocked_range<size_t>(0, add), [
            now,
            values,
            back,
            i,
            n,
            init_cof
        ](blocked_range<size_t>& blk){
            for(size_t j=blk.begin();j<blk.end();j++){
                int pre = j << (n - i);
                int suf = 1 << (n - i - 1);
                long double _0 = 0, _1 = 0;
                for (int k = 0;k < suf;++k){
                    _0 += values[pre + k] * values[pre + k];
                    _1 += values[pre + k + (1 << (n - i - 1))] * values[pre + k + (1 << (n - i - 1))];
                }
                if (_0 < eps && _1 < eps){
                    back[now + j] = 0;
                }else{
                    back[now + j] = 2 * acosl(sqrtl(_0 / (_0 + _1)));
                }
            }
        });
        now += add;
    }

    return 0;
}
