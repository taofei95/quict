#ifndef QUICT_SIM_BACKEND_UTILITY_COMPLEX_DETECT_H
#define QUICT_SIM_BACKEND_UTILITY_COMPLEX_DETECT_H

#include <complex>
#include <type_traits>

namespace util {
template <class Complex>
struct complex_is_f32 : public std::false_type {};

template <class Complex>
struct complex_is_f64 : public std::false_type {};

template <>
struct complex_is_f32<std::complex<float>> : public std::true_type {};

template <>
struct complex_is_f64<std::complex<double>> : public std::true_type {};

template <class Complex>
inline constexpr bool complex_is_f32_v = complex_is_f32<Complex>::value;

template <class Complex>
inline constexpr bool complex_is_f64_v = complex_is_f64<Complex>::value;
}  // namespace util

#endif
