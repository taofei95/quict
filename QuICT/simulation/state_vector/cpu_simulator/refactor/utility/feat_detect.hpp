// Contents in this file is copied and modified from
// https://github.com/Mysticial/FeatureDetector
//

#ifndef QUICT_SIM_BACKEND_UTILITY_FEAT_DETECT_H
#define QUICT_SIM_BACKEND_UTILITY_FEAT_DETECT_H

#include <cstdint>
#include <cstring>
#include <string>

#if _WIN32
#elif (__GNUC__) || (__clang__)
#include <cpuid.h>
#endif

namespace util {

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || \
    defined(_M_IX86)
// Check MSVC
#if _WIN32

inline __int64 xgetbv(unsigned int x) { return _xgetbv(x); }

// Check GCC & Clang
#elif defined(__GNUC__) || defined(__clang__)

inline uint64_t xgetbv(unsigned int index) {
  uint32_t eax, edx;
  __asm__ __volatile__("xgetbv" : "=a"(eax), "=d"(edx) : "c"(index));
  return ((uint64_t)edx << 32) | eax;
}

#define _XCR_XFEATURE_ENABLED_MASK 0

#else
#error "No cpuid intrinsic defined for compiler."
#endif
#else
#error "No cpuid intrinsic defined for processor architecture."
#endif

class Cpu_x86_64_Detector {
 public:
  //  Vendor
  bool Vendor_AMD;
  bool Vendor_Intel;

  //  OS Features
  bool OS_x64;
  bool OS_AVX;
  bool OS_AVX512;

  //  Misc.
  bool HW_MMX;
  bool HW_x64;
  bool HW_ABM;
  bool HW_RDRAND;
  bool HW_RDSEED;
  bool HW_BMI1;
  bool HW_BMI2;
  bool HW_ADX;
  bool HW_MPX;
  bool HW_PREFETCHW;
  bool HW_PREFETCHWT1;
  bool HW_RDPID;

  //  SIMD: 128-bit
  bool HW_SSE;
  bool HW_SSE2;
  bool HW_SSE3;
  bool HW_SSSE3;
  bool HW_SSE41;
  bool HW_SSE42;
  bool HW_SSE4a;
  bool HW_AES;
  bool HW_SHA;

  //  SIMD: 256-bit
  bool HW_AVX;
  bool HW_XOP;
  bool HW_FMA3;
  bool HW_FMA4;
  bool HW_AVX2;

  //  SIMD: 512-bit
  bool HW_AVX512_F;
  bool HW_AVX512_CD;

  //  Knights Landing
  bool HW_AVX512_PF;
  bool HW_AVX512_ER;

  //  Skylake Purley
  bool HW_AVX512_VL;
  bool HW_AVX512_BW;
  bool HW_AVX512_DQ;

  //  Cannon Lake
  bool HW_AVX512_IFMA;
  bool HW_AVX512_VBMI;

  //  Knights Mill
  bool HW_AVX512_VPOPCNTDQ;
  bool HW_AVX512_4FMAPS;
  bool HW_AVX512_4VNNIW;

  //  Cascade Lake
  bool HW_AVX512_VNNI;

  //  Cooper Lake
  bool HW_AVX512_BF16;

  //  Ice Lake
  bool HW_AVX512_VBMI2;
  bool HW_GFNI;
  bool HW_VAES;
  bool HW_AVX512_VPCLMUL;
  bool HW_AVX512_BITALG;

 public:
  Cpu_x86_64_Detector() {
    // Reset data
    std::memset(this, 0, sizeof(*this));
    // Detect features
    DetectHost();
  }

  void DetectHost() {
    //  OS Features
    OS_x64 = DetectOs_x64();
    OS_AVX = DetectOsAvx();
    OS_AVX512 = DetectOsAvx512();

    //  Vendor
    std::string vendor(GetVendorString());
    if (vendor == "GenuineIntel") {
      Vendor_Intel = true;
    } else if (vendor == "AuthenticAMD") {
      Vendor_AMD = true;
    }

    int info[4];
    CpuId(info, 0, 0);
    int nIds = info[0];

    CpuId(info, 0x80000000, 0);
    uint32_t nExIds = info[0];

    //  Detect Features
    if (nIds >= 0x00000001) {
      CpuId(info, 0x00000001, 0);
      HW_MMX = (info[3] & ((int)1 << 23)) != 0;
      HW_SSE = (info[3] & ((int)1 << 25)) != 0;
      HW_SSE2 = (info[3] & ((int)1 << 26)) != 0;
      HW_SSE3 = (info[2] & ((int)1 << 0)) != 0;

      HW_SSSE3 = (info[2] & ((int)1 << 9)) != 0;
      HW_SSE41 = (info[2] & ((int)1 << 19)) != 0;
      HW_SSE42 = (info[2] & ((int)1 << 20)) != 0;
      HW_AES = (info[2] & ((int)1 << 25)) != 0;

      HW_AVX = (info[2] & ((int)1 << 28)) != 0;
      HW_FMA3 = (info[2] & ((int)1 << 12)) != 0;

      HW_RDRAND = (info[2] & ((int)1 << 30)) != 0;
    }
    if (nIds >= 0x00000007) {
      CpuId(info, 0x00000007, 0);
      HW_AVX2 = (info[1] & ((int)1 << 5)) != 0;

      HW_BMI1 = (info[1] & ((int)1 << 3)) != 0;
      HW_BMI2 = (info[1] & ((int)1 << 8)) != 0;
      HW_ADX = (info[1] & ((int)1 << 19)) != 0;
      HW_MPX = (info[1] & ((int)1 << 14)) != 0;
      HW_SHA = (info[1] & ((int)1 << 29)) != 0;
      HW_RDSEED = (info[1] & ((int)1 << 18)) != 0;
      HW_PREFETCHWT1 = (info[2] & ((int)1 << 0)) != 0;
      HW_RDPID = (info[2] & ((int)1 << 22)) != 0;

      HW_AVX512_F = (info[1] & ((int)1 << 16)) != 0;
      HW_AVX512_CD = (info[1] & ((int)1 << 28)) != 0;
      HW_AVX512_PF = (info[1] & ((int)1 << 26)) != 0;
      HW_AVX512_ER = (info[1] & ((int)1 << 27)) != 0;

      HW_AVX512_VL = (info[1] & ((int)1 << 31)) != 0;
      HW_AVX512_BW = (info[1] & ((int)1 << 30)) != 0;
      HW_AVX512_DQ = (info[1] & ((int)1 << 17)) != 0;

      HW_AVX512_IFMA = (info[1] & ((int)1 << 21)) != 0;
      HW_AVX512_VBMI = (info[2] & ((int)1 << 1)) != 0;

      HW_AVX512_VPOPCNTDQ = (info[2] & ((int)1 << 14)) != 0;
      HW_AVX512_4FMAPS = (info[3] & ((int)1 << 2)) != 0;
      HW_AVX512_4VNNIW = (info[3] & ((int)1 << 3)) != 0;

      HW_AVX512_VNNI = (info[2] & ((int)1 << 11)) != 0;

      HW_AVX512_VBMI2 = (info[2] & ((int)1 << 6)) != 0;
      HW_GFNI = (info[2] & ((int)1 << 8)) != 0;
      HW_VAES = (info[2] & ((int)1 << 9)) != 0;
      HW_AVX512_VPCLMUL = (info[2] & ((int)1 << 10)) != 0;
      HW_AVX512_BITALG = (info[2] & ((int)1 << 12)) != 0;

      CpuId(info, 0x00000007, 1);
      HW_AVX512_BF16 = (info[0] & ((int)1 << 5)) != 0;
    }
    if (nExIds >= 0x80000001) {
      CpuId(info, 0x80000001, 0);
      HW_x64 = (info[3] & ((int)1 << 29)) != 0;
      HW_ABM = (info[2] & ((int)1 << 5)) != 0;
      HW_SSE4a = (info[2] & ((int)1 << 6)) != 0;
      HW_FMA4 = (info[2] & ((int)1 << 16)) != 0;
      HW_XOP = (info[2] & ((int)1 << 11)) != 0;
      HW_PREFETCHW = (info[2] & ((int)1 << 8)) != 0;
    }
  }

  void CpuId(int32_t out[4], int32_t eax, int32_t ecx) {
#if _MSC_VER
    __cpuidex(out, eax, ecx);
#elif (__GNUC__) || (__clang__)
    __cpuid_count(eax, ecx, out[0], out[1], out[2], out[3]);
#else
#error "Not supported this compiler!"
#endif
  }
  std::string GetVendorString() {
    int32_t cpu_info[4];
    char name[13];

    CpuId(cpu_info, 0, 0);
    memcpy(name + 0, &cpu_info[1], 4);
    memcpy(name + 4, &cpu_info[3], 4);
    memcpy(name + 8, &cpu_info[2], 4);
    name[12] = '\0';

    return name;
  }

 private:
  bool DetectOs_x64() {
    // Only compile codes in 64bit mode.
    return true;
  }

  bool DetectOsAvx() {
    //  Copied from: http://stackoverflow.com/a/22521619/922184

    bool avx_supported = false;

    int cpu_info[4];
    CpuId(cpu_info, 1, 0);

    bool os_uses_XSAVE_XRSTORE = (cpu_info[2] & (1 << 27)) != 0;
    bool cpu_avx_suport = (cpu_info[2] & (1 << 28)) != 0;

    if (os_uses_XSAVE_XRSTORE && cpu_avx_suport) {
      uint64_t xcrFeatureMask = xgetbv(_XCR_XFEATURE_ENABLED_MASK);
      avx_supported = (xcrFeatureMask & 0x6) == 0x6;
    }

    return avx_supported;
  }

  bool DetectOsAvx512() {
    if (!DetectOsAvx()) return false;

    uint64_t xcr_feature_mask = xgetbv(_XCR_XFEATURE_ENABLED_MASK);
    return (xcr_feature_mask & 0xe6) == 0xe6;
  }
};
}  // namespace util

#endif
