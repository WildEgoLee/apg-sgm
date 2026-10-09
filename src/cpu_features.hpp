#pragma once

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
#endif

namespace apg {
namespace detail {

inline bool is_avx2_supported() {
#if defined(_MSC_VER)
    int cpu_info[4];
    __cpuid(cpu_info, 0);
    const int nIds = cpu_info[0];
    if (nIds < 7) {
        return false;
    }

    // Check OSXSAVE (ECX bit 27) and AVX (ECX bit 28)
    __cpuid(cpu_info, 1);
    const int ecx1 = cpu_info[2];
    if ((ecx1 & (1 << 27)) == 0 || (ecx1 & (1 << 28)) == 0) {
        return false;
    }

    // Verify OS has enabled both XMM (bit 1) and YMM (bit 2) state via XGETBV
    const unsigned __int64 xcr0 = _xgetbv(0);
    if ((xcr0 & 0x6) != 0x6) {
        return false;
    }

    // Check AVX2 capability (EBX bit 5 of leaf 7)
    __cpuidex(cpu_info, 7, 0);
    return (cpu_info[1] & (1 << 5)) != 0;
#elif defined(__GNUC__) || defined(__clang__)
    return __builtin_cpu_supports("avx2") > 0;
#else
    return false;
#endif
}

} // namespace detail
} // namespace apg
