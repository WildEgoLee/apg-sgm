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
    int nIds = cpu_info[0];
    if (nIds >= 7) {
        __cpuidex(cpu_info, 7, 0);
        return (cpu_info[1] & (1 << 5)) != 0; // EBX bit 5 is AVX2
    }
    return false;
#elif defined(__GNUC__) || defined(__clang__)
    return __builtin_cpu_supports("avx2") > 0;
#else
    return false;
#endif
}

} // namespace detail
} // namespace apg
