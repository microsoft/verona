// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include <snmalloc/ds_core/ds_core.h>

namespace verona::rt
{
  namespace bits
  {
    using namespace snmalloc::bits;

    /**
     * Extract the bits in the range [hi, lo], where lo is shifted to the
     * rightmost bit position.
     */
    template<uint8_t hi, uint8_t lo, typename T>
    static T extract(T instr)
    {
      static_assert(std::is_integral_v<T>, "Type must be integral");
      static_assert(hi <= (sizeof(T) * 8));
      static_assert(hi >= lo);
      constexpr T len = (hi - lo) + 1;
      constexpr T mask = (T)~0 >> ((T)(sizeof(T) * 8) - len);
      return (instr >> lo) & mask;
    }

    template<class T>
    constexpr T inc_mod(T v, T mod)
    {
      static_assert(
        std::is_integral<T>::value, "inc_mod can only be used on integers");

      using S = std::make_signed_t<T>;
      size_t shift = (sizeof(S) * 8) - 1;

      T a = v + 1;
      S b = static_cast<S>(mod - a - 1);
      return a & static_cast<T>(~(b >> shift));
    }

    inline static size_t hash(uintptr_t p)
    {
      size_t x = static_cast<size_t>(p);

      if constexpr (
        std::numeric_limits<uintptr_t>::max() ==
        std::numeric_limits<uint64_t>::max())
      {
        x = ~x + (x << 21);
        x = x ^ (x >> 24);
        x = (x + (x << 3)) + (x << 8);
        x = x ^ (x >> 14);
        x = (x + (x << 2)) + (x << 4);
        x = x ^ (x >> 28);
        x = x + (x << 31);
      }
      else
      {
        x = ~x + (x << 15);
        x = x ^ (x >> 12);
        x = x + (x << 2);
        x = x ^ (x >> 4);
        x = (x + (x << 3)) + (x << 11);
        x = x ^ (x >> 16);
      }

      return x;
    }

    inline size_t clz32(uint32_t x)
    {
#if defined(_MSC_VER)
#  ifdef USE_LZCNT
      return __lzcnt(x);
#  else
      unsigned long index;
      _BitScanReverse(&index, (unsigned long)x);
      return BITS - index - 1;
#  endif
#else
#  ifdef PLATFORM_BITS_64
      return static_cast<size_t>(__builtin_clz(x));
#  else
      return static_cast<size_t>(__builtin_clzl(x));
#  endif
#endif
    }

  }; // namespace bits
}; // namespace verona::rt
