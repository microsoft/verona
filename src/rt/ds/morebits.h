// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "ds/address.h"
#include "ds/bits.h"

namespace verona::rt
{
  namespace bits
  {
    using namespace snmalloc::bits;

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

    inline static size_t hash(const void* p)
    {
      size_t x = static_cast<size_t>(snmalloc::address_cast(p));

      if (snmalloc::bits::is64())
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
