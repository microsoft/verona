// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
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

    inline static size_t hash(void* p)
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

    /**
     * Backpressure bit representation
     *
     * |31          24|23              8|7      0|
     * |   pressure   |   load history  |  load  |
     *
     */

    static inline uint8_t backpressure_pressure(uint32_t bp)
    {
      return (bp & 0xff'0000'00) >> 24;
    }

    static inline uint32_t backpressure_pressure_add(uint32_t bp, int32_t n)
    {
      int32_t p = (int32_t)backpressure_pressure(bp) + n;
      assert(p > 0);
      p = (p > 0xff) ? 0xff : p;
      return ((uint32_t)p << 24) | (bp & 0x00'ffff'ff);
    }

    static inline uint32_t backpressure_pressure_reset(uint32_t bp)
    {
      return bp & 0x00'ffff'ff;
    }

    static inline uint32_t backpressure_load(uint32_t bp)
    {
      const uint32_t h3 = (bp & 0xf000'00) >> 16;
      const uint32_t h2 = (bp & 0x0f00'00) >> 12;
      const uint32_t h1 = (bp & 0x00f0'00) >> 8;
      const uint32_t h0 = (bp & 0x000f'00) >> 4;
      return (h3 + h2 + h1 + h0) | (bp & 0xff);
    }

    static inline uint32_t backpressure_load_inc(uint32_t bp)
    {
      if ((bp & 0xff) < 0xff)
        bp++;
      return bp;
    }

    static inline uint32_t backpressure_load_reset(uint32_t bp)
    {
      uint32_t hist = (bp & 0x00'ffff'f0) << 4;
      hist &= 0x00'ffff'00;
      return (bp & 0xff'0000'00) | hist;
    }

  }; // namespace bits
}; // namespace verona::rt
