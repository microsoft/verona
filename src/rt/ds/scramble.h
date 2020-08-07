// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once
#include "test/xoroshiro.h"

#include <cstdint>
#include <iomanip>
#include <iostream>

namespace verona
{
  /**
   * This class is used to provide alternative orderings on a pointer type.
   *
   * This class encapsulates a simple Feistel network to provide a permutation
   * function on pointer sized values, which is then used to compare the
   * pointers.  This is useful for testing different orders of cown acquisition
   * inside a multi-message.
   *
   * The implementation is guaranteed to be a permutation as a Feisel network is
   * invertible.  This is not a crytpographic primitive, it is just a compact
   * way to provide random orders on pointers.
   */
  class Scramble
  {
    // Number of rounds to get value.
    static constexpr size_t ROUNDS = 8;

    static constexpr size_t PTR_HALF_SHIFT = (sizeof(uintptr_t) * 8) / 2;
    static constexpr size_t MASK_BOTTOM =
      (((uintptr_t)1) << PTR_HALF_SHIFT) - 1;

    uintptr_t keys[ROUNDS];

  public:
    Scramble() {}

    void setup(xoroshiro::p128r32& r)
    {
      for (size_t i = 0; i < ROUNDS; i++)
      {
        keys[i] = (uintptr_t)r.next();
      }
    }

    uintptr_t perm(uintptr_t p)
    {
      uintptr_t l = p & MASK_BOTTOM;
      uintptr_t r = (p >> PTR_HALF_SHIFT);

      for (size_t i = 0; i < ROUNDS; i++)
      {
        auto nl = r ^ (l * keys[i]);
        r = l;
        l = nl & MASK_BOTTOM;
      }

      return l + ((uintptr_t)r << PTR_HALF_SHIFT);
    }
  };
}
