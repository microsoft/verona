// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#ifdef USE_SYSTEMATIC_TESTING
#  include "ds/scramble.h"
#  include "test/xoroshiro.h"
#endif

namespace Systematic
{
#ifdef USE_SYSTEMATIC_TESTING
  /// Return a mutable reference to the pseudo random number generator (PRNG).
  /// It is assumed that the PRNG will only be setup once via `set_seed`. After
  /// it is setup, the PRNG must only be used via `get_prng_next`.
  static inline xoroshiro::p128r32& get_prng_for_setup()
  {
    static xoroshiro::p128r32 prng;
    return prng;
  }

  /// Return the next pseudo random number.
  static inline uint32_t get_prng_next()
  {
    auto& prng = get_prng_for_setup();
    static std::atomic_flag lock;
    snmalloc::FlagLock l{lock};
    return prng.next();
  }

  /// Return a mutable reference to the scrambler. It is assumed that the
  /// scrambler will only be setup once via `set_seed`. After it is setup, the
  /// scrambler must only be accessed via a const reference
  /// (see `get_scrambler`).
  static inline verona::Scramble& get_scrambler_for_setup()
  {
    static verona::Scramble scrambler;
    return scrambler;
  }

  /// Return a const reference to the scrambler.
  static inline const verona::Scramble& get_scrambler()
  {
    return get_scrambler_for_setup();
  }

  static inline void set_seed(uint64_t seed)
  {
    auto& rng = get_prng_for_setup();
    rng.set_state(seed);
    get_scrambler_for_setup().setup(rng);
  }

  /// 1/(2^range_bits) likelyhood of returning true
  static inline bool coin(size_t range_bits = 1)
  {
    assert(range_bits < 20);
    return (get_prng_next() & ((1ULL << range_bits) - 1)) == 0;
  }
#endif
} // namespace Systematic