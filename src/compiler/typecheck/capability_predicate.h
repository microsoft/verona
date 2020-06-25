// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "compiler/mapper.h"
#include "compiler/type.h"

namespace verona::compiler
{
  /**
   * Properties about capabilities, and by extension about types.
   *
   * A type satisfies a predicate if and only if all its disjunctions do.
   * This is why we need distinct positive and negative predicates.
   *
   * For example, `mut` is Writable, `imm` is NonWritable, but `(mut | imm)` is
   * neither.
   *
   * It is possible for uninhabited types to satisfy contradictory predicates.
   * For example bottom (the empty union) satisfies all the capabilities.
   * `(mut & imm)` satisfies both Writable and NonWritable.
   *
   * The enum uses power of two values as they get OR-ed and represented as a
   * bitmask by PredicateSet.
   */
  enum class CapabilityPredicate : uint8_t
  {
    Readable = 1,
    Writable = 2,
    NonWritable = 4,
    NonLinear = 8,
    Sendable = 16,
  };

  /**
   * Set of CapabilityPredicate values, represented as a bitmask.
   */
  struct PredicateSet
  {
    /**
     * Implicit conversion from a single predicate.
     */
    PredicateSet(CapabilityPredicate predicate);

    static PredicateSet empty();
    static PredicateSet all();

    bool contains(CapabilityPredicate predicate) const;
    bool equals(PredicateSet set) const;
    PredicateSet operator|(PredicateSet other) const;
    PredicateSet operator&(PredicateSet other) const;

  private:
    // This is the underlying type specified by the enum class.
    typedef std::underlying_type_t<CapabilityPredicate> underlying_type;

    explicit PredicateSet(underlying_type values);
    underlying_type values_;
  };

  /**
   * Combine two predicates to form a predicate set.
   */
  PredicateSet operator|(CapabilityPredicate lhs, CapabilityPredicate rhs);

  /**
   * Determine the set of predicates satisfied by a type.
   */
  PredicateSet predicates_for_type(TypePtr type);

  std::ostream& operator<<(std::ostream& out, CapabilityPredicate predicate);
}
