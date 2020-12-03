// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "compiler/mapper.h"

namespace verona::compiler
{
  /*
   * List of types used to replace type parameters.
   *
   * An Instantiation covers all type parameters in scope, from the top-level.
   * For instance, given the following code, an instantiation for method bar
   * must have 3 types, for X, Y, and Z.
   *
   * ```
   * class C[X, Y]
   *   fun m[Z]()
   * ```
   */
  struct Instantiation
  {
    friend class Applier;

  public:
    explicit Instantiation(TypeList types) : types_(types) {}
    explicit Instantiation(TypeList types, const TypeList& more_types)
    : types_(types)
    {
      std::copy(
        more_types.begin(), more_types.end(), std::back_inserter(types_));
    }

    static Instantiation empty()
    {
      return Instantiation(TypeList());
    }

    template<typename T>
    auto apply(Context& context, const T& value) const
    {
      Applier v(context, *this);
      return v.apply(value);
    }

    const TypeList& types() const
    {
      return types_;
    }

    bool operator<(const Instantiation& other) const
    {
      return types_ < other.types_;
    }

    bool operator==(const Instantiation& other) const
    {
      return types_ == other.types_;
    }

  private:
    TypeList types_;

    class Applier : public RecursiveTypeMapper
    {
    public:
      explicit Applier(Context& context, const Instantiation& instance)
      : RecursiveTypeMapper(context), instance_(instance)
      {}

      TypePtr visit_type_parameter(const TypeParameterPtr& ty) final;

    private:
      const Instantiation& instance_;
    };
  };
}
