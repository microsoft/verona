// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include "mlir/IR/Dialect.h"

namespace mlir::verona
{
  namespace detail
  {
    struct IntegerTypeStorage;
  } // namespace detail

  namespace VeronaTypes
  {
    // Needs to be an enum (not an enum class) because 'kindof' methods compare
    // unsigned values and not class values.
    enum Kind
    {
      Integer = Type::Kind::FIRST_PRIVATE_EXPERIMENTAL_0_TYPE
    };
  } // namespace VeronaTypes

#include "dialect/VeronaOpsDialect.h.inc"

  struct IntegerType
  : public Type::TypeBase<IntegerType, Type, detail::IntegerTypeStorage>
  {
    using Base::Base;

    static IntegerType get(MLIRContext* context, size_t width, unsigned sign);

    size_t getWidth() const;
    bool getSign() const;

    static bool kindof(unsigned kind)
    {
      return kind == VeronaTypes::Integer;
    }
  };

} // namespace mlir::verona
