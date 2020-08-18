// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include "mlir/IR/Dialect.h"

namespace mlir::verona
{
  Type parseVeronaType(DialectAsmParser& parser);
  void printVeronaType(Type type, DialectAsmPrinter& os);

  /// Returns true if the type is one defined by the Verona dialect.
  bool isaVeronaType(Type type);
  /// Returns true if all types in the array are ones defined by the Verona
  /// dialect.
  bool areVeronaTypes(llvm::ArrayRef<Type> types);

  /// Normalize a type by distributing unions and intersections, putting the
  /// type in disjunctive normal form. This is a necessary step in order for
  /// subtyping to recognise certain relations.
  ///
  /// TODO: normalizing types is a potentially expensive operation, so we should
  /// try to cache the results.
  Type normalizeType(Type type);

  namespace detail
  {
    struct MeetTypeStorage;
    struct JoinTypeStorage;
    struct IntegerTypeStorage;
    struct CapabilityTypeStorage;
  }

  // In the long term we should claim a range in LLVM's DialectSymbolRegistry,
  // rather than use the "experimental" range.
  static constexpr unsigned FIRST_VERONA_TYPE =
    Type::Kind::FIRST_PRIVATE_EXPERIMENTAL_0_TYPE;
  static constexpr unsigned LAST_VERONA_TYPE =
    Type::Kind::LAST_PRIVATE_EXPERIMENTAL_0_TYPE;

  namespace VeronaTypes
  {
    enum Kind
    {
      Meet = FIRST_VERONA_TYPE,
      Join,
      Integer,
      Capability,
    };
  }

  struct MeetType
  : public Type::TypeBase<MeetType, Type, detail::MeetTypeStorage>
  {
    using Base::Base;
    static MeetType
    get(MLIRContext* ctx, llvm::ArrayRef<mlir::Type> elementTypes);
    llvm::ArrayRef<mlir::Type> getElements() const;

    static bool kindof(unsigned kind)
    {
      return kind == VeronaTypes::Meet;
    }
  };

  struct JoinType
  : public Type::TypeBase<JoinType, Type, detail::JoinTypeStorage>
  {
    using Base::Base;
    static JoinType
    get(MLIRContext* ctx, llvm::ArrayRef<mlir::Type> elementTypes);
    llvm::ArrayRef<mlir::Type> getElements() const;

    static bool kindof(unsigned kind)
    {
      return kind == VeronaTypes::Join;
    }
  };

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

  enum class Capability
  {
    Isolated,
    Mutable,
    Immutable,
  };

  struct CapabilityType
  : public Type::TypeBase<CapabilityType, Type, detail::CapabilityTypeStorage>
  {
    using Base::Base;
    static CapabilityType get(MLIRContext* ctx, Capability cap);
    Capability getCapability() const;

    static bool kindof(unsigned kind)
    {
      return kind == VeronaTypes::Capability;
    }
  };
}
