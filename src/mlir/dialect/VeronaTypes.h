// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include "VeronaOps.h"
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

  /// Look up the type of a field in an `origin` type.
  ///
  /// The function returns a pair of types, used respectively to read and write
  /// to the field. For example, given classes C and D with fields of type T
  /// and U, reading the field from `C | D` yields a `T | U`, whereas a value of
  /// type `T & U` must be written to it.
  ///
  /// Both types will be null if the field cannot be found in the origin.
  std::pair<Type, Type> lookupFieldType(Type origin, StringRef name);

  namespace detail
  {
    struct MeetTypeStorage;
    struct JoinTypeStorage;
    struct IntegerTypeStorage;
    struct CapabilityTypeStorage;
    struct ClassTypeStorage;
  }

  struct MeetType
  : public Type::TypeBase<MeetType, Type, detail::MeetTypeStorage>
  {
    using Base::Base;
    static MeetType
    get(MLIRContext* ctx, llvm::ArrayRef<mlir::Type> elementTypes);
    llvm::ArrayRef<mlir::Type> getElements() const;
  };

  struct JoinType
  : public Type::TypeBase<JoinType, Type, detail::JoinTypeStorage>
  {
    using Base::Base;
    static JoinType
    get(MLIRContext* ctx, llvm::ArrayRef<mlir::Type> elementTypes);
    llvm::ArrayRef<mlir::Type> getElements() const;
  };

  struct IntegerType
  : public Type::TypeBase<IntegerType, Type, detail::IntegerTypeStorage>
  {
    using Base::Base;

    static IntegerType get(MLIRContext* context, size_t width, unsigned sign);

    size_t getWidth() const;
    bool getSign() const;
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
  };

  /**
   * A class is described both by its name and its list of fields. A class
   * named C with fields f and g would be represented as follows:
   *
   *   !verona.class<"C", "f": T, "g": U>
   *
   * Recursive classes are represented by omitting its body in the recursive
   * use. The example below shows a class D with a field to an instance of the
   * same class `D`.
   *
   *   !verona.class<"D", "f": class<"D">>
   *
   * Only the name is used to unique the type. This means you may not have two
   * type classes with different list of fields. This allows a two step
   * construction of class types, necessary to construct recursive classes.
   *
   * A ClassType is constructed by calling `ClassType::get(name)` followed by a
   * call to `setFields` to initialize the contents. In the case of recursive
   * classes, the result of the `get` call may be used to construct the field
   * types.
   *
   * For example, the type corresponding to the following class:
   *
   *   class A {
   *     f: A
   *   }
   *
   * would be constructed as follows:
   *
   *   ClassType a = ClassType::get(ctx, "A");
   *   FieldsRef fields = { { "f", a } };
   *   a.setFields(a);
   *
   * TODO: While recursive types can be constructed programmatically, they can
   * neither be parsed nor printed yet.
   */
  struct ClassType
  : public Type::TypeBase<ClassType, Type, detail::ClassTypeStorage>
  {
    using Base::Base;

    /// This is used to keep the list of (field name, field type) pairs.
    /// We want a container that preserves insertion order so we get
    /// deterministic behaviour and can round-trip the IR.
    ///
    /// TODO: An llvm::MapVector may be more suitable for this, so we have
    /// constant-time lookup of fields. However it isn't supported by MLIR's
    /// TypeStorageAllocator.
    using FieldsRef = ArrayRef<std::pair<StringRef, Type>>;

    /// Get a reference to the class with the given name. The returned type may
    /// not be fully initialized yet until setFields is called.
    static ClassType get(MLIRContext* ctx, StringRef name);

    /// Get a reference to the class with the given name, initializing it if
    /// necessary. Returns null if the class was already initialized with
    /// different contents.
    static ClassType get(MLIRContext* ctx, StringRef name, FieldsRef fields);

    /// Set the list of fields contained in this class.
    ///
    /// Returns a failure if the type has already been initialized but with
    /// different contents.
    LogicalResult setFields(FieldsRef fields);

    StringRef getName() const;
    FieldsRef getFields() const;
    Type getFieldType(StringRef name) const;
  };
}
