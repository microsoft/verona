// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "dialect/VeronaTypes.h"

#include "dialect/VeronaOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

#include "llvm/ADT/TypeSwitch.h"

namespace mlir::verona::detail
{
  struct ClassTypeStorage : public ::mlir::TypeStorage
  {
    using FieldsRef = ClassType::FieldsRef;
    using KeyTy = StringRef;

    // Only the class name is used to unique the type. The `isInitialized` flag
    // and `fields` array are part of the type's mutable component.
    StringRef name;
    bool isInitialized;
    FieldsRef fields;

    ClassTypeStorage(StringRef name)
    : name(name), isInitialized(false), fields()
    {}

    static llvm::hash_code hashKey(const KeyTy& key)
    {
      return llvm::hash_value(key);
    }

    bool operator==(const KeyTy& key) const
    {
      return key == name;
    }

    static ClassTypeStorage*
    construct(TypeStorageAllocator& allocator, const KeyTy& key)
    {
      StringRef name = allocator.copyInto(key);
      return new (allocator.allocate<ClassTypeStorage>())
        ClassTypeStorage(name);
    }

    LogicalResult mutate(TypeStorageAllocator& allocator, FieldsRef new_fields)
    {
      if (isInitialized)
      {
        return new_fields == this->fields ? success() : failure();
      }

      // We construct a temporary array, in which the field names have been
      // copied into `allocator`. Later this array itself will be copied into
      // `allocator`.
      //
      // TODO: this could be made more efficient if TypeStorageAllocator could
      // hand us an uninitialized (or default-initialized) mutable array, which
      // we could later fill with the copied field names.
      SmallVector<std::pair<StringRef, Type>, 4> temp_fields;
      temp_fields.reserve(new_fields.size());
      for (auto [field_name, field_type] : new_fields)
      {
        temp_fields.push_back({allocator.copyInto(field_name), field_type});
      }
      this->fields = allocator.copyInto(FieldsRef(temp_fields));

      isInitialized = true;
      return success();
    }
  };
} // namespace mlir::verona::detail

namespace mlir::verona
{
  ClassType ClassType::get(MLIRContext* ctx, StringRef name)
  {
    return Base::get(ctx, name);
  }

  ClassType ClassType::get(MLIRContext* ctx, StringRef name, FieldsRef fields)
  {
    ClassType type = ClassType::get(ctx, name);
    if (succeeded(type.setFields(fields)))
      return type;
    else
      return ClassType();
  }

  LogicalResult
  ClassType::setFields(ArrayRef<std::pair<StringRef, Type>> fields)
  {
    return Base::mutate(fields);
  }

  StringRef ClassType::getName() const
  {
    return getImpl()->name;
  }

  ClassType::FieldsRef ClassType::getFields() const
  {
    // We may not have a full declaration available
    // TODO: Make this an assert when we have modules
    if (getImpl()->isInitialized)
      return getImpl()->fields;
    else
      return ClassType::FieldsRef();
  }

  Type ClassType::getFieldType(StringRef name) const
  {
    for (auto it : getFields())
    {
      if (it.first == name)
        return it.second;
    }
    return nullptr;
  }

  LogicalResult
  JoinType::verifyConstructionInvariants(Location loc, ArrayRef<Type> elements)
  {
    if (!areVeronaTypes(elements))
    {
      return emitError(loc, "Elements of join aren't Verona types: ")
        << elements;
    }

    return success();
  }

  LogicalResult
  MeetType::verifyConstructionInvariants(Location loc, ArrayRef<Type> elements)
  {
    if (!areVeronaTypes(elements))
    {
      return emitError(loc, "Elements of meet aren't Verona types: ")
        << elements;
    }

    return success();
  }

  LogicalResult
  DescriptorType::verifyConstructionInvariants(Location loc, Type describedType)
  {
    if (!isaVeronaType(describedType))
    {
      return emitError(
               loc, "Underlying type of descriptor isn't a Verona type: ")
        << describedType;
    }

    return success();
  }

  LogicalResult ViewpointType::verifyConstructionInvariants(
    Location loc, Type leftType, Type rightType)
  {
    if (!isaVeronaType(leftType))
    {
      return emitError(
               loc, "Left hand side of a viewpoint isn't a Verona type: ")
        << leftType;
    }

    if (!isaVeronaType(rightType))
    {
      return emitError(
               loc, "Right hand side of viewpoint isn't a Verona type: ")
        << rightType;
    }

    return success();
  }

  bool isaVeronaType(Type type)
  {
    return type.isa<
      MeetType,
      JoinType,
      UnknownType,
      DescriptorType,
      CapabilityType,
      ClassType,
      ViewpointType>();
  }

  bool areVeronaTypes(llvm::ArrayRef<Type> types)
  {
    return llvm::all_of(types, isaVeronaType);
  }

  /// Distribute a lattice type (join or meet) by applying `f` to every element
  /// of it. Each return value of the continuation is added to `result`.
  ///
  /// Assuming `type` is in normal form, this method will process nested `T`s
  /// as well.
  ///
  /// For example, given `join<A, join<B, C>>`, this method will add
  /// `f(A), f(B), f(C)` to `result`.
  template<typename T>
  static void distributeType(
    SmallVectorImpl<Type>& result, T type, llvm::function_ref<Type(Type)> f)
  {
    for (Type element : type.getElements())
    {
      if (auto nested = element.dyn_cast<T>())
        distributeType<T>(result, nested, f);
      else
        result.push_back(f(element));
    }
  }

  /// If the argument `type` is of kind `T` (where `T` is a lattice type, ie.
  /// JoinType or MeetType), distribute it by applying `f` to every element of
  /// it. The return values are combined to form a new lattice type of the same
  /// kind. If `type` is not of kind `T`, it is directly applied to `f`.
  ///
  /// Assuming `type` is in normal form, this method will process nested `T`s
  /// as well.
  ///
  /// For example, given `join<A, join<B, C>>`, this method will return
  /// `join<f(A), f(B), f(C)>`.
  template<typename T>
  static Type
  distributeType(MLIRContext* ctx, Type type, llvm::function_ref<Type(Type)> f)
  {
    if (auto node = type.dyn_cast<T>())
    {
      SmallVector<Type, 4> result;
      distributeType<T>(result, node, f);
      return T::get(ctx, result);
    }
    else
    {
      return f(type);
    }
  }

  /// Distribute all join and meets found in `type`, by applying `f` to every
  /// "atom" in the type. `type` is assumed to be in normal form already.
  ///
  /// For example, given `join<meet<A, B>, C>`, this function returns
  /// `join<meet<f(A), f(B)>, f(C)>`.
  static Type
  distributeAll(MLIRContext* ctx, Type type, llvm::function_ref<Type(Type)> f)
  {
    return distributeType<JoinType>(ctx, type, [&](Type inner) {
      return distributeType<MeetType>(ctx, inner, f);
    });
  }

  /// Normalize a meet type.
  /// This function returns the normal form of `meet<normalized..., rest...>`,
  /// distributing any nested joins.
  ///
  /// Types in `normalized` must be in normal form and not contain any joins.
  /// Types in `rest` may be in any form.
  ///
  /// This method uses `normalized` as scratch space; it recurses with more
  /// elements pushed to it. When it returns, `normalized` will always have its
  /// original length and contents.
  ///
  /// TODO: this function uses recursion to iterate over the `rest` array,
  /// because that works well with normalizeType. It could be rewritten to use
  /// loops, which is probably more efficient and doesn't risk blowing the
  /// stack.
  Type normalizeMeet(
    MLIRContext* ctx, SmallVectorImpl<Type>& normalized, ArrayRef<Type> rest)
  {
    if (rest.empty())
      return MeetType::get(ctx, normalized);

    Type element = normalizeType(rest.front());
    return distributeType<JoinType>(ctx, element, [&](auto inner) {
      normalized.push_back(inner);
      auto result = normalizeMeet(ctx, normalized, rest.drop_front());
      normalized.pop_back();
      return result;
    });
  }

  /// Normalize a meet type.
  /// This function returns the normal form of `meet<elements...>`,
  /// distributing any nested joins.
  Type normalizeMeet(MLIRContext* ctx, ArrayRef<Type> elements)
  {
    SmallVector<Type, 4> result;
    return normalizeMeet(ctx, result, elements);
  }

  /// Normalize a join type.
  /// This function returns the normal form of `join<elements...>`. The only
  /// effect of this is individually normalizing the contents of `elements`.
  Type normalizeJoin(MLIRContext* ctx, ArrayRef<Type> elements)
  {
    SmallVector<Type, 4> result;
    llvm::transform(elements, std::back_inserter(result), [&](Type element) {
      return normalizeType(element);
    });
    return JoinType::get(ctx, result);
  }

  /// Normalize a viewpoint type, distributing any join or meet found in either
  /// own its constituent types.
  Type normalizeViewpoint(MLIRContext* ctx, Type left, Type right)
  {
    Type normalizedLeft = normalizeType(left);
    Type normalizedRight = normalizeType(right);

    return distributeAll(ctx, normalizedLeft, [&](Type distributedLeft) {
      return distributeAll(ctx, normalizedRight, [&](Type distributedRight) {
        return ViewpointType::get(ctx, distributedLeft, distributedRight);
      });
    });
  }

  // TODO: The amount of normalization done is quite limited. In particular it
  // does not always flatten types (eg. changing `join<A, join<B, C>>` into
  // `join<A, B, C>`), nor does it do any simplification (eg. `join<A, A, B>`
  // into `join<A, B>`). These normalizations aren't necessary for subtyping,
  // but could help with other places in the compiler.
  Type normalizeType(Type type)
  {
    MLIRContext* ctx = type.getContext();
    assert(isaVeronaType(type));
    return TypeSwitch<Type, Type>(type)
      .Case<JoinType>(
        [&](JoinType type) { return normalizeJoin(ctx, type.getElements()); })
      .Case<MeetType>(
        [&](MeetType type) { return normalizeMeet(ctx, type.getElements()); })
      .Case<ViewpointType>([&](ViewpointType type) {
        return normalizeViewpoint(ctx, type.getLeftType(), type.getRightType());
      })
      .Default([&](Type type) { return type; });
  }

  /// Look up a field's type from a meet type.
  ///
  /// The field must be present in at least one element of the meet.
  static std::pair<Type, Type>
  lookupMeetFieldType(MeetType meetType, StringRef name)
  {
    MLIRContext* ctx = meetType.getContext();

    bool found;
    SmallVector<Type, 4> readElements;
    SmallVector<Type, 4> writeElements;

    for (Type origin : meetType.getElements())
    {
      auto [readType, writeType] = lookupFieldType(origin, name);
      assert((readType == nullptr) == (writeType == nullptr));

      if (readType != nullptr)
      {
        readElements.push_back(readType);
        writeElements.push_back(writeType);
        found = true;
      }
    }

    if (found)
      return {MeetType::get(ctx, readElements),
              JoinType::get(ctx, writeElements)};
    else
      return {nullptr, nullptr};
  }

  /// Look up a field's type from a join type.
  ///
  /// The field must be present in all the elements of the meet. This extends to
  /// empty joins: looking up a field will always succeed.
  static std::pair<Type, Type>
  lookupJoinFieldType(JoinType joinType, StringRef name)
  {
    MLIRContext* ctx = joinType.getContext();
    SmallVector<Type, 4> readElements;
    SmallVector<Type, 4> writeElements;

    for (Type origin : joinType.getElements())
    {
      auto [readType, writeType] = lookupFieldType(origin, name);
      assert((readType == nullptr) == (writeType == nullptr));

      if (readType == nullptr)
        return {nullptr, nullptr};
    }

    return {JoinType::get(ctx, readElements),
            MeetType::get(ctx, writeElements)};
  }

  std::pair<Type, Type> lookupFieldType(Type origin, StringRef name)
  {
    return TypeSwitch<Type, std::pair<Type, Type>>(origin)
      .Case<MeetType>(
        [&](MeetType origin) { return lookupMeetFieldType(origin, name); })
      .Case<JoinType>(
        [&](JoinType origin) { return lookupJoinFieldType(origin, name); })
      .Case<ClassType>([&](ClassType origin) -> std::pair<Type, Type> {
        Type field = origin.getFieldType(name);
        return {field, field};
      })
      .Case<ViewpointType>([&](ViewpointType origin) {
        return lookupFieldType(origin.getRightType(), name);
      })
      .Default([](Type origin) -> std::pair<Type, Type> {
        return {nullptr, nullptr};
      });
  }
}

#define GET_TYPEDEF_CLASSES
#include "dialect/VeronaTypes.cpp.inc"
