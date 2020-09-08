// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "dialect/VeronaTypes.h"

#include "dialect/VeronaOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/StandardTypes.h"

#include "llvm/ADT/TypeSwitch.h"

namespace mlir::verona::detail
{
  struct MeetTypeStorage : public TypeStorage
  {
    using KeyTy = llvm::ArrayRef<Type>;

    llvm::ArrayRef<Type> elements;

    MeetTypeStorage(llvm::ArrayRef<Type> elements) : elements(elements) {}

    bool operator==(const KeyTy& key) const
    {
      return key == elements;
    }

    static MeetTypeStorage*
    construct(TypeStorageAllocator& allocator, const KeyTy& key)
    {
      llvm::ArrayRef<mlir::Type> elements = allocator.copyInto(key);
      return new (allocator.allocate<MeetTypeStorage>())
        MeetTypeStorage(elements);
    }
  };

  struct JoinTypeStorage : public TypeStorage
  {
    using KeyTy = llvm::ArrayRef<Type>;

    llvm::ArrayRef<Type> elements;

    JoinTypeStorage(llvm::ArrayRef<Type> elements) : elements(elements) {}

    bool operator==(const KeyTy& key) const
    {
      return key == elements;
    }

    static JoinTypeStorage*
    construct(TypeStorageAllocator& allocator, const KeyTy& key)
    {
      llvm::ArrayRef<mlir::Type> elements = allocator.copyInto(key);
      return new (allocator.allocate<JoinTypeStorage>())
        JoinTypeStorage(elements);
    }
  };

  struct IntegerTypeStorage : public ::mlir::TypeStorage
  {
    uint8_t width;
    enum SignType
    {
      Unknown,
      Unsigned,
      Signed
    };
    unsigned sign;

    // width, sign
    using KeyTy = std::tuple<size_t, unsigned>;
    IntegerTypeStorage(const KeyTy& key)
    : width(std::get<0>(key)), sign(std::get<1>(key))
    {}

    bool operator==(const KeyTy& key) const
    {
      return key == KeyTy(width, sign);
    }

    static IntegerTypeStorage*
    construct(TypeStorageAllocator& allocator, const KeyTy& key)
    {
      return new (allocator.allocate<IntegerTypeStorage>())
        IntegerTypeStorage(key);
    }
  };

  struct CapabilityTypeStorage : public ::mlir::TypeStorage
  {
    Capability capability;

    using KeyTy = Capability;

    CapabilityTypeStorage(const KeyTy& key) : capability(key) {}

    static llvm::hash_code hashKey(const KeyTy& key)
    {
      return llvm::hash_value(key);
    }

    bool operator==(const KeyTy& key) const
    {
      return key == capability;
    }

    static CapabilityTypeStorage*
    construct(TypeStorageAllocator& allocator, const KeyTy& key)
    {
      return new (allocator.allocate<CapabilityTypeStorage>())
        CapabilityTypeStorage(key);
    }
  };

  struct ClassTypeStorage : public ::mlir::TypeStorage
  {
    using FieldsRef = ClassType::FieldsRef;
    using KeyTy = StringRef;

    // Only the class name is used to unique the type. The `isInitialized` flag
    // and `fields` array are part of the type's mutable component.
    StringRef class_name;
    bool isInitialized;
    FieldsRef fields;

    ClassTypeStorage(StringRef class_name)
    : class_name(class_name), isInitialized(false), fields()
    {}

    static llvm::hash_code hashKey(const KeyTy& key)
    {
      return llvm::hash_value(key);
    }

    bool operator==(const KeyTy& key) const
    {
      return key == class_name;
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

  struct ViewpointTypeStorage : public ::mlir::TypeStorage
  {
    Type left;
    Type right;

    using KeyTy = std::tuple<Type, Type>;

    ViewpointTypeStorage(Type left, Type right) : left(left), right(right) {}

    static llvm::hash_code hashKey(const KeyTy& key)
    {
      return llvm::hash_value(key);
    }

    bool operator==(const KeyTy& key) const
    {
      return key == std::tie(left, right);
    }

    static ViewpointTypeStorage*
    construct(TypeStorageAllocator& allocator, const KeyTy& key)
    {
      return new (allocator.allocate<ViewpointTypeStorage>())
        ViewpointTypeStorage(std::get<0>(key), std::get<1>(key));
    }
  };
} // namespace mlir::verona::detail

namespace mlir::verona
{
  MeetType MeetType::get(MLIRContext* ctx, llvm::ArrayRef<mlir::Type> elements)
  {
    assert(areVeronaTypes(elements));
    return Base::get(ctx, elements);
  }

  llvm::ArrayRef<mlir::Type> MeetType::getElements() const
  {
    return getImpl()->elements;
  }

  JoinType JoinType::get(MLIRContext* ctx, llvm::ArrayRef<mlir::Type> elements)
  {
    assert(areVeronaTypes(elements));
    return Base::get(ctx, elements);
  }

  llvm::ArrayRef<mlir::Type> JoinType::getElements() const
  {
    return getImpl()->elements;
  }

  IntegerType IntegerType::get(MLIRContext* ctx, size_t width, unsigned sign)
  {
    return Base::get(ctx, width, sign);
  }

  size_t IntegerType::getWidth() const
  {
    return getImpl()->width;
  }

  bool IntegerType::getSign() const
  {
    return getImpl()->sign;
  }

  CapabilityType CapabilityType::get(MLIRContext* ctx, Capability cap)
  {
    return Base::get(ctx, cap);
  }

  Capability CapabilityType::getCapability() const
  {
    return getImpl()->capability;
  }

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
    return getImpl()->class_name;
  }

  ClassType::FieldsRef ClassType::getFields() const
  {
    assert(getImpl()->isInitialized);
    return getImpl()->fields;
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

  ViewpointType ViewpointType::get(MLIRContext* ctx, Type left, Type right)
  {
    return Base::get(ctx, std::make_tuple(left, right));
  }

  Type ViewpointType::getLeftType() const
  {
    return getImpl()->left;
  }

  Type ViewpointType::getRightType() const
  {
    return getImpl()->right;
  }

  /// Parse a list of types, surrounded by angle brackets and separated by
  /// commas. The types inside the list must be Verona types and should not use
  /// the `!verona.` prefix.
  ///
  /// Empty lists are allowed, but must still use angle brackets, i.e. `< >`.
  /// Lists of one elements are also allowed.
  static ParseResult
  parseTypeList(DialectAsmParser& parser, llvm::SmallVectorImpl<Type>& result)
  {
    if (parser.parseLess())
      return failure();

    if (succeeded(parser.parseOptionalGreater()))
      return success();

    do
    {
      mlir::Type element = parseVeronaType(parser);
      if (!element)
        return failure();

      result.push_back(element);
    } while (succeeded(parser.parseOptionalComma()));

    if (parser.parseGreater())
      return failure();

    return success();
  }

  static Type parseMeetType(MLIRContext* ctx, DialectAsmParser& parser)
  {
    SmallVector<mlir::Type, 2> elements;
    if (parseTypeList(parser, elements))
      return Type();
    return MeetType::get(ctx, elements);
  }

  static Type parseJoinType(MLIRContext* ctx, DialectAsmParser& parser)
  {
    SmallVector<mlir::Type, 2> elements;
    if (parseTypeList(parser, elements))
      return Type();
    return JoinType::get(ctx, elements);
  }

  static Type parseIntegerType(
    MLIRContext* ctx, DialectAsmParser& parser, StringRef keyword)
  {
    size_t width = 0;
    if (keyword.substr(1).getAsInteger(10, width))
    {
      parser.emitError(parser.getNameLoc(), "unknown verona type: ") << keyword;
      return Type();
    }
    bool sign = keyword.startswith("S");
    return IntegerType::get(ctx, width, sign);
  }

  // Annoyingly, DialectAsmParser only exposes `parseOptionalString`, no
  // `parseString`. This method implements the latter based on the former.
  ParseResult parseString(DialectAsmParser& parser, StringRef* value)
  {
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseOptionalString(value)))
      return parser.emitError(loc, "expected string literal");
    else
      return success();
  }

  static Type parseClassType(MLIRContext* ctx, DialectAsmParser& parser)
  {
    auto loc = parser.getNameLoc();

    StringRef name;
    SmallVector<std::pair<StringRef, Type>, 4> fields;

    if (parser.parseLess() || parseString(parser, &name))
      return Type();

    // TODO: Support parsing recursive types by constructing an uninitialized
    // ClassType first and filling it up later. This would require passing
    // around the set of "pending" classes around.
    // See the LLVM dialect's implementation of struct types for an example.
    while (succeeded(parser.parseOptionalComma()))
    {
      StringRef field_name;
      Type field_type;
      if (
        parseString(parser, &field_name) || parser.parseColon() ||
        !(field_type = parseVeronaType(parser)))
        return Type();

      fields.push_back({field_name, field_type});
    }

    if (parser.parseGreater())
      return Type();

    ClassType pending = ClassType::get(ctx, name);
    if (failed(pending.setFields(fields)))
    {
      InFlightDiagnostic diag = parser.emitError(loc)
        << "class type \"" << pending.getName()
        << "\" already used with different definition";
      return Type();
    }

    return pending;
  }

  static Type parseViewpointType(MLIRContext* ctx, DialectAsmParser& parser)
  {
    Type left;
    Type right;
    if (
      parser.parseLess() || !(left = parseVeronaType(parser)) ||
      parser.parseComma() || !(right = parseVeronaType(parser)) ||
      parser.parseGreater())
      return Type();

    return ViewpointType::get(ctx, left, right);
  }

  Type parseVeronaType(DialectAsmParser& parser)
  {
    MLIRContext* ctx = parser.getBuilder().getContext();

    StringRef keyword;
    if (parser.parseKeyword(&keyword))
      return Type();

    if (keyword == "class")
      return parseClassType(ctx, parser);
    else if (keyword == "meet")
      return parseMeetType(ctx, parser);
    else if (keyword == "join")
      return parseJoinType(ctx, parser);
    else if (keyword == "viewpoint")
      return parseViewpointType(ctx, parser);
    else if (keyword == "top")
      return MeetType::get(ctx, {});
    else if (keyword == "bottom")
      return JoinType::get(ctx, {});
    else if (keyword == "iso")
      return CapabilityType::get(ctx, Capability::Isolated);
    else if (keyword == "mut")
      return CapabilityType::get(ctx, Capability::Mutable);
    else if (keyword == "imm")
      return CapabilityType::get(ctx, Capability::Immutable);
    else if (keyword.startswith("U") || keyword.startswith("S"))
      return parseIntegerType(ctx, parser, keyword);

    parser.emitError(parser.getNameLoc(), "unknown verona type: ") << keyword;
    return Type();
  }

  static void printTypeList(ArrayRef<Type> types, DialectAsmPrinter& os)
  {
    os << "<";
    llvm::interleaveComma(
      types, os, [&](auto element) { printVeronaType(element, os); });
    os << ">";
  }

  void printClassType(ClassType type, DialectAsmPrinter& os)
  {
    // TODO: Support printing recursive types. If we're already in the process
    // of printing a class' body and reach the same class again, we should skip
    // its body. See the LLVM dialect's implementation of struct types for an
    // example.
    os << "class<\"" << type.getName() << "\"";
    for (auto [field_name, field_type] : type.getFields())
    {
      os << ", \"" << field_name << "\" : ";
      printVeronaType(field_type, os);
    }
    os << ">";
  }

  void printVeronaType(Type type, DialectAsmPrinter& os)
  {
    TypeSwitch<Type>(type)
      .Case<IntegerType>([&](IntegerType type) {
        if (type.getSign())
        {
          os << "S";
        }
        else
        {
          os << "U";
        }
        os << type.getWidth();
      })
      .Case<MeetType>([&](MeetType type) {
        if (type.getElements().empty())
        {
          os << "top";
        }
        else
        {
          os << "meet";
          printTypeList(type.getElements(), os);
        }
      })
      .Case<JoinType>([&](JoinType type) {
        if (type.getElements().empty())
        {
          os << "bottom";
        }
        else
        {
          os << "join";
          printTypeList(type.getElements(), os);
        }
      })
      .Case<CapabilityType>([&](CapabilityType type) {
        switch (type.getCapability())
        {
          case Capability::Isolated:
            os << "iso";
            break;
          case Capability::Mutable:
            os << "mut";
            break;
          case Capability::Immutable:
            os << "imm";
            break;
        }
      })
      .Case<ClassType>([&](ClassType type) { printClassType(type, os); })
      .Case<ViewpointType>([&](ViewpointType type) {
        os << "viewpoint<" << type.getLeftType() << ", " << type.getRightType()
           << ">";
      });
  }

  bool isaVeronaType(Type type)
  {
    return type.isa<MeetType, JoinType, IntegerType, CapabilityType>();
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
