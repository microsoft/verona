// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "dialect/VeronaTypes.h"

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

  Type parseVeronaType(DialectAsmParser& parser)
  {
    MLIRContext* ctx = parser.getBuilder().getContext();

    StringRef keyword;
    if (parser.parseKeyword(&keyword))
      return Type();

    if (keyword == "meet")
      return parseMeetType(ctx, parser);
    else if (keyword == "join")
      return parseJoinType(ctx, parser);
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
      .Case<JoinType>([&](Type type) {
        return normalizeJoin(ctx, type.cast<JoinType>().getElements());
      })
      .Case<MeetType>([&](Type type) {
        return normalizeMeet(ctx, type.cast<MeetType>().getElements());
      })
      .Default([&](Type type) { return type; });
  }
}
