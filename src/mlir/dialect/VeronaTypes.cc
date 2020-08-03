#include "dialect/VeronaTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/StandardTypes.h"

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
    return Base::get(ctx, VeronaTypes::Meet, elements);
  }

  llvm::ArrayRef<mlir::Type> MeetType::getElements() const
  {
    return getImpl()->elements;
  }

  JoinType JoinType::get(MLIRContext* ctx, llvm::ArrayRef<mlir::Type> elements)
  {
    assert(areVeronaTypes(elements));
    return Base::get(ctx, VeronaTypes::Join, elements);
  }

  llvm::ArrayRef<mlir::Type> JoinType::getElements() const
  {
    return getImpl()->elements;
  }

  IntegerType IntegerType::get(MLIRContext* ctx, size_t width, unsigned sign)
  {
    return Base::get(ctx, VeronaTypes::Integer, width, sign);
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
    return Base::get(ctx, VeronaTypes::Capability, cap);
  }

  Capability CapabilityType::getCapability() const
  {
    return getImpl()->capability;
  }

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
    switch (type.getKind())
    {
      case VeronaTypes::Integer:
      {
        auto iTy = type.cast<IntegerType>();
        if (iTy.getSign())
        {
          os << "S";
        }
        else
        {
          os << "U";
        }
        os << iTy.getWidth();
        break;
      }

      case VeronaTypes::Meet:
      {
        auto meetType = type.cast<MeetType>();
        if (meetType.getElements().empty())
        {
          os << "top";
        }
        else
        {
          os << "meet";
          printTypeList(meetType.getElements(), os);
        }
        break;
      }

      case VeronaTypes::Join:
      {
        auto joinType = type.cast<JoinType>();
        if (joinType.getElements().empty())
        {
          os << "bottom";
        }
        else
        {
          os << "join";
          printTypeList(joinType.getElements(), os);
        }
        break;
      }

      case VeronaTypes::Capability:
      {
        auto capType = type.cast<CapabilityType>();
        switch (capType.getCapability())
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
        break;
      }
    }
  }

  bool isaVeronaType(Type type)
  {
    return type.getKind() >= FIRST_VERONA_TYPE &&
      type.getKind() < LAST_VERONA_TYPE;
  }

  bool areVeronaTypes(llvm::ArrayRef<Type> types)
  {
    return llvm::all_of(types, isaVeronaType);
  }
}
