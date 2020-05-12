// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "VeronaDialect.h"

#include "VeronaOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/StandardTypes.h"

using namespace mlir;
using namespace mlir::verona;

//===----------------------------------------------------------------------===//
// Verona dialect.
//===----------------------------------------------------------------------===//

VeronaDialect::VeronaDialect(mlir::MLIRContext* context)
: Dialect(getDialectNamespace(), context)
{
  addOperations<
#define GET_OP_LIST
#include "dialect/VeronaOps.cpp.inc"
    >();
  addTypes<IntegerType>();
  allowUnknownOperations();
  allowUnknownTypes();
}

Type VeronaDialect::parseType(DialectAsmParser& parser) const
{
  StringRef keyword;
  if (parser.parseKeyword(&keyword))
    return Type();

  if (keyword.startswith("U") || keyword.startswith("S"))
  {
    size_t width = 0;
    if (keyword.substr(1).getAsInteger(10, width))
    {
      parser.emitError(parser.getNameLoc(), "unknown verona type: ") << keyword;
      return Type();
    }
    bool sign = keyword.startswith("S");
    return IntegerType::get(getContext(), width, sign);
  }

  parser.emitError(parser.getNameLoc(), "unknown verona type: ") << keyword;
  return Type();
}

void VeronaDialect::printType(Type type, DialectAsmPrinter& os) const
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
      return;
    }
    default:
      llvm_unreachable("unexpected 'verona' type kind");
  }
}

//===----------------------------------------------------------------------===//
// Verona types.
//===----------------------------------------------------------------------===//

namespace mlir::verona::detail
{
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
    : TypeStorage(), width(std::get<0>(key)), sign(std::get<1>(key))
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
} // namespace mlir::verona::detail

verona::IntegerType
verona::IntegerType::get(MLIRContext* context, size_t width, unsigned sign)
{
  return Base::get(context, VeronaTypes::Kind::Integer, width, sign);
}
size_t verona::IntegerType::getWidth() const
{
  return getImpl()->width;
}
bool verona::IntegerType::getSign() const
{
  return getImpl()->sign;
}
