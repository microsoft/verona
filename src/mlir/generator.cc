// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "generator.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Types.h"

#include <string>

namespace mlir::verona
{
  // ====== Helpers to interface consumers and transformers with the generator

  OpBuilder& MLIRGenerator::getBuilder()
  {
    return builder;
  }

  SymbolTableT& MLIRGenerator::getSymbolTable()
  {
    return symbolTable;
  }

  void MLIRGenerator::push_back(FuncOp func)
  {
    module->push_back(func);
  }

  OwningModuleRef MLIRGenerator::finish()
  {
    return std::move(module);
  }

  // ==================================== Generic helpers that manipulate MLIR

  bool MLIRGenerator::hasTerminator(mlir::Block* bb)
  {
    return !bb->getOperations().empty() &&
      bb->back().mightHaveTrait<mlir::OpTrait::IsTerminator>();
  }

  bool MLIRGenerator::isPointer(mlir::Value val)
  {
    return val && val.getType().isa<PointerType>();
  }

  mlir::Type MLIRGenerator::getElementType(mlir::Value val)
  {
    assert(isPointer(val) && "Bad type");
    return val.getType().dyn_cast<PointerType>().getElementType();
  }

  bool MLIRGenerator::isStructPointer(mlir::Value val)
  {
    return isPointer(val) && getElementType(val).isa<StructType>();
  }

  mlir::Type MLIRGenerator::getFieldType(StructType type, int offset)
  {
    auto field = type.getBody().begin();
    std::advance(field, offset);
    return PointerType::get(*field);
  }

  // ==================================================== Top level generators

  llvm::Expected<FuncOp> MLIRGenerator::Proto(
    Location loc,
    llvm::StringRef name,
    llvm::ArrayRef<Type> types,
    llvm::ArrayRef<Type> retTy)
  {
    // Create function
    auto funcTy = builder.getFunctionType(types, {retTy});
    auto func = FuncOp::create(loc, name, funcTy);
    // FIXME: This should be private unless we export, but for now we make
    // it public to test IR generation before implementing public visibility
    func.setVisibility(SymbolTable::Visibility::Public);
    return func;
  }

  llvm::Expected<FuncOp> MLIRGenerator::EmptyFunction(
    Location loc,
    llvm::StringRef name,
    llvm::ArrayRef<Type> types,
    llvm::ArrayRef<Type> retTy)
  {
    // If it's not declared yet, do so. This simplifies direct declaration
    // of compiler functions. User functions should be checked at the parse
    // level.
    auto func = module->lookupSymbol<FuncOp>(name);
    if (!func)
    {
      auto proto = Proto(loc, name, types, retTy);
      if (auto err = proto.takeError())
        return std::move(err);
      func = *proto;
    }

    // Create entry block, set builder entry point
    auto& entryBlock = *func.addEntryBlock();
    builder.setInsertionPointToStart(&entryBlock);

    return func;
  }

  llvm::Expected<Value> MLIRGenerator::Call(
    Location loc, FuncOp func, llvm::ArrayRef<Value> args)
  {
    // TODO: Implement dynamic method calls
    auto call = builder.create<CallOp>(loc, func, args);
    // TODO: Implement multiple return values (tuples?)
    return call->getOpResult(0);
  }

  llvm::Expected<Value> MLIRGenerator::Arithmetic(
    Location loc, llvm::StringRef opName, Value lhs, Value rhs)
  {
    // FIXME: Implement all unary and binary operators
    assert(lhs && rhs && "No binary operation with less than two arguments");

    // Make sure we're dealing with values, not pointers
    // FIXME: This shouldn't be necessary at this point
    if (isPointer(lhs))
      lhs = Load(loc, lhs);
    if (isPointer(rhs))
      rhs = Load(loc, rhs);

    // Promote types to be the same, or ops don't work, in the end, both
    // types are identical and the same as the return type.
    std::tie(lhs, rhs) = Promote(lhs, rhs);
    auto retTy = lhs.getType();

    // FIXME: We already converted U32 to i32 so this "works". But we need
    // to make sure we want that conversion as early as it is, and if not,
    // we need to implement this as a standard select and convert that
    // later. However, that would only work if U32 has a method named "+",
    // or if we declare it on the fly and then clean up when we remove the
    // call.

    // Floating point arithmetic
    if (retTy.isF32() || retTy.isF64())
    {
      auto op = llvm::StringSwitch<Value>(opName)
                  .Case("+", builder.create<AddFOp>(loc, retTy, lhs, rhs))
                  .Default({});
      assert(op && "Unknown arithmetic operator");
      return op;
    }

    // Integer arithmetic
    assert(retTy.isa<IntegerType>() && "Bad arithmetic types");
    auto op = llvm::StringSwitch<Value>(opName)
                .Case("+", builder.create<AddIOp>(loc, retTy, lhs, rhs))
                .Default({});
    assert(op && "Unknown arithmetic operator");
    return op;
  }

  // ==================================================== Low level generators

  Value MLIRGenerator::Convert(Value val, Type ty)
  {
    auto valTy = val.getType();
    auto valSize = valTy.getIntOrFloatBitWidth();
    auto tySize = ty.getIntOrFloatBitWidth();
    if (valSize == tySize)
      return val;

    // Integer upcasts
    // TODO: Consiger sign, too
    auto valInt = valTy.dyn_cast<IntegerType>();
    auto tyInt = ty.dyn_cast<IntegerType>();
    if (valInt && tyInt)
    {
      if (valSize < tySize)
        return builder.create<SignExtendIOp>(val.getLoc(), ty, val);
      else
        return builder.create<TruncateIOp>(val.getLoc(), ty, val);
    }

    // Floating point casts
    auto valFP = valTy.dyn_cast<FloatType>();
    auto tyFP = ty.dyn_cast<FloatType>();
    if (valFP && tyFP)
    {
      if (valSize < tySize)
        return builder.create<FPExtOp>(val.getLoc(), ty, val);
      else
        return builder.create<FPTruncOp>(val.getLoc(), ty, val);
    }

    // If not compatible, assert
    assert(false && "Type casts between incompatible types");

    // Appease MSVC warnings
    return Value();
  }

  std::pair<mlir::Value, mlir::Value>
  MLIRGenerator::Promote(mlir::Value lhs, mlir::Value rhs)
  {
    auto lhsType = lhs.getType();
    auto rhsType = rhs.getType();

    // Shortcut for when both are the same
    if (lhsType == rhsType)
      return {lhs, rhs};

    auto lhsSize = lhsType.getIntOrFloatBitWidth();
    auto rhsSize = rhsType.getIntOrFloatBitWidth();

    // Promote the smallest to the largest
    if (lhsSize < rhsSize)
      lhs = Convert(lhs, rhsType);
    else
      rhs = Convert(rhs, lhsType);

    return {lhs, rhs};
  }

  Value MLIRGenerator::Alloca(Location loc, Type ty)
  {
    PointerType pointerTy;
    Value len = Constant(builder.getI32Type(), 1);
    pointerTy = PointerType::get(ty);
    return builder.create<LLVM::AllocaOp>(loc, pointerTy, len);
  }

  Value MLIRGenerator::GEP(Location loc, Value addr, int offset)
  {
    llvm::SmallVector<Value> offsetList;
    // First argument is always in context of a list
    if (isStructPointer(addr))
    {
      auto zero = Zero(builder.getI32Type());
      offsetList.push_back(zero);
    }
    // Second argument is in context of the struct
    auto len = Constant(builder.getI32Type(), offset);
    offsetList.push_back(len);
    ValueRange index(offsetList);
    Type retTy = addr.getType();
    if (auto structTy = getElementType(addr).dyn_cast<StructType>())
      retTy = getFieldType(structTy, offset);
    return builder.create<LLVM::GEPOp>(loc, retTy, addr, index);
  }

  Value MLIRGenerator::Load(Location loc, Value addr, int offset)
  {
    if (!isa<LLVM::GEPOp>(addr.getDefiningOp()))
      addr = GEP(loc, addr, offset);
    else
      assert(offset == 0 && "Can't take an offset of a GEP");
    return builder.create<LLVM::LoadOp>(loc, addr);
  }

  Value
  MLIRGenerator::AutoLoad(Location loc, Value addr, Type ty, int offset)
  {
    // If it's not an address, there's nothing to load
    if (!isPointer(addr))
      return addr;

    // If the expected type is a pointer, we want the address, not the value
    if (ty && ty.isa<PointerType>())
      return addr;

    auto elmTy = getElementType(addr);

    // If type was specified, check it matches the address type
    if (ty)
      assert(elmTy == ty && "Invalid pointer load");

    return Load(loc, addr, offset);
  }

  void
  MLIRGenerator::Store(Location loc, Value addr, Value val, int offset)
  {
    if (!isa<LLVM::GEPOp>(addr.getDefiningOp()))
      addr = GEP(loc, addr, offset);
    else
      assert(offset == 0 && "Can't take an offset of a GEP");
    builder.create<LLVM::StoreOp>(loc, val, addr);
  }

  std::string
  MLIRGenerator::mangleConstantName(Type ty, std::variant<int, double> val)
  {
    // This is similar to MLIR's own naming, but with two leading underscores,
    // to make it easier to find in a debugger
    std::string name = "__c";
    if (ty.isa<FloatType>())
    {
      name += std::to_string(std::get<double>(val));
      name += "_f";
      name += std::to_string(ty.getIntOrFloatBitWidth());
    }
    else if (ty.isa<IntegerType>())
    {
      name += std::to_string(std::get<int>(val));
      name += "_i";
      name += std::to_string(ty.getIntOrFloatBitWidth());
    }
    else if (ty.isa<IndexType>())
    {
      name += std::to_string(std::get<int>(val));
    }

    return name;
  }

  Value MLIRGenerator::Constant(Type ty, std::variant<int, double> val)
  {
    // Use symbol table with mangled name to avoid duplication
    auto name = mangleConstantName(ty, val);
    auto value = symbolTable.lookup(name);
    if (value)
      return value;

    auto loc = builder.getUnknownLoc();
    if (ty.isIndex())
    {
      value = builder.create<ConstantIndexOp>(loc, std::get<int>(val));
    }
    else if (auto it = ty.dyn_cast<IntegerType>())
    {
      value = builder.create<ConstantIntOp>(loc, std::get<int>(val), it);
    }
    else if (auto ft = ty.dyn_cast<FloatType>())
    {
      APFloat floatValue = APFloat(std::get<double>(val));
      value = builder.create<ConstantFloatOp>(loc, floatValue, ft);
    }

    assert(value && "Type not supported for constant");

    symbolTable.insert(name, value);
    return value;
  }

  Value MLIRGenerator::Zero(Type ty)
  {
    if (ty.isa<FloatType>())
      return Constant(ty, 0.0);
    else
      return Constant(ty, 0);
  }

  Value MLIRGenerator::ConstantString(StringRef str, StringRef name)
  {
    // Use auto-generated name if none provided
    static size_t incr = 0;
    std::string nameStr;
    if (name.empty())
      nameStr = "_string" + std::to_string(incr++);
    else
      nameStr = name.str();

    // In LLVM, strings are arrays of i8 elements
    auto i8 = builder.getIntegerType(8);
    auto strTy = ArrayType::get(i8, str.size());
    auto strAttr = builder.getStringAttr(str);

    // In LLVM, constant strings are global objects
    auto moduleBuilder = OpBuilder(*module);
    auto global = moduleBuilder.create<LLVM::GlobalOp>(
      builder.getUnknownLoc(),
      strTy,
      /*isConstant=*/true,
      LLVM::Linkage::Private,
      nameStr,
      strAttr);
    module->push_back(global);

    // But their addresses are a local operation
    return builder.create<LLVM::AddressOfOp>(builder.getUnknownLoc(), global);
  }
}
