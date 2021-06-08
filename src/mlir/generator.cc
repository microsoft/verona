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
  void MLIRGenerator::initializeArithmetic()
  {
    // FIXME: Using MLIR standard dialect comes with complications, since the
    // arithmetic in MLIR is mostly for ML and doens't have the semantics we
    // need for generic lowering. We should probably use the LLVM dialect + LLVM
    // intrinsic calls for the rest.

    // Floating-point arithmetic
    arithmetic.emplace("std.absf", 1);
    arithmetic.emplace("std.ceilf", 1);
    arithmetic.emplace("std.floorf", 1);
    arithmetic.emplace("std.negf", 1);

    arithmetic.emplace("std.addf", 2);
    arithmetic.emplace("std.subf", 2);
    arithmetic.emplace("std.mulf", 2);
    arithmetic.emplace("std.divf", 2);

    arithmetic.emplace("std.fmaf", 3);

    // Integer arithmetic
    arithmetic.emplace("std.addi", 2);
    arithmetic.emplace("std.subi", 2);
    arithmetic.emplace("std.muli", 2);
    arithmetic.emplace("std.divi_unsigned", 2);
    arithmetic.emplace("std.remi_unsigned", 2);
    arithmetic.emplace("std.divi_signed", 2);
    arithmetic.emplace("std.remi_signed", 2);
    arithmetic.emplace("std.ceildivi_signed", 2);
    arithmetic.emplace("std.floordivi_signed", 2);

    // Logical operators
    arithmetic.emplace("std.and", 2);
    arithmetic.emplace("std.or", 2);
    arithmetic.emplace("std.xor", 2);
    arithmetic.emplace("std.select", 2);
    arithmetic.emplace("std.shift_left", 2);
    arithmetic.emplace("std.shift_right_signed", 2);
    arithmetic.emplace("std.shift_right_unsigned", 2);

    // Conversions
    arithmetic.emplace("std.fpext", 1);
    arithmetic.emplace("std.fptrunc", 1);
    arithmetic.emplace("std.sexti", 1);
    arithmetic.emplace("std.zexti", 1);
    arithmetic.emplace("std.trunci", 1);

    // Casts
    arithmetic.emplace("std.index_cast", 1);
    arithmetic.emplace("std.fptosi", 1);
    arithmetic.emplace("std.fptoui", 1);
    arithmetic.emplace("std.sitofp", 1);
    arithmetic.emplace("std.uitofp", 1);

    // TODO: Add comparators
    // TODO: Add llvm intrinsics
  }

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

  bool MLIRGenerator::hasTerminator(Block* bb)
  {
    return !bb->getOperations().empty() &&
      bb->back().mightHaveTrait<OpTrait::IsTerminator>();
  }

  bool MLIRGenerator::isPointer(Value val)
  {
    return (val && val.getType().isa<PointerType>());
  }

  Type MLIRGenerator::getPointedType(mlir::Value val)
  {
    if (!val)
      return Type();
    auto pTy = val.getType().dyn_cast<PointerType>();
    if (!pTy)
      return Type();
    return pTy.getElementType();
  }

  bool MLIRGenerator::isStructPointer(Value val)
  {
    auto pTy = getPointedType(val);
    return (pTy && pTy.isa<StructType>());
  }

  StructType MLIRGenerator::getPointedStructType(Value val, bool anonymous)
  {
    auto pTy = getPointedType(val);
    if (!pTy)
      return StructType();

    // This is a pointer, but is it to a structure?
    auto sTy = pTy.dyn_cast<StructType>();
    if (!sTy)
      return StructType();

    // This is a pointer to a structure, but is it anonymous?
    if (anonymous && sTy.isIdentified())
      return StructType();

    return sTy;
  }

  Type MLIRGenerator::getFieldType(StructType type, int offset)
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

  llvm::Expected<Value>
  MLIRGenerator::Call(Location loc, FuncOp func, llvm::ArrayRef<Value> args)
  {
    // TODO: Implement dynamic method calls
    auto call = builder.create<CallOp>(loc, func, args);
    // TODO: Implement multiple return values (tuples?)
    return call->getOpResult(0);
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

  std::pair<Value, mlir::Value>
  MLIRGenerator::Promote(Value lhs, mlir::Value rhs)
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

  Value MLIRGenerator::GEP(Location loc, Value addr, std::optional<int> offset)
  {
    llvm::SmallVector<Value> offsetList;
    bool extractElementFromStruct = false;

    // First argument is always in context of a list, so if there is no value
    // and this is a struct pointer, get the "first" struct, which is the only
    // one. See: https://www.llvm.org/docs/GetElementPtr.html
    if (isStructPointer(addr) && offset.has_value())
    {
      auto zero = Zero(builder.getI32Type());
      offsetList.push_back(zero);
      extractElementFromStruct = true;
    }

    // Default offset is zero
    auto offsetValue = offset.value_or(0);

    // Second argument is in context of the struct
    auto len = Constant(builder.getI32Type(), offsetValue);
    offsetList.push_back(len);
    ValueRange index(offsetList);
    Type retTy = addr.getType();

    // If the offset really was zero (not unset), and the address has a struct
    // type, we need to extract the element form it
    if (extractElementFromStruct)
    {
      auto structTy = getPointedType(addr).dyn_cast<StructType>();
      retTy = getFieldType(structTy, offsetValue);
    }
    return builder.create<LLVM::GEPOp>(loc, retTy, addr, index);
  }

  Value MLIRGenerator::Load(Location loc, Value addr, std::optional<int> offset)
  {
    if (!isa<LLVM::GEPOp>(addr.getDefiningOp()))
      addr = GEP(loc, addr, offset);
    else
      assert(
        (!offset.has_value() || offset == 0) &&
        "Can't take an offset of a GEP");
    return builder.create<LLVM::LoadOp>(loc, addr);
  }

  Value MLIRGenerator::AutoLoad(
    Location loc, Value addr, Type ty, std::optional<int> offset)
  {
    // If it's not an address, there's nothing to load
    if (!isPointer(addr))
      return addr;

    // If the expected type is a pointer, we want the address, not the value
    if (ty && ty.isa<PointerType>())
      return addr;

    auto elmTy = getPointedType(addr);

    // If type was specified, check it matches the address type
    if (ty)
      assert(elmTy == ty && "Invalid pointer load");

    return Load(loc, addr, offset);
  }

  void MLIRGenerator::Store(
    Location loc, Value addr, Value val, std::optional<int> offset)
  {
    if (!isa<LLVM::GEPOp>(addr.getDefiningOp()))
      addr = GEP(loc, addr, offset);
    else
      assert(
        (!offset.has_value() || offset == 0) &&
        "Can't take an offset of a GEP");
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
    // Use contents as name if none provided
    if (name.empty())
      name = str;

    // Avoid redefinition
    auto global = module->lookupSymbol<LLVM::GlobalOp>(name);
    if (!global)
    {
      // In LLVM, strings are arrays of i8 elements
      auto i8 = builder.getIntegerType(8);
      auto strTy = ArrayType::get(i8, str.size());
      auto strAttr = builder.getStringAttr(str);

      // In LLVM, constant strings are global objects
      auto moduleBuilder = OpBuilder(*module);
      global = moduleBuilder.create<LLVM::GlobalOp>(
        builder.getUnknownLoc(),
        strTy,
        /*isConstant=*/true,
        LLVM::Linkage::Private,
        name,
        strAttr);
      module->push_back(global);
    }

    // But their addresses are a local operation
    return builder.create<LLVM::AddressOfOp>(builder.getUnknownLoc(), global);
  }

  Value MLIRGenerator::Arithmetic(Location loc, StringRef name, Value ops)
  {
    // FIXME: We already converted U32 to i32 so this "works". But we need
    // to make sure we want that conversion as early as it is, and if not,
    // we need to implement this as a standard select and convert that
    // later. However, that would only work if U32 has a method named "+",
    // or if we declare it on the fly and then clean up when we remove the
    // call.
    auto it = arithmetic.find(name);
    // FIXME: Implement call to intrinsics, too
    assert(it != arithmetic.end() && "Unknown arithmetic operation");
    auto numOps = it->second;

    auto getOperand = [this, loc, ops](size_t offset) {
      auto ptr = GEP(loc, ops, offset);
      return Load(loc, ptr);
    };

    llvm::SmallVector<Value> values;
    Type retTy;
    switch (numOps)
    {
      case 1:
        values.push_back(ops);
        retTy = ops.getType();
        break;
      case 2:
      {
        auto structTy = getPointedStructType(ops, /*anonymous*/ true);
        assert(
          structTy && structTy.getBody().size() == 2 &&
          "Binary op needs two operands");

        // Promote types to be the same, or ops don't work, in the end, both
        // types are identical and the same as the return type.
        auto lhs = getOperand(0);
        auto rhs = getOperand(1);
        std::tie(lhs, rhs) = Promote(lhs, rhs);
        values.append({lhs, rhs});
        retTy = rhs.getType();
        break;
      }
      default:
        assert(false && "Unsupported arithmetic operation");
        return Value();
    }

    // If the operation is known, lower as MLIR op
    ValueRange valRange{values};
    auto state = OperationState(loc, name, valRange, retTy, /*attrs*/ {});
    auto op = builder.createOperation(state);
    auto value = op->getResult(0);
    return value;
  }
}
