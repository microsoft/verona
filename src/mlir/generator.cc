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
  size_t MLIRGenerator::numArithmeticOps(llvm::StringRef name)
  {
    // FIXME: Using MLIR standard dialect comes with complications, since the
    // arithmetic in MLIR is mostly for ML and doens't have the semantics we
    // need for generic lowering. We should probably use the LLVM dialect + LLVM
    // intrinsic calls for the rest.

    // TODO: Add comparators
    // TODO: Add llvm intrinsics
    return llvm::StringSwitch<size_t>(name)
      .Case("std.absf", 1)
      .Case("std.ceilf", 1)
      .Case("std.floorf", 1)
      .Case("std.negf", 1)
      .Case("std.addf", 2)
      .Case("std.subf", 2)
      .Case("std.mulf", 2)
      .Case("std.divf", 2)
      .Case("std.fmaf", 3)
      .Case("std.addi", 2)
      .Case("std.subi", 2)
      .Case("std.muli", 2)
      .Case("std.divi_unsigned", 2)
      .Case("std.remi_unsigned", 2)
      .Case("std.divi_signed", 2)
      .Case("std.remi_signed", 2)
      .Case("std.ceildivi_signed", 2)
      .Case("std.floordivi_signed", 2)
      .Case("std.and", 2)
      .Case("std.or", 2)
      .Case("std.xor", 2)
      .Case("std.select", 2)
      .Case("std.shift_left", 2)
      .Case("std.shift_right_signed", 2)
      .Case("std.shift_right_unsigned", 2)
      .Case("std.fpext", 1)
      .Case("std.fptrunc", 1)
      .Case("std.sexti", 1)
      .Case("std.zexti", 1)
      .Case("std.trunci", 1)
      .Case("std.index_cast", 1)
      .Case("std.fptosi", 1)
      .Case("std.fptoui", 1)
      .Case("std.sitofp", 1)
      .Case("std.uitofp", 1)
      .Default(0);
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
    // Empty type for (anon && ident) || (!anon && !ident)
    if (anonymous == sTy.isIdentified())
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

  FuncOp MLIRGenerator::Proto(
    Location loc,
    llvm::StringRef name,
    llvm::ArrayRef<Type> types,
    llvm::ArrayRef<Type> retTy)
  {
    // Should not declare two functions with the same name
    auto func = module->lookupSymbol<FuncOp>(name);
    assert(!func && "Redeclaration of existing function");

    // Create function
    auto funcTy = builder.getFunctionType(types, {retTy});
    func = FuncOp::create(loc, name, funcTy);
    // FIXME: This should be private unless we export, but for now we make
    // it public to test IR generation before implementing public visibility
    func.setVisibility(SymbolTable::Visibility::Public);
    return func;
  }

  FuncOp MLIRGenerator::StartFunction(FuncOp& func)
  {
    // If it was declared, make sure it wasn't also defined
    assert(
      func.getRegion().getBlocks().size() == 0 &&
      "Redefinition of existing function");

    // Create entry block, set builder entry point
    auto& entryBlock = *func.addEntryBlock();
    builder.setInsertionPointToStart(&entryBlock);

    return func;
  }

  Value
  MLIRGenerator::Call(Location loc, FuncOp func, llvm::ArrayRef<Value> args)
  {
    // TODO: Implement dynamic method calls
    auto call = builder.create<CallOp>(loc, func, args);
    // TODO: Implement multiple return values (tuples?)
    return call->getOpResult(0);
  }

  // ==================================================== Low level generators

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

  Value
  MLIRGenerator::Arithmetic(Location loc, StringRef name, Value ops, Type retTy)
  {
    // FIXME: We already converted U32 to i32 so this "works". But we need
    // to make sure we want that conversion as early as it is, and if not,
    // we need to implement this as a standard select and convert that
    // later. However, that would only work if U32 has a method named "+",
    // or if we declare it on the fly and then clean up when we remove the
    // call.
    auto numOps = numArithmeticOps(name);
    // FIXME: Implement call to intrinsics, too
    assert(numOps && "Unknown arithmetic operation");

    auto getOperand = [this, loc, ops](size_t offset) {
      auto ptr = GEP(loc, ops, offset);
      return Load(loc, ptr);
    };

    llvm::SmallVector<Value> values;
    switch (numOps)
    {
      case 1:
        // Update operands and return type
        values.push_back(ops);
        if (!retTy)
          retTy = ops.getType();
        break;
      case 2:
      {
        auto structTy = getPointedStructType(ops, /*anonymous*/ true);
        // Make sure this is a tuple
        assert(
          structTy && structTy.getBody().size() == 2 &&
          "Binary op needs two operands");

        // Get both operands from a tuple
        auto lhs = getOperand(0);
        auto rhs = getOperand(1);

        // Make sure the types are the same
        assert(
          lhs.getType() == rhs.getType() && "Binop types must be identical");

        // Update operands and return type
        values.append({lhs, rhs});
        if (!retTy)
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

  void MLIRGenerator::Return(Location loc, FuncOp& func, Value ret)
  {
    // Check if needs to return a value at all
    if (hasTerminator(builder.getBlock()))
      return;

    // Lower return value
    bool needsReturn = !func.getType().getResults().empty();
    if (needsReturn)
    {
      assert(ret && "Function return value not found");
      auto retTy = func.getType().getResults()[0];
      assert(
        retTy == ret.getType() && "Last operand and return types mismatch");
      builder.create<ReturnOp>(loc, ret);
    }
    else
    {
      assert(!ret && "Value passed to function that returns void");
      builder.create<ReturnOp>(loc);
    }
  }
}
