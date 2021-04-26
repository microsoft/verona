// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "generator.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Types.h"

#include <string>

using namespace verona::parser;

namespace
{
  /// Helper to make sure the basic block has a terminator
  bool hasTerminator(mlir::Block* bb)
  {
    return !bb->getOperations().empty() &&
      bb->back().mightHaveTrait<mlir::OpTrait::IsTerminator>();
  }

  /// Return true if the value was created by an alloca operation.
  /// FIXME: So far, this is the only way to know if the value is an address
  bool isAlloca(mlir::Value val)
  {
    return val && val.getDefiningOp() &&
      llvm::isa<mlir::memref::AllocaOp>(val.getDefiningOp());
  }

  /// Return the type that needs to be allocated by the alloca instruction.
  /// FIXME: Still using memref (largest type x num elms), should use LLVM's
  /// own alloca on native types instead.
  std::pair<mlir::Type, long> getAllocaSize(mlir::LLVM::LLVMStructType structTy)
  {
    mlir::Type largestTy;
    unsigned int largestSize = 0;
    unsigned int elms = 0;
    for (auto elm : structTy.getBody())
    {
      if (elm.isIntOrFloat())
      {
        // "Native" checks size and increment span
        auto size = elm.getIntOrFloatBitWidth();
        if (size > largestSize)
        {
          largestSize = size;
          largestTy = elm;
        }
        elms++;
      }
      else
      {
        // Class, recurse and update from nested info
        auto structTy = elm.dyn_cast<mlir::LLVM::LLVMStructType>();
        auto [subTy, subElms] = getAllocaSize(structTy);
        auto size = subTy.getIntOrFloatBitWidth();
        if (size > largestSize)
        {
          largestSize = size;
          largestTy = subTy;
        }
        elms += subElms;
      }
    }
    return std::make_pair(largestTy, elms);
  }

  /// Get node as a shared pointer of a sub-type
  template<class T>
  Node<T> nodeAs(Ast from)
  {
    return std::make_shared<T>(from->as<T>());
  }
}

namespace mlir::verona
{
  // ===================================================== Public Interface
  llvm::Expected<OwningModuleRef>
  Generator::lower(MLIRContext* context, Ast ast)
  {
    Generator gen(context);
    auto err = gen.parseRootModule(ast);
    if (err)
      return std::move(err);

    return std::move(gen.module);
  }

  // ===================================================== Helpers
  Location Generator::getLocation(Ast ast)
  {
    if (!ast->location.source)
      return builder.getUnknownLoc();

    auto path = ast->location.source->origin;
    auto [line, column] = ast->location.linecol();
    return mlir::FileLineColLoc::get(builder.getIdentifier(path), line, column);
  }

  std::pair<mlir::Value, mlir::Value>
  Generator::typePromotion(mlir::Value lhs, mlir::Value rhs)
  {
    auto lhsType = lhs.getType();
    auto rhsType = rhs.getType();

    // Shortcut for when both are the same
    if (lhsType == rhsType)
      return {lhs, rhs};

    auto lhsSize = lhsType.getIntOrFloatBitWidth();
    auto rhsSize = rhsType.getIntOrFloatBitWidth();

    // Integer upcasts
    auto lhsInt = lhsType.dyn_cast<IntegerType>();
    auto rhsInt = rhsType.dyn_cast<IntegerType>();
    if (lhsInt && rhsInt)
    {
      if (lhsSize < rhsSize)
        lhs = builder.create<SignExtendIOp>(lhs.getLoc(), rhsType, lhs);
      else
        rhs = builder.create<SignExtendIOp>(rhs.getLoc(), lhsType, rhs);

      return {lhs, rhs};
    }

    // Floating point casts
    auto lhsFP = lhsType.dyn_cast<FloatType>();
    auto rhsFP = rhsType.dyn_cast<FloatType>();
    if (lhsFP && rhsFP)
    {
      if (lhsSize < rhsSize)
        lhs = builder.create<FPExtOp>(lhs.getLoc(), rhsType, lhs);
      else
        rhs = builder.create<FPExtOp>(rhs.getLoc(), lhsType, rhs);

      return {lhs, rhs};
    }

    // If not compatible, assert
    assert(false && "Upcast between incompatible types");

    // Appease MSVC warnings
    return {lhs, rhs};
  }

  std::string Generator::mangleName(
    llvm::StringRef name, llvm::ArrayRef<llvm::StringRef> scope)
  {
    if (scope.empty())
      scope = functionScope;

    // FIXME: This is a hack to help running LLVM modules
    if (name == "main")
      return name.str();

    // TODO: This is inefficient but works for now
    std::string fullName;
    for (auto s : scope)
    {
      fullName += s.str() + "__";
    }
    fullName += name.str();
    return fullName;
  }

  // ===================================================== AST -> MLIR
  llvm::Error Generator::parseRootModule(Ast ast)
  {
    auto node = nodeAs<Class>(ast);
    assert(node && "Bad node");

    // Modules are just global classes
    return parseClass(ast);
  }

  llvm::Error Generator::parseClass(Ast ast)
  {
    auto node = nodeAs<Class>(ast);
    assert(node && "Bad node");
    auto loc = getLocation(ast);

    StringRef modName;
    if (!module)
    {
      // Creates the global module
      module = ModuleOp::create(loc, StringRef(rootModuleName));
      modName = rootModuleName;
    }
    else
    {
      modName = node->id.view();
    }

    // Push another scope for variables, functions and types
    SymbolScopeT var_scope(symbolTable);
    functionScope.push_back(modName);
    auto type =
      LLVM::LLVMStructType::getIdentified(builder.getContext(), modName);
    llvm::SmallVector<Type, 4> fields;

    // Lower members, types, functions
    for (auto sub : node->members)
    {
      switch (sub->kind())
      {
        case Kind::Class:
        {
          auto err = parseClass(sub);
          if (err)
            return err;
          functionScope.pop_back();
          break;
        }
        case Kind::Using:
          // Ignore for now as this is just a reference to the module name
          // that will be lowered, but module names aren't being lowered
          // now.
          break;
        case Kind::Function:
        {
          auto func = parseNode(sub);
          if (auto err = func.takeError())
            return err;
          module->push_back(func->get<FuncOp>());
          break;
        }
        case Kind::Field:
        {
          auto field = parseField(sub);
          if (auto err = field.takeError())
            return err;
          fields.push_back(*field);
          break;
        }
        default:
          return runtimeError("Wrong member in class");
      }
    }
    if (mlir::failed(type.setBody(fields, /*packed*/ false)))
      return runtimeError("Error setting fields to class");

    return llvm::Error::success();
  }

  llvm::Expected<ReturnValue> Generator::parseNode(Ast ast)
  {
    switch (ast->kind())
    {
      case Kind::Function:
        return parseFunction(ast);
      case Kind::Lambda:
        return parseLambda(ast);
      case Kind::Select:
        return parseSelect(ast);
      case Kind::Ref:
        return parseRef(ast);
      case Kind::Assign:
        return parseAssign(ast);
      case Kind::Let:
      case Kind::Var:
        return parseLocalDecl(ast);
      case Kind::Oftype:
        return parseOfType(ast);
      case Kind::Character:
      case Kind::Int:
      case Kind::Float:
      case Kind::Hex:
      case Kind::Binary:
      case Kind::Bool:
        return parseLiteral(ast);
      default:
        // TODO: Implement all others
        break;
    }

    return runtimeError(
      "Node " + std::string(kindname(ast->kind())) + " not implemented yet");
  }

  llvm::Expected<ReturnValue> Generator::parseFunction(Ast ast)
  {
    auto func = nodeAs<Function>(ast);
    assert(func && "Bad node");
    auto loc = getLocation(ast);

    // Find all arguments
    llvm::SmallVector<llvm::StringRef, 1> argNames;
    llvm::SmallVector<Type, 1> types;
    for (auto p : func->params)
    {
      auto param = nodeAs<Param>(p);
      assert(param && "Bad Node");
      argNames.push_back(param->location.view());
      types.push_back(parseType(param->type));
      // TODO: Handle default init
    }

    // Check return type (TODO: implement multiple returns)
    llvm::SmallVector<Type, 1> retTy;
    if (func->result)
    {
      retTy.push_back(parseType(func->result));
    }

    // Declare all arguments on current scope
    SymbolScopeT var_scope(symbolTable);
    auto name = mangleName(func->name.view());
    auto def =
      generateEmptyFunction(getLocation(ast), name, argNames, types, retTy);
    if (auto err = def.takeError())
      return std::move(err);
    auto& funcIR = *def;

    // Lower body
    auto body = func->body;
    auto last = parseNode(body);
    if (auto err = last.takeError())
      return std::move(err);

    // Check if needs to return a value at all
    if (hasTerminator(builder.getBlock()))
      return funcIR;

    // Lower return value
    // (TODO: cast type if not the same)
    bool needsReturn = !retTy.empty();

    if (needsReturn)
    {
      assert(last->hasValue() && "No value to return");
      auto retVal = last->get<Value>();
      builder.create<ReturnOp>(loc, retVal);
    }
    else
    {
      builder.create<ReturnOp>(loc);
    }

    return funcIR;
  }

  llvm::Expected<Type> Generator::parseField(Ast ast)
  {
    auto field = nodeAs<Field>(ast);
    assert(field && "Bad node");

    auto type = parseType(field->type);
    // TODO: Add names to a hash so we can access for field read/write.
    // TODO: Implement initialiser

    return type;
  }

  llvm::Expected<ReturnValue> Generator::parseLambda(Ast ast)
  {
    auto lambda = nodeAs<Lambda>(ast);
    assert(lambda && "Bad Node");

    // Blocks add lexical context
    SymbolScopeT var_scope{symbolTable};

    ReturnValue last;
    llvm::SmallVector<Ast, 1> nodes;
    for (auto sub : lambda->body)
    {
      auto node = parseNode(sub);
      if (auto err = node.takeError())
        return std::move(err);
      last = *node;
    }
    return last;
  }

  // FIXME: This is a hack to make arithmetic work. We'll have to add
  // recognition of numeric types somewhere but it's probably not here.
  // Though, before we move things from here, we need to know what to do
  // when a select is in a numeric class that doesn't have those methods.
  llvm::Expected<ReturnValue> Generator::parseSelect(Ast ast)
  {
    auto select = nodeAs<Select>(ast);
    assert(select && "Bad Node");
    auto loc = getLocation(ast);

    Value lhs, rhs;

    // This is either:
    //  * the RHS of a binary operator
    //  * the argument of a unary operator
    //  * the arguments of the function call as a tuple
    if (select->args)
    {
      // TODO: Implement tuple for multiple argument
      auto rhsNode = parseNode(select->args);
      if (auto err = rhsNode.takeError())
        return std::move(err);
      rhs = rhsNode->get<Value>();
      if (isAlloca(rhs))
        rhs = generateLoad(loc, rhs);
    }

    // FIXME: "special case" return for now, to make it work without method
    // call
    if (select->typenames[0]->location.view() == "return")
      return rhs;

    // Typenames indicate the context and the function name
    llvm::SmallVector<llvm::StringRef, 3> scope;
    size_t end = select->typenames.size() - 1;
    for (size_t i = 0; i < end; i++)
    {
      scope.push_back(select->typenames[i]->location.view());
    }
    std::string opName =
      mangleName(select->typenames[end]->location.view(), scope);

    // This is either:
    //  * the LHS of a binary operator
    //  * the selector for a static/dynamic call of a class member
    if (select->expr)
    {
      auto lhsNode = parseNode(select->expr);
      if (auto err = lhsNode.takeError())
        return std::move(err);
      lhs = lhsNode->get<Value>();
      if (isAlloca(lhs))
        lhs = generateLoad(loc, lhs);
    }

    // Check the function table for a symbol that matches the opName
    // TODO: Use scope to find the right function with the same name
    if (auto funcOp = module->lookupSymbol<FuncOp>(opName))
    {
      // Handle arguments
      // TODO: Handle tuples
      llvm::SmallVector<Value, 1> args;
      if (rhs)
        args.push_back(rhs);
      auto res = generateCall(loc, funcOp, args);
      if (auto err = res.takeError())
        return std::move(err);
      return *res;
    }

    // If function does not exist, it's either arithmetic or an error
    // for arithmetic, we only take the op name, not the context
    opName = select->typenames[end]->location.view();
    auto res = generateArithmetic(loc, opName, lhs, rhs);
    if (auto err = res.takeError())
      return std::move(err);
    return *res;
  }

  llvm::Expected<ReturnValue> Generator::parseRef(Ast ast)
  {
    auto ref = nodeAs<Ref>(ast);
    assert(ref && "Bad Node");
    return symbolTable.lookup(ref->location.view());
  }

  llvm::Expected<ReturnValue> Generator::parseLocalDecl(Ast ast)
  {
    // FIXME: for now, just creates a new empty value that can be updated.
    return symbolTable.insert(ast->location.view(), Value());
  }

  llvm::Expected<ReturnValue> Generator::parseOfType(Ast ast)
  {
    auto ofty = nodeAs<Oftype>(ast);
    assert(ofty && "Bad Node");
    auto name = ofty->expr->location.view();

    // Make sure the variable is uninitialized
    auto val = symbolTable.lookup(name, /*local scope*/ true);
    assert(!val);

    // FIXME: for now, just updates the reference's type
    auto newTy = parseType(ofty->type);
    // FIXME: This is probably the wrong place to do this
    Value addr = generateAlloca(getLocation(ofty), newTy);
    return symbolTable.update(name, addr);
  }

  llvm::Expected<ReturnValue> Generator::parseAssign(Ast ast)
  {
    auto assign = nodeAs<Assign>(ast);
    assert(assign && "Bad Node");

    // lhs has to be an addressable expression (ref, let, var)
    auto res = parseNode(assign->left);
    if (auto err = res.takeError())
      return std::move(err);
    auto addr = res->get<Value>();

    // Evaluate the right hand side and assign to the binded name
    auto rhsNode = parseNode(assign->right);
    if (auto err = rhsNode.takeError())
      return std::move(err);
    auto rhs = rhsNode->get<Value>();

    // No address means inline let/var (incl. temps), which has no type
    // We evaluate the RHS first to get its type and create an address of the
    // same type to store in.
    if (!isAlloca(addr))
    {
      auto name = assign->left->location.view();
      auto type = rhs.getType();
      addr = generateAlloca(getLocation(ast), type);
      symbolTable.update(name, addr);
    }
    assert(isAlloca(addr) && "Couldn't create an address for lhs in assign");

    // Load the existing value to return (most of the time unused, elided)
    auto old = generateLoad(getLocation(assign), addr);

    // Store the new value in the same address
    generateStore(getLocation(assign), addr, rhs);

    // Return the previous value
    return old;
  }

  llvm::Expected<ReturnValue> Generator::parseLiteral(Ast ast)
  {
    auto loc = getLocation(ast);
    switch (ast->kind())
    {
      case Kind::Int:
      {
        auto I = nodeAs<Int>(ast);
        assert(I && "Bad Node");
        auto str = I->location.view();
        auto val = std::stol(str.data());
        auto type = parseType(ast);
        auto op = builder.create<ConstantIntOp>(loc, val, type);
        return op->getOpResult(0);
        break;
      }
      case Kind::Character:
      case Kind::Float:
      case Kind::Hex:
      case Kind::Binary:
      case Kind::Bool:
        assert(false && "Not implemented yet");
      default:
        assert(false && "Bad Node");
    }

    return Value();
  }

  Type Generator::parseType(Ast ast)
  {
    switch (ast->kind())
    {
      case Kind::Int:
        // TODO: Understand what the actual size is
        return builder.getIntegerType(64);
      case Kind::Float:
        // TODO: Understand what the actual size is
        return builder.getF64Type();
      case Kind::TypeRef:
      {
        auto R = nodeAs<TypeRef>(ast);
        assert(R && "Bad Node");
        // TODO: Implement type list
        return parseType(R->typenames[0]);
      }
      case Kind::TypeName:
      {
        auto C = nodeAs<TypeName>(ast);
        assert(C && "Bad Node");
        auto name = C->location.view();
        // FIXME: This is possibly too early to do this conversion, but
        // helps us run lots of tests before actually implementing classes,
        // etc.
        Type type = llvm::StringSwitch<Type>(name)
                      .Case("U32", builder.getI32Type())
                      .Case("U64", builder.getI64Type())
                      .Case("F32", builder.getF32Type())
                      .Case("F64", builder.getF64Type())
                      .Default(Type());
        // If type wasn't detected, it must be a class
        // The order of declaration doesn't matter, so we create empty
        // classes if they're not declared yet.
        if (!type)
        {
          type =
            LLVM::LLVMStructType::getIdentified(builder.getContext(), name);
        }
        assert(type && "Type not found");
        return type;
      }
      case Kind::Character:
      case Kind::Hex:
      case Kind::Binary:
      case Kind::Bool:
        assert(false && "Not implemented yet");
      default:
        assert(false && "Bad Node");
    }
    return Type();
  }

  // ===================================================== MLIR Generators
  llvm::Expected<FuncOp> Generator::generateProto(
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

  llvm::Expected<FuncOp> Generator::generateEmptyFunction(
    Location loc,
    llvm::StringRef name,
    llvm::ArrayRef<llvm::StringRef> args,
    llvm::ArrayRef<Type> types,
    llvm::ArrayRef<Type> retTy)
  {
    assert(args.size() == types.size() && "Argument/type mismatch");

    // If it's not declared yet, do so. This simplifies direct declaration
    // of compiler functions. User functions should be checked at the parse
    // level.
    auto func = module->lookupSymbol<FuncOp>(name);
    if (!func)
    {
      auto proto = generateProto(loc, name, types, retTy);
      if (auto err = proto.takeError())
        return std::move(err);
      func = *proto;
    }

    // Create entry block, set builder entry point
    auto& entryBlock = *func.addEntryBlock();
    auto argVals = entryBlock.getArguments();
    assert(args.size() == argVals.size() && "Argument/value mismatch");
    builder.setInsertionPointToStart(&entryBlock);

    // Declare all arguments
    for (auto arg_val : llvm::zip(args, argVals))
    {
      auto name = std::get<0>(arg_val);
      auto val = std::get<1>(arg_val);
      auto addr = generateAlloca(val.getLoc(), val.getType());
      generateStore(val.getLoc(), addr, val);
      symbolTable.insert(name, addr);
    }

    return func;
  }

  llvm::Expected<Value>
  Generator::generateCall(Location loc, FuncOp func, llvm::ArrayRef<Value> args)
  {
    // TODO: Implement static/dynamic method calls
    auto call = builder.create<CallOp>(loc, func, args);
    // TODO: Implement multiple return values
    return call->getOpResult(0);
  }

  llvm::Expected<Value> Generator::generateArithmetic(
    Location loc, llvm::StringRef opName, Value lhs, Value rhs)
  {
    // FIXME: Implement all unary and binary operators

    // Upcast types to be the same, or ops don't work, in the end, both
    // types are identical and the same as the return type.
    std::tie(lhs, rhs) = typePromotion(lhs, rhs);
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
    assert(retTy.dyn_cast<IntegerType>() && "Bad arithmetic types");
    auto op = llvm::StringSwitch<Value>(opName)
                .Case("+", builder.create<AddIOp>(loc, retTy, lhs, rhs))
                .Default({});
    assert(op && "Unknown arithmetic operator");
    return op;
  }

  Value Generator::generateAlloca(Location loc, Type ty)
  {
    // FIXME: This is the wrong way of doing it but does allocate enough
    // space for all elements.
    mlir::MemRefType memrefTy;
    if (ty.isIntOrFloat())
    {
      memrefTy = mlir::MemRefType::get({1}, ty);
    }
    else
    {
      auto structTy = ty.dyn_cast<LLVM::LLVMStructType>();
      auto [largestTy, elms] = getAllocaSize(structTy);
      memrefTy = mlir::MemRefType::get({elms}, largestTy);
    }
    return builder.create<memref::AllocaOp>(loc, memrefTy);
  }

  Value Generator::generateLoad(Location loc, Value addr)
  {
    auto zero = generateZero(builder.getIndexType());
    ValueRange index(zero);
    return builder.create<memref::LoadOp>(loc, addr, index);
  }

  void Generator::generateStore(Location loc, Value addr, Value val)
  {
    auto zero = generateZero(builder.getIndexType());
    ValueRange index(zero);
    builder.create<memref::StoreOp>(loc, val, addr, index);
  }

  Value Generator::generateZero(Type ty)
  {
    auto loc = builder.getUnknownLoc();
    if (ty.isIndex())
    {
      return builder.create<ConstantIndexOp>(loc, 0);
    }
    else if (auto it = ty.dyn_cast<IntegerType>())
    {
      return builder.create<ConstantIntOp>(loc, 0, it);
    }
    else if (auto ft = ty.dyn_cast<FloatType>())
    {
      APFloat zero = APFloat(0.0);
      return builder.create<ConstantFloatOp>(loc, zero, ft);
    }

    assert(0 && "Type not supported for zero");

    // Appease MSVC warnings
    return Value();
  }
}
