// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "generator.h"

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
    return !bb->getOperations().empty() && bb->back().isKnownTerminator();
  }

  /// Return true if the value was created by an alloca operation.
  /// FIXME: So far, this is the only way to know if the value is an address
  bool isAlloca(mlir::Value val)
  {
    return val.getDefiningOp() &&
      llvm::isa<mlir::AllocaOp>(val.getDefiningOp());
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
    auto err = gen.parseModule(ast);
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
    return builder.getFileLineColLoc(
      Identifier::get(path, context), line, column);
  }

  std::pair<mlir::Value, mlir::Value>
  Generator::upcast(mlir::Value lhs, mlir::Value rhs)
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
  }

  // ===================================================== AST -> MLIR
  llvm::Error Generator::parseModule(Ast ast)
  {
    assert(ast->kind() == Kind::Class && "Bad node");

    // Modules are just global classes
    auto res = parseClass(ast);
    if (auto err = res.takeError())
      return err;

    module = res->get<ModuleOp>();
    return llvm::Error::success();
  }

  llvm::Expected<ReturnValue> Generator::parseClass(Ast ast)
  {
    assert(ast->kind() == Kind::Class && "Bad node");
    auto node = ast->as<Class>();
    auto loc = getLocation(ast);

    // Push another scope for variables and functions
    SymbolScopeT var_scope(symbolTable);
    FunctionScopeT func_scope(functionTable);

    // Creates the scope, each class/module is a new sub-module.
    auto name = node.id.view();
    // Only one module is unnamed, the root one
    if (name.empty())
      name = rootModuleName;
    auto scope = ModuleOp::create(loc, StringRef(name));

    // Lower members, types, functions
    for (auto sub : node.members)
    {
      switch (sub->kind())
      {
        case Kind::Class:
        {
          auto mem = parseNode(sub);
          if (auto err = mem.takeError())
            return std::move(err);
          scope.push_back(mem->get<ModuleOp>());
          break;
        }
        case Kind::Using:
          // Ignore for now as this is just a reference to the module name
          // that will be lowered, but module names aren't being lowered now.
          break;
        case Kind::Function:
        {
          auto func = parseNode(sub);
          if (auto err = func.takeError())
            return std::move(err);
          scope.push_back(func->get<FuncOp>());
          break;
        }
        case Kind::Field:
        default:
          return runtimeError("Wrong member in class");
      }
    }

    return scope;
  }

  llvm::Expected<ReturnValue> Generator::parseNode(Ast ast)
  {
    switch (ast->kind())
    {
      case Kind::Class:
        return parseClass(ast);
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
    Types types;
    for (auto p : func->params)
    {
      auto param = nodeAs<Param>(p);
      assert(param && "Bad Node");
      argNames.push_back(param->location.view());
      types.push_back(parseType(param->type));
      // TODO: Handle default init
    }

    // Check return type (TODO: implement multiple returns)
    Types retTy;
    if (func->result)
    {
      retTy.push_back(parseType(func->result));
    }

    // Declare all arguments on current scope
    SymbolScopeT var_scope(symbolTable);
    auto name = func->name.view();
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
    bool hasLast = last->hasValue();

    if (hasLast)
    {
      auto retVal = last->get<Value>();
      builder.create<ReturnOp>(loc, retVal);
    }
    else
    {
      builder.create<ReturnOp>(loc);
    }

    return funcIR;
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
  // Though, before we move things from here, we need to know what to do when
  // a select is in a numeric class that doesn't have those methods.
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

    // FIXME: Multiple method names?
    auto opName = select->typenames[0]->location.view();

    // FIXME: "special case" return for now, to make it work without method call
    if (opName == "return")
      return rhs;

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
    if (auto funcOp = functionTable.lookup(opName))
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
    Value addr = generateAlloca(getLocation(ofty), newTy);
    return symbolTable.update(name, addr);
  }

  llvm::Expected<ReturnValue> Generator::parseAssign(Ast ast)
  {
    auto assign = nodeAs<Assign>(ast);
    assert(assign && "Bad Node");

    // lhs has to be an addressable expression (ref, let, var)
    auto res = parseRef(assign->left);
    if (auto err = res.takeError())
      return std::move(err);
    auto addr = res->get<Value>();

    // Must be an address
    if (!isAlloca(addr))
      return runtimeError("Assign lhs not an address");

    // Load the existing value to return
    auto old = generateLoad(getLocation(assign), addr);

    // Evaluate the right hand side and assign to the binded name
    auto rhsNode = parseNode(assign->right);
    if (auto err = rhsNode.takeError())
      return std::move(err);
    auto rhs = rhsNode->get<Value>();
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
        // FIXME: This is possibly too early to do this conversion, but helps us
        // run lots of tests before actually implementing classes, etc.
        Type type = llvm::StringSwitch<Type>(name)
                      .Case("U32", builder.getI32Type())
                      .Case("U64", builder.getI64Type())
                      .Case("F32", builder.getF32Type())
                      .Case("F64", builder.getF64Type())
                      .Default(Type());
        assert(type && "Classes not implemented yet");
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
    func.setVisibility(SymbolTable::Visibility::Private);
    return functionTable.insert(name, func);
  }

  llvm::Expected<FuncOp> Generator::generateEmptyFunction(
    Location loc,
    llvm::StringRef name,
    llvm::ArrayRef<llvm::StringRef> args,
    llvm::ArrayRef<Type> types,
    llvm::ArrayRef<Type> retTy)
  {
    assert(args.size() == types.size() && "Argument/type mismatch");

    // If it's not declared yet, do so. This simplifies direct declaration of
    // compiler functions. User functions should be checked at the parse level.
    auto func = functionTable.inScope(name);
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

    // Upcast types to be the same, or ops don't work, in the end, both types
    // are identical and the same as the return type.
    std::tie(lhs, rhs) = upcast(lhs, rhs);
    auto retTy = lhs.getType();

    // FIXME: We already converted U32 to i32 so this "works". But we need to
    // make sure we want that conversion as early as it is, and if not, we need
    // to implement this as a standard select and convert that later. However,
    // that would only work if U32 has a method named "+", or if we declare it
    // on the fly and then clean up when we remove the call.

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
    // FIXME: Get sizeof(). We probably will need alloc/load/store on our own
    // dialect soon, not to have to depend on memref and its idiosyncrasies
    auto memrefTy = mlir::MemRefType::get({1}, ty);
    return builder.create<AllocaOp>(loc, memrefTy);
  }

  Value Generator::generateLoad(Location loc, Value addr)
  {
    ValueRange index(generateZero(builder.getIndexType()));
    return builder.create<LoadOp>(loc, addr, index);
  }

  void Generator::generateStore(Location loc, Value addr, Value val)
  {
    ValueRange index(generateZero(builder.getIndexType()));
    builder.create<StoreOp>(loc, val, addr, index);
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
  }
}
