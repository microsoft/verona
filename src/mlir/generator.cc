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

  /// Get node as a shared pointer of a sub-type
  template<class T>
  Node<T> nodeAs(Ast from)
  {
    return std::make_shared<T>(from->as<T>());
  }
}

namespace mlir::verona
{
  auto Generator::rootModuleName = "__";

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

  mlir::Type Generator::compatibleArithmeticType(mlir::Type lhs, mlir::Type rhs)
  {
    // Shortcut for when both are the same
    if (lhs == rhs)
      return lhs;

    auto lhsSize = lhs.getIntOrFloatBitWidth();
    auto rhsSize = lhs.getIntOrFloatBitWidth();

    // Check compatibility options
    // TODO: This is overly simplistic.
    if (lhs.isSignedInteger() && rhs.isSignedInteger())
    {
      return builder.getIntegerType(std::max(lhsSize, rhsSize));
    }
    if (lhs.isUnsignedInteger() && rhs.isUnsignedInteger())
    {
      return builder.getIntegerType(std::max(lhsSize, rhsSize));
    }
    // Ugly, there is no isFloat... :( but we know it's not int either here
    if (lhs.isIntOrFloat() && rhs.isIntOrFloat())
    {
      switch (std::max(lhsSize, rhsSize))
      {
        case 32:
          return builder.getF32Type();
        case 64:
          return builder.getF64Type();
      }
    }

    // If not compatible, return empty type
    return Type();
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
      case Kind::Let:
        return parseLet(ast);
      case Kind::Assign:
        return parseAssign(ast);
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

  llvm::Expected<ReturnValue> Generator::parseSelect(Ast ast)
  {
    auto select = nodeAs<Select>(ast);
    assert(select && "Bad Node");
    auto loc = getLocation(ast);

    // FIXME: This is a hack to make arithmetic work. We'll have to add
    // recognition of numeric types somewhere but it's probably not here.
    // Though, before we move things from here, we need to know what to do when
    // a select is in a numeric class that doesn't have those methods.
    auto rhsNode = parseNode(select->args);
    if (auto err = rhsNode.takeError())
      return std::move(err);
    auto rhs = rhsNode->get<Value>();
    auto rhsTy = rhs.getType();

    // FIXME: Multiple method names?
    auto opName = select->typenames[0]->location.view();

    // FIXME: "special case" return for now, to make it work without method call
    // TODO: If this isn't wrong, we need to add other unary operators here, too
    if (opName == "return")
      return parseNode(select->args);

    // Binary operators have the left hand side as well as right hand side
    auto lhsNode = parseNode(select->expr);
    if (auto err = lhsNode.takeError())
      return std::move(err);
    auto lhs = lhsNode->get<Value>();
    auto lhsTy = lhs.getType();

    // FIXME: We already converted U32 to i32 so this "works". But we need to
    // make sure we want that conversion as early as it is, and if not, we need
    // to implement this as a standard select and convert that later. However,
    // that would only work if U32 has a method named "+", or if we declare it
    // on the fly and then clean up when we remove the call.
    auto compatibleTy = compatibleArithmeticType(lhsTy, rhsTy);
    if (compatibleTy)
    {
      // Floating point arithmetic
      if (compatibleTy.isF32() || compatibleTy.isF64())
      {
        auto op =
          llvm::StringSwitch<Value>(opName)
            .Case("+", builder.create<AddFOp>(loc, compatibleTy, lhs, rhs))
            .Default({});
        assert(op && "Unknown arithmetic operator");
        return op;
      }
      // Integer arithmetic
      auto op =
        llvm::StringSwitch<Value>(opName)
          .Case("+", builder.create<AddIOp>(loc, compatibleTy, lhs, rhs))
          .Default({});
      assert(op && "Unknown arithmetic operator");
      return op;
    }

    return runtimeError("Select not implemented yet");
  }

  llvm::Expected<ReturnValue> Generator::parseRef(Ast ast)
  {
    auto ref = nodeAs<Ref>(ast);
    assert(ref && "Bad Node");
    return symbolTable.lookup(ref->location.view());
  }

  llvm::Expected<ReturnValue> Generator::parseLet(Ast ast)
  {
    auto let = nodeAs<Let>(ast);
    assert(let && "Bad Node");
    // FIXME: Just binding an empty value for now. Later we'll have to make
    // sure the value is only replaced if empty, unlike `var` that can be
    // reassigned multiple times.
    return symbolTable.insert(let->location.view(), mlir::Value());
  }

  llvm::Expected<ReturnValue> Generator::parseVar(Ast ast)
  {
    auto var = nodeAs<Var>(ast);
    assert(var && "Bad Node");
    // FIXME: Just binding an empty value for now. Var can be overriten as many
    // times as needed.
    return symbolTable.insert(var->location.view(), mlir::Value());
  }

  llvm::Expected<ReturnValue> Generator::parseAssign(Ast ast)
  {
    auto assign = nodeAs<Assign>(ast);
    assert(assign && "Bad Node");

    // Name to bind and old value to return
    std::string_view bind;
    mlir::Value old;
    bool onlyIfEmpty;

    // Grab lhs has to be an addressable expression
    // FIXME: We still don't have the representation of an address, so we
    // restrict assigns to work with let/ref and custom-lower it here.
    if (auto ref = nodeAs<Ref>(assign->left))
    {
      bind = ref->location.view();
      auto res = parseRef(ref);
      if (auto err = res.takeError())
        return std::move(err);
      old = res->get<Value>();
      // FIXME: How do we know this is a let or a var?
      onlyIfEmpty = true;
    }
    else if (auto let = nodeAs<Let>(assign->left))
    {
      bind = let->location.view();
      auto res = parseLet(assign->left);
      if (auto err = res.takeError())
        return std::move(err);
      old = res->get<Value>();
      onlyIfEmpty = true;
    }
    else if (auto let = nodeAs<Var>(assign->left))
    {
      bind = let->location.view();
      auto res = parseVar(assign->left);
      if (auto err = res.takeError())
        return std::move(err);
      old = res->get<Value>();
      onlyIfEmpty = false;
    }
    else
    {
      return runtimeError("Invalid assign lhs");
    }

    // Evaluate the right hand side and assign to the binded name
    auto rhsNode = parseNode(assign->right);
    if (auto err = rhsNode.takeError())
      return std::move(err);
    auto rhs = rhsNode->get<Value>();
    symbolTable.update(bind, rhs, onlyIfEmpty);

    // BROKEN: Actually assign the value to the lhs name
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
      symbolTable.insert(std::get<0>(arg_val), std::get<1>(arg_val));
    }

    return func;
  }
}
