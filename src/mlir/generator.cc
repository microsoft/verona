// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "generator.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Types.h"

using namespace verona::parser;

namespace
{
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
          auto func = parseFunction(sub);
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

    // TODO: Lower arguments
    llvm::SmallVector<llvm::StringRef, 1> argNames;
    Types types;
    llvm::SmallVector<mlir::Type, 1> retTy;

    // Declare all arguments on current scope
    SymbolScopeT var_scope(symbolTable);
    auto name = func->name.view();
    auto def =
      generateEmptyFunction(getLocation(ast), name, argNames, types, retTy);
    if (auto err = def.takeError())
      return std::move(err);
    auto& funcIR = *def;

    // TODO: Lower body

    // TODO: Lower return value
    builder.create<mlir::ReturnOp>(loc);

    return funcIR;
  }

  // ===================================================== MLIR Generators
  llvm::Expected<mlir::FuncOp> Generator::generateProto(
    mlir::Location loc,
    llvm::StringRef name,
    llvm::ArrayRef<mlir::Type> types,
    llvm::ArrayRef<mlir::Type> retTy)
  {
    // Create function
    auto funcTy = builder.getFunctionType(types, retTy);
    auto func = mlir::FuncOp::create(loc, name, funcTy);
    func.setVisibility(mlir::SymbolTable::Visibility::Private);
    return functionTable.insert(name, func);
  }

  llvm::Expected<mlir::FuncOp> Generator::generateEmptyFunction(
    mlir::Location loc,
    llvm::StringRef name,
    llvm::ArrayRef<llvm::StringRef> args,
    llvm::ArrayRef<mlir::Type> types,
    llvm::ArrayRef<mlir::Type> retTy)
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

    // TODO: Declare all arguments
    return func;
  }
}
