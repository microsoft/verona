// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "generator.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Types.h"

using namespace verona::parser;

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
      return mlir::UnknownLoc::get(context);

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
        case Kind::Field:
        case Kind::Function:
        default:
          return runtimeError("Wrong member in class");
      }
    }

    return scope;
  }

  llvm::Expected<ReturnValue> Generator::parseBlock(AstPath nodes)
  {
    // Blocks add lexical context
    SymbolScopeT var_scope{symbolTable};

    ReturnValue last;
    for (auto sub : nodes)
    {
      auto node = parseNode(sub);
      if (auto err = node.takeError())
        return std::move(err);
      last = *node;
    }
    return last;
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
}
