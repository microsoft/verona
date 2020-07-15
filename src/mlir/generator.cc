// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "generator.h"

#include "ast-utils.h"
#include "dialect/VeronaDialect.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

namespace mlir::verona
{
  // FIXME: Until we decide how the interface is going to be, this helps to
  // keep the idea of separation without actually doing it. We could have just
  // a namespace and functions, a static class, or a stateful class, all of
  // which will have different choices on the calls to the interface.
  using namespace ASTInterface;

  // ===================================================== Public Interface
  llvm::Error Generator::readAST(const ::ast::Ast& ast)
  {
    // Parse the AST with the rules below
    if (auto err = parseModule(ast))
      return err;

    // On error, dump module for debug purposes
    if (mlir::failed(mlir::verify(*module)))
    {
      module->dump();
      return runtimeError("MLIR verification failed from Verona file");
    }
    return llvm::Error::success();
  }

  llvm::Error Generator::readMLIR(const std::string& filename)
  {
    if (filename.empty())
      return runtimeError("No input filename provided");

    // Read an MLIR file
    auto srcOrErr = llvm::MemoryBuffer::getFileOrSTDIN(filename);

    if (auto err = srcOrErr.getError())
      return runtimeError(
        "Cannot open file " + filename + ": " + err.message());

    // Setup source manager and parse
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(*srcOrErr), llvm::SMLoc());
    module = mlir::parseSourceFile(sourceMgr, builder.getContext());

    // On error, dump module for debug purposes
    if (mlir::failed(mlir::verify(*module)))
    {
      module->dump();
      return runtimeError("MLIR verification failed from MLIR file");
    }
    return llvm::Error::success();
  }

  llvm::Error Generator::emitMLIR(llvm::StringRef filename, unsigned optLevel)
  {
    if (filename.empty())
      return runtimeError("No output filename provided");

    // Write to the file requested
    std::error_code error;
    auto out = llvm::raw_fd_ostream(filename, error);
    if (error)
      return runtimeError("Cannot open output filename");

    // We're not optimising the MLIR module like we do for LLVM output
    // because this is mostly for debug and testing. We could do that
    // in the future.
    module->print(out);
    return llvm::Error::success();
  }

  // FIXME: This function will not work. It must receive an MLIR module that is
  // 100% composed of dialects that can be fully converted to LLVM dialect.
  // We keep this code here as future reference on how to lower to LLVM.
  llvm::Error Generator::emitLLVM(llvm::StringRef filename, unsigned optLevel)
  {
    if (filename.empty())
      return runtimeError("No output filename provided");

    // The lowering "pass manager"
    mlir::PassManager pm(&context);
    if (optLevel > 0)
    {
      pm.addPass(mlir::createInlinerPass());
      pm.addPass(mlir::createSymbolDCEPass());
      mlir::OpPassManager& optPM = pm.nest<mlir::FuncOp>();
      optPM.addPass(mlir::createCanonicalizerPass());
      optPM.addPass(mlir::createCSEPass());
    }
    pm.addPass(mlir::createLowerToLLVMPass());

    // First lower to LLVM dialect
    if (mlir::failed(pm.run(module.get())))
    {
      module->dump();
      return runtimeError("Failed to lower to LLVM dialect");
    }

    // Then lower to LLVM IR
    auto llvm = mlir::translateModuleToLLVMIR(module.get());
    if (!llvm)
      return runtimeError("Failed to lower to LLVM IR");

    // Write to the file requested
    std::error_code error;
    auto out = llvm::raw_fd_ostream(filename, error);
    if (error)
      return runtimeError("Cannot open output filename");

    llvm->print(out, nullptr);
    return llvm::Error::success();
  }

  // ===================================================== AST -> MLIR
  mlir::Location Generator::getLocation(const ::ast::Ast& ast)
  {
    auto path = getPath(ast);
    return builder.getFileLineColLoc(
      Identifier::get(path.file, &context), path.line, path.column);
  }

  llvm::Error Generator::parseModule(const ::ast::Ast& ast)
  {
    assert(isClass(ast) && "Bad node");
    module = mlir::ModuleOp::create(getLocation(ast));
    // TODO: Support more than just functions at the module level
    auto body = getClassBody(ast);
    for (auto f : getSubNodes(body))
    {
      auto fun = parseFunction(f.lock());
      if (auto err = fun.takeError())
        return err;
      module->push_back(*fun);
    }
    return llvm::Error::success();
  }

  llvm::Expected<mlir::FuncOp> Generator::parseProto(const ::ast::Ast& ast)
  {
    assert(isFunction(ast) && "Bad node");
    auto name = getFunctionName(ast);
    if (functionTable.inScope(name))
      return functionTable.lookup(name);

    // Parse 'where' clause
    auto constraints = getFunctionConstraints(ast);
    for (auto c : constraints)
    {
      // This is wrong. Constraints are not aliases, but with
      // the oversimplified representaiton we have and the fluid
      // state of the type system, this will "work" for now.
      auto alias = getID(c);
      auto ty = getType(c);
      typeTable.insert(alias, parseType(ty.lock()));
    }

    // Function type from signature
    Types types;
    auto args = getFunctionArgs(ast);
    for (auto arg : args)
      types.push_back(parseType(getType(arg).lock()));
    auto retTy = parseType(getFunctionType(ast).lock());
    auto funcTy = builder.getFunctionType(types, retTy);

    // Create function
    auto func = mlir::FuncOp::create(getLocation(ast), name, funcTy);
    functionTable.insert(name, func);
    return func;
  }

  llvm::Expected<mlir::FuncOp> Generator::parseFunction(const ::ast::Ast& ast)
  {
    assert(isFunction(ast) && "Bad node");

    // Declare function signature
    TypeScopeT alias_scope(typeTable);
    auto name = getFunctionName(ast);
    auto proto = parseProto(ast);
    if (auto err = proto.takeError())
      return std::move(err);
    auto& func = *proto;
    auto retTy = func.getType().getResult(0);

    // Create entry block
    auto& entryBlock = *func.addEntryBlock();
    builder.setInsertionPointToStart(&entryBlock);

    // Declare all arguments on current scope
    SymbolScopeT var_scope(symbolTable);
    auto args = getFunctionArgs(ast);
    auto argVals = entryBlock.getArguments();
    assert(args.size() == argVals.size() && "Argument mismatch");
    for (auto var_val : llvm::zip(args, argVals))
    {
      auto name = getID(std::get<0>(var_val).lock());
      auto value = std::get<1>(var_val);
      declareVariable(name, value);
    }

    // Lower body
    auto body = getFunctionBody(ast);
    auto last = parseNode(body.lock());
    if (auto err = last.takeError())
      return std::move(err);

    // Return last value
    if (*last && last->getType() != retTy)
    {
      // Cast type (we trust the ast)
      last = genOperation(last->getLoc(), "verona.cast", {*last}, retTy);
    }
    else
    {
      last = genOperation(getLocation(ast), "verona.none", {}, retTy);
    }
    if (auto err = last.takeError())
      return std::move(err);
    builder.create<mlir::ReturnOp>(getLocation(ast), *last);

    return func;
  }

  mlir::Type Generator::parseType(const ::ast::Ast& ast)
  {
    assert(isType(ast) && "Bad node");
    auto desc = getTypeDesc(ast);
    if (desc.empty())
      return builder.getNoneType();

    // If type is in the alias table, get it
    if (typeTable.inScope(desc))
      return typeTable.lookup(desc);

    // Else, insert into the table and return
    auto type = genOpaqueType(desc, context);
    typeTable.insert(desc, type);
    return type;
  }

  void Generator::declareVariable(llvm::StringRef name, mlir::Value val)
  {
    assert(!symbolTable.inScope(name) && "Redeclaration");
    symbolTable.insert(name, val);
  }

  void Generator::updateVariable(llvm::StringRef name, mlir::Value val)
  {
    assert(symbolTable.inScope(name) && "Variable not declared");
    symbolTable.update(name, val);
  }

  llvm::Expected<mlir::Value> Generator::parseBlock(const ::ast::Ast& ast)
  {
    mlir::Value last;
    for (auto sub : getSubNodes(ast))
    {
      auto node = parseNode(sub.lock());
      if (auto err = node.takeError())
        return std::move(err);
      last = *node;
    }
    return last;
  }

  llvm::Expected<mlir::Value> Generator::parseNode(const ::ast::Ast& ast)
  {
    if (isValue(ast))
      return parseValue(ast);

    switch (getKind(ast))
    {
      case NodeKind::Localref:
        return parseValue(ast);
      case NodeKind::Block:
        return parseBlock(ast);
      case NodeKind::ID:
        return parseValue(ast);
      case NodeKind::Assign:
        return parseAssign(ast);
      case NodeKind::Call:
        return parseCall(ast);
      default:
        return parsingError(
          "Node " + getName(ast) + " not implemented yet", getLocation(ast));
    }
  }

  llvm::Expected<mlir::Value> Generator::parseValue(const ::ast::Ast& ast)
  {
    // Variables
    if (isLocalRef(ast))
    {
      // We use allocas to track location and load/stores to track access
      auto var = symbolTable.lookup(getTokenValue(ast));
      if (var.getType() == allocaTy)
        return genOperation(getLocation(ast), "verona.load", {var}, unkTy);
      return var;
    }

    // Constants
    if (isConstant(ast))
    {
      // We lower each constant to their own values for now as we
      // don't yet have a good scheme for the types and MLIR can't
      // have attributes from unknown types. Once we set on a type
      // system compatibility between Verona and MLIR, we can change
      // this to emit the attribute right away.
      auto type = genOpaqueType(getName(ast), context);
      auto value = getTokenValue(ast);
      return genOperation(
        getLocation(ast), "verona.constant(" + value + ")", {}, type);
    }

    // TODO: Literals need attributes and types
    assert(isValue(ast) && "Bad node");
    return parsingError(
      "Value [" + getName(ast) + " = " + getTokenValue(ast) +
        "] not implemented yet",
      getLocation(ast));
  }

  llvm::Expected<mlir::Value> Generator::parseAssign(const ::ast::Ast& ast)
  {
    assert(isAssign(ast) && "Bad node");

    // Must be a Let declaring a variable (for now).
    auto let = getLHS(ast);
    auto name = getLocalName(let);

    // If the variable wasn't declared yet, create an alloca
    if (!symbolTable.inScope(name))
    {
      auto alloca = genOperation(
        getLocation(ast), "verona.alloca", {}, allocaTy);
      if (auto err = alloca.takeError())
        return std::move(err);
      declareVariable(name, *alloca);
    }
    auto store = symbolTable.lookup(name);

    // The right-hand side can be any expression
    // This is the value and we update the variable
    auto rhs = parseNode(getRHS(ast).lock());
    if (auto err = rhs.takeError())
      return std::move(err);

    // Store the value in the alloca
    auto op = genOperation(
      getLocation(ast), "verona.store", {*rhs, store}, unkTy);
    if (auto err = op.takeError())
      return std::move(err);
    return store;
  }

  llvm::Expected<mlir::Value> Generator::parseCall(const ::ast::Ast& ast)
  {
    assert(isCall(ast) && "Bad node");
    auto name = getID(ast);

    // All operations are calls, only calls to previously defined functions
    // are function calls. FIXME: Is this really what we want?
    if (functionTable.inScope(name))
    {
      // TODO: Lower calls.
      return parsingError(
        "Function calls not implemented yet", getLocation(ast));
    }

    // Else, it should be an operation that we can lower natively
    if (isUnary(ast))
    {
      return parsingError(
        "Unary Operation '" + name + "' not implemented yet", getLocation(ast));
    }
    else if (isBinary(ast))
    {
      // Get bpth arguments
      auto arg0 = parseNode(getOperand(ast, 0).lock());
      if (auto err = arg0.takeError())
        return std::move(err);
      auto arg1 = parseNode(getOperand(ast, 1).lock());
      if (auto err = arg1.takeError())
        return std::move(err);
      if (name == "+")
      {
        return genOperation(
          getLocation(ast), "verona.add", {*arg0, *arg1}, unkTy);
      }
      else if (name == "-")
      {
        return genOperation(
          getLocation(ast), "verona.sub", {*arg0, *arg1}, unkTy);
      }
      else if (name == "*")
      {
        return genOperation(
          getLocation(ast), "verona.mul", {*arg0, *arg1}, unkTy);
      }
      else if (name == "/")
      {
        return genOperation(
          getLocation(ast), "verona.div", {*arg0, *arg1}, unkTy);
      }

      if (name == "==")
      {
        return genOperation(
          getLocation(ast), "verona.eq", {*arg0, *arg1}, boolTy);
      }
      else if (name == "!=")
      {
        return genOperation(
          getLocation(ast), "verona.ne", {*arg0, *arg1}, boolTy);
      }
      else if (name == ">")
      {
        return genOperation(
          getLocation(ast), "verona.gt", {*arg0, *arg1}, boolTy);
      }
      else if (name == "<")
      {
        return genOperation(
          getLocation(ast), "verona.lt", {*arg0, *arg1}, boolTy);
      }
      else if (name == ">=")
      {
        return genOperation(
          getLocation(ast), "verona.ge", {*arg0, *arg1}, boolTy);
      }
      else if (name == "<=")
      {
        return genOperation(
          getLocation(ast), "verona.le", {*arg0, *arg1}, boolTy);
      }
    }

    return parsingError(
      "Operation '" + name + "' not implemented yet", getLocation(ast));
  }

  llvm::Expected<mlir::Value> Generator::genOperation(
    mlir::Location loc,
    llvm::StringRef name,
    llvm::ArrayRef<mlir::Value> ops,
    mlir::Type retTy)
  {
    auto opName = OperationName(name, &context);
    auto state = OperationState(loc, opName);
    state.addOperands(ops);
    state.addTypes({retTy});
    auto op = builder.createOperation(state);
    return op->getResult(0);
  }

  mlir::OpaqueType
  Generator::genOpaqueType(llvm::StringRef name, mlir::MLIRContext& context)
  {
    auto dialect = mlir::Identifier::get("type", &context);
    return mlir::OpaqueType::get(dialect, name, &context);
  }
}
