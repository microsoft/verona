// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "generator.h"

#include "abi.h"
#include "ast-utils.h"
#include "dialect/VeronaDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"

#include "llvm/ADT/StringSwitch.h"

namespace
{
  /// Helper to make sure the basic block has a terminator
  bool hasTerminator(mlir::Block* bb)
  {
    return !bb->getOperations().empty() && bb->back().isKnownTerminator();
  }

  /// Add a new basic block into a region and return it
  mlir::Block* addBlock(mlir::Region* region)
  {
    region->push_back(new mlir::Block());
    return &region->back();
  }
}

namespace mlir::verona
{
  // ===================================================== Public Interface
  llvm::Expected<mlir::OwningModuleRef>
  Generator::lower(MLIRContext* context, const ::ast::Ast& ast)
  {
    Generator gen(context);
    if (auto err = gen.parseModule(ast))
      return std::move(err);

    return std::move(gen.module);
  }

  // ===================================================== Helpers
  mlir::Location Generator::getLocation(const ::ast::Ast& ast)
  {
    auto path = AST::getPath(ast);
    return builder.getFileLineColLoc(
      Identifier::get(path.file, context), path.line, path.column);
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

  void Generator::declareFunction(
    llvm::StringRef name,
    llvm::ArrayRef<llvm::StringRef> types,
    llvm::StringRef retTy)
  {
    // If already declared, ignore
    if (functionTable.inScope(name))
      return;

    // Map type names to MLIR types
    Types argTys;
    for (auto a : types)
      argTys.push_back(genOpaqueType(a));
    llvm::SmallVector<mlir::Type, 1> retTys;
    if (!retTy.empty())
      retTys.push_back(genOpaqueType(retTy));

    // Generate the function and check: this should never fail
    auto func = generateProto(unkLoc, name, argTys, retTys);
    if (auto err = func.takeError())
      assert(false && "FIXME: broken function generator");

    // Add function declaration to the module
    module->push_back(*func);
  }

  // ===================================================== AST -> MLIR
  llvm::Error Generator::parseModule(const ::ast::Ast& ast)
  {
    assert(AST::isClass(ast) && "Bad node");
    module = mlir::ModuleOp::create(getLocation(ast));
    // TODO: Support more than just functions at the module level
    auto body = AST::getClassBody(ast);
    llvm::SmallVector<::ast::WeakAst, 4> funcs;
    AST::getSubNodes(funcs, body);
    for (auto f : funcs)
    {
      auto fun = parseFunction(f.lock());
      if (auto err = fun.takeError())
        return err;
      module->push_back(*fun);
    }
    return llvm::Error::success();
  }

  llvm::Expected<mlir::FuncOp> Generator::parseFunction(const ::ast::Ast& ast)
  {
    assert(AST::isFunction(ast) && "Bad node");

    // Parse 'where' clause
    TypeScopeT alias_scope(typeTable);
    llvm::SmallVector<::ast::WeakAst, 4> constraints;
    AST::getFunctionConstraints(constraints, ast);
    for (auto c : constraints)
    {
      // This is wrong. Constraints are not aliases, but with
      // the oversimplified representaiton we have and the fluid
      // state of the type system, this will "work" for now.
      auto alias = AST::getID(c);
      auto ty = AST::getType(c);
      typeTable.insert(alias, parseType(ty.lock()));
    }

    // Function type from signature
    llvm::SmallVector<llvm::StringRef, 4> argNames;
    Types types;
    llvm::SmallVector<::ast::WeakAst, 4> args;
    AST::getFunctionArgs(args, ast);
    for (auto arg : args)
    {
      argNames.push_back(AST::getID(arg));
      types.push_back(parseType(AST::getType(arg).lock()));
    }

    // Return type is nothing if no type
    llvm::SmallVector<mlir::Type, 1> retTy;
    if (AST::hasType(AST::getFunctionType(ast)))
      retTy.push_back(parseType(AST::getFunctionType(ast).lock()));

    // If just declaration, return the proto value
    auto name = AST::getFunctionName(ast);
    if (!AST::hasFunctionBody(ast))
    {
      // Declare function signature
      auto proto = generateProto(getLocation(ast), name, types, retTy);
      if (auto err = proto.takeError())
        return std::move(err);
      auto& func = *proto;
      return func;
    }

    // Declare all arguments on current scope
    SymbolScopeT var_scope(symbolTable);
    auto def =
      generateEmptyFunction(getLocation(ast), name, argNames, types, retTy);
    if (auto err = def.takeError())
      return std::move(err);
    auto& func = *def;

    // Lower body
    auto body = AST::getFunctionBody(ast);
    auto last = parseNode(body.lock());
    if (auto err = last.takeError())
      return std::move(err);

    // Check if needs to return a value at all
    if (hasTerminator(builder.getBlock()))
      return func;

    // Return last value (or none)
    // TODO: Implement multiple return values for tuples
    bool hasLast = last->hasValue();
    bool hasRetTy = !retTy.empty();
    if (hasLast && hasRetTy)
    {
      // Function has return value and there is a last value,
      // check types, cast if not the same, return.
      auto cast = generateAutoCast(last->get().getLoc(), last->get(), retTy[0]);
      builder.create<mlir::ReturnOp>(getLocation(ast), cast);
    }
    else if (!hasRetTy)
    {
      // Function return value is void, ignore last value and return.
      builder.create<mlir::ReturnOp>(getLocation(ast));
    }
    else
    {
      // Has return type but no value, emit an error.
      return parsingError("Function has no value to return", getLocation(ast));
    }

    return func;
  }

  mlir::Type Generator::parseType(const ::ast::Ast& ast)
  {
    assert(AST::isType(ast) && "Bad node");
    auto desc = AST::getTypeDesc(ast);
    if (desc.empty())
      return unkTy;

    // If type is in the alias table, get it
    auto type = typeTable.lookup(desc);
    if (type)
      return type;

    // Special treatment for type name 'bool'
    // TODO: Type matching when we have proper Verona types
    if (desc == "bool")
      return boolTy;

    // Else, insert into the table and return
    type = genOpaqueType(desc);
    typeTable.insert(desc, type);
    return type;
  }

  llvm::Expected<ReturnValue> Generator::parseBlock(const ::ast::Ast& ast)
  {
    ReturnValue last;
    llvm::SmallVector<::ast::WeakAst, 4> nodes;
    AST::getSubNodes(nodes, ast);
    for (auto sub : nodes)
    {
      auto node = parseNode(sub.lock());
      if (auto err = node.takeError())
        return std::move(err);
      last = *node;
    }
    return last;
  }

  llvm::Expected<ReturnValue> Generator::parseNode(const ::ast::Ast& ast)
  {
    switch (AST::getKind(ast))
    {
      case AST::NodeKind::Block:
      case AST::NodeKind::Seq:
        return parseBlock(ast);
      case AST::NodeKind::Assign:
        return parseAssign(ast);
      case AST::NodeKind::Call:
        return parseCall(ast);
      case AST::NodeKind::Return:
        return parseReturn(ast);
      case AST::NodeKind::If:
        return parseCondition(ast);
      case AST::NodeKind::While:
        return parseWhileLoop(ast);
      case AST::NodeKind::For:
        return parseForLoop(ast);
      case AST::NodeKind::Continue:
        return parseContinue(ast);
      case AST::NodeKind::Break:
        return parseBreak(ast);
    }

    if (AST::isValue(ast))
      return parseValue(ast);

    return parsingError(
      "Node " + AST::getName(ast) + " not implemented yet", getLocation(ast));
  }

  llvm::Expected<ReturnValue> Generator::parseValue(const ::ast::Ast& ast)
  {
    // Variables
    if (AST::isLocalRef(ast))
    {
      // We use allocas to track location and load/stores to track access
      auto name = AST::getTokenValue(ast);
      auto var = symbolTable.lookup(name);
      assert(var && "Undeclared variable lookup, broken ast");
      if (var.getType() == allocaTy)
        return generateLoad(getLocation(ast), var);
      return var;
    }

    // Constants
    if (AST::isConstant(ast))
    {
      return generateConstant(
        getLocation(ast), AST::getTokenValue(ast), AST::getName(ast));
    }

    // TODO: Literals need attributes and types
    assert(AST::isValue(ast) && "Bad node");
    return parsingError(
      "Value [" + AST::getName(ast) + " = " + AST::getTokenValue(ast) +
        "] not implemented yet",
      getLocation(ast));
  }

  llvm::Expected<ReturnValue> Generator::parseAssign(const ::ast::Ast& ast)
  {
    assert(AST::isAssign(ast) && "Bad node");

    // Can either be a let (new variable) or localref (existing variable).
    auto var = AST::getLHS(ast);
    auto name = AST::getLocalName(var);

    // If the variable wasn't declared yet in this context, create an alloca
    // TODO: Implement declaration of tuples (multiple values)
    if (AST::isLet(var))
    {
      auto alloca = generateAlloca(getLocation(ast));
      declareVariable(name, alloca);
    }
    auto store = symbolTable.lookup(name);
    if (!store)
      return parsingError(
        "Variable " + name + " not declared before use",
        getLocation(var.lock()));

    // The right-hand side can be any expression
    // This is the value and we update the variable
    auto rhs = parseNode(AST::getRHS(ast).lock());
    if (auto err = rhs.takeError())
      return std::move(err);

    // Store the value in the alloca
    return generateStore(getLocation(ast), rhs->get(), store);
  }

  llvm::Expected<ReturnValue> Generator::parseCall(const ::ast::Ast& ast)
  {
    assert(AST::isCall(ast) && "Bad node");
    auto name = AST::getID(ast);

    // All operations are calls, only calls to previously defined functions
    // are function calls.
    if (auto func = functionTable.lookup(name))
    {
      llvm::SmallVector<::ast::WeakAst, 4> argNodes;
      AST::getAllOperands(argNodes, ast);
      auto argTypes = func.getType().getInputs();
      assert(argNodes.size() == argTypes.size() && "Wrong number of arguments");
      llvm::SmallVector<mlir::Value, 4> args;

      // For each argument / type, cast.
      for (const auto& val_ty : llvm::zip(argNodes, argTypes))
      {
        // Arguments lowered before the call
        auto arg = std::get<0>(val_ty).lock();
        auto val = parseNode(arg);
        if (auto err = val.takeError())
          return std::move(err);

        // Types could be incomplete here, casts may be needed
        auto argTy = std::get<1>(val_ty);
        auto cast = generateAutoCast(getLocation(arg), val->get(), argTy);
        args.push_back(cast);
      }

      auto call = builder.create<mlir::CallOp>(getLocation(ast), func, args);
      auto res = call.getResults();
      return res;
    }

    // Else, it should be an operation that we can lower natively
    if (AST::isUnary(ast))
    {
      return parsingError(
        "Unary Operation '" + name + "' not implemented yet", getLocation(ast));
    }
    else if (AST::isBinary(ast))
    {
      // Get both arguments
      // TODO: If the arguments are tuples, do we need to apply element-wise?
      auto arg0 = parseNode(AST::getOperand(ast, 0).lock());
      if (auto err = arg0.takeError())
        return std::move(err);
      auto arg1 = parseNode(AST::getOperand(ast, 1).lock());
      if (auto err = arg1.takeError())
        return std::move(err);

      // Get op name and type
      using opPairTy = std::pair<llvm::StringRef, mlir::Type>;
      opPairTy opTy = llvm::StringSwitch<opPairTy>(name)
                        .Case("+", {"verona.add", unkTy})
                        .Case("-", {"verona.sub", unkTy})
                        .Case("*", {"verona.mul", unkTy})
                        .Case("/", {"verona.div", unkTy})
                        .Case("==", {"verona.eq", boolTy})
                        .Case("!=", {"verona.ne", boolTy})
                        .Case(">", {"verona.gt", boolTy})
                        .Case("<", {"verona.lt", boolTy})
                        .Case(">=", {"verona.ge", boolTy})
                        .Case("<=", {"verona.le", boolTy})
                        .Default(std::make_pair("", unkTy));
      auto opName = opTy.first;
      auto opType = opTy.second;

      // Match, return the right op with the right type
      if (!opName.empty())
      {
        return genOperation(
          getLocation(ast), opName, {arg0->get(), arg1->get()}, opType);
      }

      return parsingError(
        "Binary operation '" + name + "' not implemented yet",
        getLocation(ast));
    }

    return parsingError(
      "Operation '" + name + "' not implemented yet", getLocation(ast));
  }

  llvm::Expected<ReturnValue> Generator::parseReturn(const ::ast::Ast& ast)
  {
    assert(AST::isReturn(ast) && "Bad node");

    // TODO: Implement returning multiple values
    ReturnValue expr;
    if (AST::hasSubs(ast))
    {
      auto node = parseNode(AST::getSingleSubNode(ast).lock());
      if (auto err = node.takeError())
        return std::move(err);
      expr = *node;
    }

    // Check which type of region we're in to
    // get the function return type (for casts)
    auto region = builder.getInsertionBlock()->getParent();
    mlir::Type retTy;
    auto op = region->getParentOp();
    while (!isa<mlir::FuncOp>(op))
    {
      op = op->getParentRegion()->getParentOp();
    }
    if (auto func = dyn_cast<mlir::FuncOp>(op))
    {
      // Void returns don't have results
      if (func.getType().getNumResults() > 0)
        retTy = func.getType().getResult(0);
    }
    else
    {
      return parsingError(
        "Return operation without parent function", getLocation(ast));
    }

    // Either returns empty or auto-cast'ed value
    if (expr.hasValue())
    {
      assert(retTy && "Return value from a void function");
      // Emit the cast if necessary (we trust the ast)
      auto cast = generateAutoCast(expr.get().getLoc(), expr.get(), retTy);
      builder.create<mlir::ReturnOp>(getLocation(ast), cast);
    }
    else
    {
      assert(!retTy && "Return void from a valued function");
      builder.create<mlir::ReturnOp>(getLocation(ast));
    }

    // No values to return, basic block is terminated.
    return ReturnValue();
  }

  llvm::Expected<ReturnValue> Generator::parseCondition(const ::ast::Ast& ast)
  {
    assert(AST::isIf(ast) && "Bad node");

    // TODO: MLIR doesn't support conditions with literals
    // we need to make a constexpr decision and only lower the right block
    if (AST::isConstant(AST::getCond(ast)))
    {
      return parsingError(
        "Conditionals with literals not supported yet", getLocation(ast));
    }

    // Create local context for condition variables (valid for both if/else)
    SymbolScopeT var_scope{symbolTable};

    // First node is a sequence of conditions
    // lower in the current basic block.
    auto condNode = AST::getCond(ast).lock();
    auto condLoc = getLocation(condNode);
    auto cond = parseNode(condNode);
    if (auto err = cond.takeError())
      return std::move(err);

    // Create basic-blocks, conditionally branch to if/else
    auto region = builder.getInsertionBlock()->getParent();
    mlir::ValueRange empty{};
    auto ifBB = addBlock(region);
    mlir::Block* elseBB = nullptr;
    if (AST::hasElse(ast))
      elseBB = addBlock(region);
    auto exitBB = addBlock(region);
    if (AST::hasElse(ast))
    {
      if (
        auto err =
          generateCondBranch(condLoc, cond->get(), ifBB, empty, elseBB, empty))
        return std::move(err);
    }
    else
    {
      if (
        auto err =
          generateCondBranch(condLoc, cond->get(), ifBB, empty, exitBB, empty))
        return std::move(err);
    }

    {
      // Create local context for the if block variables
      SymbolScopeT if_scope{symbolTable};

      // If block
      auto ifNode = AST::getIfBlock(ast).lock();
      auto ifLoc = getLocation(ifNode);
      builder.setInsertionPointToEnd(ifBB);
      auto ifBlock = parseNode(ifNode);
      if (auto err = ifBlock.takeError())
        return std::move(err);
      if (!hasTerminator(builder.getBlock()))
        builder.create<mlir::BranchOp>(ifLoc, exitBB, empty);
    }

    // Else block
    // We don't need to lower the else part if it's empty
    if (AST::hasElse(ast))
    {
      // Create local context for the else block variables
      SymbolScopeT else_scope{symbolTable};

      auto elseNode = AST::getElseBlock(ast).lock();
      auto elseLoc = getLocation(elseNode);
      builder.setInsertionPointToEnd(elseBB);
      auto elseBlock = parseNode(elseNode);
      if (auto err = elseBlock.takeError())
        return std::move(err);
      if (!hasTerminator(builder.getBlock()))
        builder.create<mlir::BranchOp>(elseLoc, exitBB, empty);
    }

    // Move to exit block, where the remaining instructions will be lowered.
    builder.setInsertionPointToEnd(exitBB);

    // No values to return from lexical constructs.
    return ReturnValue();
  }

  llvm::Expected<ReturnValue> Generator::parseWhileLoop(const ::ast::Ast& ast)
  {
    assert(AST::isWhile(ast) && "Bad node");

    // TODO: MLIR doesn't support conditions with literals
    // we need to make a constexpr decision and only lower the right block
    if (AST::isConstant(AST::getCond(ast)))
    {
      return parsingError(
        "Loop conditions with literals not supported yet", getLocation(ast));
    }

    // Create the head basic-block, which will check the condition
    // and dispatch the loop to the body block or exit.
    auto region = builder.getInsertionBlock()->getParent();
    mlir::ValueRange empty{};
    auto headBB = addBlock(region);
    auto bodyBB = addBlock(region);
    auto exitBB = addBlock(region);
    builder.create<mlir::BranchOp>(getLocation(ast), headBB, empty);

    // Create local context for loop variables
    SymbolScopeT var_scope{symbolTable};

    // First node is a sequence of conditions
    // lower in the head basic block, with the conditional branch.
    auto condNode = AST::getCond(ast).lock();
    auto condLoc = getLocation(condNode);
    builder.setInsertionPointToEnd(headBB);
    auto cond = parseNode(condNode);
    if (auto err = cond.takeError())
      return std::move(err);
    if (
      auto err =
        generateCondBranch(condLoc, cond->get(), bodyBB, empty, exitBB, empty))
      return std::move(err);

    // Create local head/tail basic-block context for continue/break
    BasicBlockScopeT loop_scope{loopTable};
    loopTable.insert("head", headBB);
    loopTable.insert("tail", exitBB);

    // Loop body, branch back to head node which will decide exit criteria
    auto bodyNode = AST::getLoopBlock(ast).lock();
    auto bodyLoc = getLocation(bodyNode);
    builder.setInsertionPointToEnd(bodyBB);
    auto bodyBlock = parseNode(bodyNode);
    if (auto err = bodyBlock.takeError())
      return std::move(err);
    if (!hasTerminator(builder.getBlock()))
      builder.create<mlir::BranchOp>(bodyLoc, headBB, empty);

    // Move to exit block, where the remaining instructions will be lowered.
    builder.setInsertionPointToEnd(exitBB);

    // No values to return from lexical constructs.
    return ReturnValue();
  }

  llvm::Expected<ReturnValue> Generator::parseForLoop(const ::ast::Ast& ast)
  {
    assert(AST::isFor(ast) && "Bad node");

    // For loops are of the shape (item in list), which need to initialise
    // the item, take the next from the list and exit if there is none.
    // All of that is within the scope of the loop, so we need to create a
    // scope now, to drop them from future ops.
    SymbolScopeT var_scope{symbolTable};

    // First, we identify the iterator. Nested loops will have their own
    // iterators, of the same name, but within their own lexical blocks.
    auto seqNode = AST::getLoopSeq(ast).lock();
    auto list = parseNode(seqNode);
    if (auto err = list.takeError())
      return std::move(err);
    declareVariable(ABI::LoopIterator::handler, list->get());
    llvm::SmallVector<mlir::Value, 1> iter{
      symbolTable.lookup(ABI::LoopIterator::handler)};

    // Create the head basic-block, which will check the condition
    // and dispatch the loop to the body block or exit.
    auto region = builder.getInsertionBlock()->getParent();
    mlir::ValueRange empty{};
    auto headBB = addBlock(region);
    auto bodyBB = addBlock(region);
    auto exitBB = addBlock(region);
    builder.create<mlir::BranchOp>(getLocation(ast), headBB, empty);

    // First node is a check if the list has value, returns boolean.
    declareFunction(
      ABI::LoopIterator::check::name,
      {ABI::LoopIterator::check::types[0]},
      ABI::LoopIterator::check::retTy);
    auto has_value = functionTable.lookup(ABI::LoopIterator::check::name);
    auto condLoc = getLocation(seqNode);
    builder.setInsertionPointToEnd(headBB);
    auto cond = builder.create<mlir::CallOp>(condLoc, has_value, iter);
    if (
      auto err = generateCondBranch(
        condLoc, cond.getResult(0), bodyBB, empty, exitBB, empty))
      return std::move(err);

    // Create local head/tail basic-block context for continue/break
    BasicBlockScopeT loop_scope{loopTable};
    loopTable.insert("head", headBB);
    loopTable.insert("tail", exitBB);

    // Preamble for the loop body is:
    //  val = $iter.apply();
    // val must have been declared in outer scope
    declareFunction(
      ABI::LoopIterator::apply::name,
      {ABI::LoopIterator::apply::types[0]},
      ABI::LoopIterator::apply::retTy);
    auto apply = functionTable.lookup(ABI::LoopIterator::apply::name);
    builder.setInsertionPointToEnd(bodyBB);
    auto indVarName = AST::getTokenValue(AST::getLoopInd(ast));
    auto value = builder.create<mlir::CallOp>(condLoc, apply, iter);
    declareVariable(indVarName, value.getResult(0));
    //  %iter.next();
    declareFunction(
      ABI::LoopIterator::next::name,
      {ABI::LoopIterator::next::types[0]},
      ABI::LoopIterator::next::retTy);
    auto next = functionTable.lookup(ABI::LoopIterator::next::name);
    builder.create<mlir::CallOp>(condLoc, next, iter);

    // Loop body, branch back to head node which will decide exit criteria
    auto bodyNode = AST::getLoopBlock(ast).lock();
    auto bodyLoc = getLocation(bodyNode);
    auto bodyBlock = parseNode(bodyNode);
    if (auto err = bodyBlock.takeError())
      return std::move(err);
    if (!hasTerminator(builder.getBlock()))
      builder.create<mlir::BranchOp>(bodyLoc, headBB, empty);

    // Move to exit block, where the remaining instructions will be lowered.
    builder.setInsertionPointToEnd(exitBB);

    // No values to return from lexical constructs.
    return ReturnValue();
  }

  llvm::Expected<ReturnValue> Generator::parseContinue(const ::ast::Ast& ast)
  {
    assert(AST::isContinue(ast) && "Bad node");
    // Nested loops have multiple heads, we only care about the last one
    if (auto err = generateLoopBranch(getLocation(ast), "head"))
      return std::move(err);

    // No values to return, basic block is terminated.
    return ReturnValue();
  }

  llvm::Expected<ReturnValue> Generator::parseBreak(const ::ast::Ast& ast)
  {
    assert(AST::isBreak(ast) && "Bad node");
    // Nested loops have multiple tails, we only care about the last one
    if (auto err = generateLoopBranch(getLocation(ast), "tail"))
      return std::move(err);

    // No values to return, basic block is terminated.
    return ReturnValue();
  }

  // ===================================================== MLIR Generators
  llvm::Expected<mlir::FuncOp> Generator::generateProto(
    mlir::Location loc,
    llvm::StringRef name,
    llvm::ArrayRef<mlir::Type> types,
    llvm::ArrayRef<mlir::Type> retTy)
  {
    assert(!functionTable.inScope(name) && "Duplicated function declaration");

    // Create function
    auto funcTy = builder.getFunctionType(types, retTy);
    auto func = mlir::FuncOp::create(loc, name, funcTy);
    functionTable.insert(name, func);
    return func;
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
    if (!functionTable.inScope(name))
    {
      auto proto = generateProto(loc, name, types, retTy);
      if (auto err = proto.takeError())
        return std::move(err);
      functionTable.insert(name, *proto);
    }
    auto func = functionTable.lookup(name);

    // Create entry block, set builder entry point
    auto& entryBlock = *func.addEntryBlock();
    auto argVals = entryBlock.getArguments();
    assert(args.size() == argVals.size() && "Argument/value mismatch");
    builder.setInsertionPointToStart(&entryBlock);

    // Declare all arguments
    for (auto var_val : llvm::zip(args, argVals))
    {
      // Get the argument name/value
      auto name = std::get<0>(var_val);
      auto value = std::get<1>(var_val);
      // Allocate space in the stack & store the argument value
      auto alloca = generateAlloca(loc);
      auto store = generateStore(loc, value, alloca);
      // Associate the name with the alloca SSA value
      declareVariable(name, alloca);
    }

    return func;
  }

  llvm::Error Generator::generateCondBranch(
    mlir::Location loc,
    mlir::Value cond,
    mlir::Block* ifBB,
    mlir::ValueRange ifArgs,
    mlir::Block* elseBB,
    mlir::ValueRange elseArgs)
  {
    // Cast to i1 if necessary
    auto cast = generateAutoCast(loc, cond, builder.getI1Type());
    builder.create<mlir::CondBranchOp>(
      loc, cast, ifBB, ifArgs, elseBB, elseArgs);
    return llvm::Error::success();
  }

  llvm::Error
  Generator::generateLoopBranch(mlir::Location loc, llvm::StringRef blockName)
  {
    if (!loopTable.inScope(blockName))
      return parsingError("Loop branch without a loop", loc);
    auto block = loopTable.lookup(blockName);
    mlir::ValueRange empty{};
    // We assume the branch is the last operation in its basic block
    // and that was checked by the parser
    builder.create<mlir::BranchOp>(loc, block, empty);
    return llvm::Error::success();
  }

  // ======================================================= Generator Helpers
  mlir::Value Generator::generateAutoCast(
    mlir::Location loc, mlir::Value value, mlir::Type type)
  {
    // No cast needed
    if (value.getType() == type)
      return value;

    // Cast needed
    return genOperation(value.getLoc(), "verona.cast", {value}, type);
  }

  mlir::Value Generator::generateConstant(
    mlir::Location loc, llvm::StringRef value, llvm::StringRef typeName)
  {
    // We lower each constant to their own values for now as we
    // don't yet have a good scheme for the types and MLIR can't
    // have attributes from unknown types. Once we set on a type
    // system compatibility between Verona and MLIR, we can change
    // this to emit the attribute right away.
    mlir::Type type = unkTy;
    if (!typeName.empty())
      type = genOpaqueType(typeName);
    return genOperation(loc, "verona.constant(" + value.str() + ")", {}, type);
  }

  mlir::Value
  Generator::generateAlloca(mlir::Location loc, llvm::StringRef typeName)
  {
    mlir::Type type = allocaTy;
    if (!typeName.empty())
      type = genOpaqueType(typeName);
    return genOperation(loc, "verona.alloca", {}, type);
  }

  mlir::Value Generator::generateLoad(mlir::Location loc, mlir::Value addr)
  {
    // TODO: Check if addr's type is a known pointer type and dereference
    // the type to use here instead of unkTy.
    return genOperation(loc, "verona.load", {addr}, unkTy);
  }

  mlir::Value Generator::generateStore(
    mlir::Location loc, mlir::Value value, mlir::Value addr)
  {
    // TODO: Check if addr's type is a known pointer type and dereference
    // the type to use here instead of unkTy.
    return genOperation(loc, "verona.store", {value, addr}, unkTy);
  }

  // =============================================================== Temporary
  mlir::Value Generator::genOperation(
    mlir::Location loc,
    llvm::StringRef name,
    llvm::ArrayRef<mlir::Value> ops,
    mlir::Type retTy)
  {
    auto opName = OperationName(name, context);
    auto state = OperationState(loc, opName);
    state.addOperands(ops);
    state.addTypes({retTy});
    auto op = builder.createOperation(state);
    auto res = op->getResult(0);
    return res;
  }

  mlir::OpaqueType Generator::genOpaqueType(llvm::StringRef name)
  {
    auto dialect = mlir::Identifier::get("type", context);
    return mlir::OpaqueType::get(dialect, name, context);
  }
}
