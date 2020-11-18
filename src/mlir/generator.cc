// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "generator.h"

#include "ast-utils.h"
#include "dialect/VeronaDialect.h"
#include "free-vars.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"

namespace
{
  /// Helper to make sure the basic block has a terminator
  bool hasTerminator(mlir::Block* bb)
  {
    return !bb->getOperations().empty() && bb->back().isKnownTerminator();
  }

  /// Add a new basic block into a region and return it
  mlir::Block* addBlock(
    mlir::Region* region, size_t numArgs = 0, mlir::Type type = mlir::Type())
  {
    region->push_back(new mlir::Block());
    auto block = &region->back();
    llvm::SmallVector<mlir::Type, 2> bbTypes(numArgs, type);
    mlir::TypeRange bbArgs(bbTypes);
    block->addArguments(bbArgs);
    return block;
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

  // ===================================================== AST -> MLIR
  llvm::Error Generator::parseModule(const ::ast::Ast& ast)
  {
    // Modules are nothing but global classes
    auto global = parseClass(ast);
    if (auto err = global.takeError())
      return err;
    module = *global;

    return llvm::Error::success();
  }

  llvm::Expected<ModuleOp>
  Generator::parseClass(const ::ast::Ast& ast, mlir::Type parent)
  {
    assert(AST::isClass(ast) && "Bad node");
    auto loc = getLocation(ast);

    // Push another scope for variables and functions
    SymbolScopeT var_scope(symbolTable);
    FunctionScopeT func_scope(functionTable);

    // Declare before building fields to allow for recursive declaration
    // If class is used in definitions before, it has been declared empty
    // already, so we use `update` to fetch it.
    auto name = AST::getID(ast);
    auto type = ClassType::get(context, name);
    typeTable.getOrAdd(name, type);

    // Creates the scope, each class/module is a new sub-module.
    auto scope = mlir::ModuleOp::create(getLocation(ast), name);

    // Nested classes, field names and types, methods, etc.
    llvm::SmallVector<::ast::Ast, 4> nodes;
    AST::getSubNodes(nodes, AST::getClassBody(ast));
    llvm::SmallVector<std::pair<StringRef, mlir::Type>, 4> fields;

    // Set the parent class
    if (parent)
      fields.push_back({"$parent", parent});

    // Scan all nodes for nested classes, fields and methods
    for (auto node : nodes)
    {
      if (AST::isClass(node))
      {
        // Recurse into nested classes
        auto classMod = parseClass(node, type);
        if (auto err = classMod.takeError())
          return std::move(err);
        // Push sub-class to scope
        scope.push_back(*classMod);
      }
      else if (AST::isField(node))
      {
        // Get field name/type for class type declaration
        auto fieldName = AST::getID(node);
        auto fieldType = parseType(AST::getType(node));
        fields.push_back({fieldName, fieldType});
      }
      else if (AST::isFunction(node))
      {
        // Methods
        auto func = parseFunction(node);
        if (auto err = func.takeError())
          return std::move(err);
        // Associate function with module (late mangling)
        func->setAttr("class", TypeAttr::get(type));
        // Add qualifiers as attributes
        llvm::SmallVector<::ast::Ast, 4> quals;
        AST::getFunctionQualifiers(quals, node);
        if (!quals.empty())
        {
          llvm::SmallVector<mlir::Attribute, 4> qualAttrs;
          for (auto qual : quals)
            qualAttrs.push_back(
              StringAttr::get(AST::getTokenValue(qual), context));
          func->setAttr("qualifiers", ArrayAttr::get(qualAttrs, context));
        }
        // Push function to scope
        scope.push_back(*func);
      }
      else
      {
        return parsingError(
          "Expecting field or function on class " + name.str(), loc);
      }
    }
    type.setFields(fields);

    return scope;
  }

  llvm::Expected<mlir::FuncOp> Generator::parseFunction(const ::ast::Ast& ast)
  {
    assert(AST::isFunction(ast) && "Bad node");

    // Runs the free variable analysis on the function to help build all
    // arguments and return values of each basic block inside it
    freeVars.runOnFunction(ast);

    // Parse 'where' clause
    TypeScopeT alias_scope(typeTable);
    llvm::SmallVector<::ast::Ast, 4> constraints;
    AST::getFunctionConstraints(constraints, ast);
    for (auto c : constraints)
    {
      // This is wrong. Constraints are not aliases, but with
      // the oversimplified representaiton we have and the fluid
      // state of the type system, this will "work" for now.
      auto alias = AST::getID(c);
      auto ty = AST::getType(c);
      typeTable.insert(alias, parseType(ty));
    }

    // Function type from signature
    llvm::SmallVector<llvm::StringRef, 4> argNames;
    Types types;
    llvm::SmallVector<::ast::Ast, 4> args;
    AST::getFunctionArgs(args, ast);
    for (auto arg : args)
    {
      argNames.push_back(AST::getID(arg));
      types.push_back(parseType(AST::getType(arg)));
    }

    // Return type is nothing if no type
    llvm::SmallVector<mlir::Type, 1> retTy;
    if (AST::hasType(AST::getFunctionType(ast)))
      retTy.push_back(parseType(AST::getFunctionType(ast)));

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
    auto last = parseNode(body);
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

  mlir::Type Generator::generateType(llvm::StringRef name)
  {
    // If already created, return symbol
    if (auto type = typeTable.lookup(name))
      return type;

    // Capabilities / boolean
    if (name == "iso")
      return typeTable.insert(name, getIso(context));
    else if (name == "mut")
      return typeTable.insert(name, getMut(context));
    else if (name == "imm")
      return typeTable.insert(name, getImm(context));

    // Every other type is just a class that we don't know yet
    return typeTable.insert(name, ClassType::get(context, name));
  }

  mlir::Type Generator::parseType(const ::ast::Ast& ast)
  {
    // Qualified types are references to classes
    if (AST::isQualType(ast))
      return generateType(AST::getID(ast));

    assert(AST::isTypeHolder(ast) && "Bad node");

    // Get type components
    char sep = 0;
    llvm::SmallVector<::ast::Ast, 1> nodes;
    AST::getTypeElements(ast, sep, nodes);

    // No types, return "unknown"
    if (nodes.size() == 0)
      return unkTy;

    // Simple types should work directly
    if (nodes.size() == 1)
      return generateType(AST::getID(nodes[0]));

    // Composite types (meet, join) may require recursion
    llvm::SmallVector<mlir::Type, 1> types;
    for (auto node : nodes)
    {
      // Recursive nodes
      if (AST::isTypeHolder(node))
      {
        types.push_back(parseType(node));
      }
      // Direct nodes
      else
      {
        types.push_back(generateType(AST::getID(node)));
      }
    }

    // Return set of nodes, no need to cache
    switch (sep)
    {
      case '|':
        return JoinType::get(context, types);
      case '&':
        return MeetType::get(context, types);
      default:
        assert(false && "Invalid type operation");
    }

    // TODO: We need a nicer fall back here, but the code should never get here
    llvm_unreachable("Unrecoverable error parsing types");
  }

  llvm::Expected<ReturnValue> Generator::parseBlock(const ::ast::Ast& ast)
  {
    // Blocks add lexical context
    SymbolScopeT scope{symbolTable};
    return parseSeq(ast);
  }

  llvm::Expected<ReturnValue> Generator::parseSeq(const ::ast::Ast& ast)
  {
    ReturnValue last;
    llvm::SmallVector<::ast::Ast, 4> nodes;
    AST::getSubNodes(nodes, ast);
    for (auto sub : nodes)
    {
      auto node = parseNode(sub);
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
        return parseBlock(ast);
      case AST::NodeKind::Seq:
        return parseSeq(ast);
      case AST::NodeKind::Assign:
        return parseAssign(ast);
      case AST::NodeKind::Call:
      case AST::NodeKind::Invoke:
      case AST::NodeKind::StaticCall:
        return parseCall(ast);
      case AST::NodeKind::Return:
        return parseReturn(ast);
      case AST::NodeKind::If:
        return parseCondition(ast);
      case AST::NodeKind::While:
        return parseWhileLoop(ast);
      case AST::NodeKind::Continue:
        return parseContinue(ast);
      case AST::NodeKind::Break:
        return parseBreak(ast);
      case AST::NodeKind::New:
        return parseNew(ast);
      case AST::NodeKind::Member:
        return parseFieldRead(ast);
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
      "Value [" + AST::getName(ast) + " = " + AST::getTokenValue(ast).str() +
        "] not implemented yet",
      getLocation(ast));
  }

  llvm::Expected<ReturnValue> Generator::parseAssign(const ::ast::Ast& ast)
  {
    assert(AST::isAssign(ast) && "Bad node");
    auto loc = getLocation(ast);

    // The right-hand side can be any expression
    // This is the value and we update the variable
    // We parse it first to lower field-write correctly
    auto rhs = parseNode(AST::getRHS(ast));
    if (auto err = rhs.takeError())
      return std::move(err);
    auto value = rhs->get();

    auto lhs = AST::getLHS(ast);
    // If the LHS is a field member, we have a Verona operation to represent
    // writing to a field without an explicit alloca/store.
    if (AST::isMember(lhs))
      return parseFieldWrite(lhs, value);

    // Else, it can either be a let (new variable)
    // or localref (existing variable).
    auto name = AST::getLocalName(lhs);

    // If the variable wasn't declared yet in this context
    if (AST::isLet(lhs))
    {
      // If type was declared, use it
      auto type = unkTy;
      auto declType = AST::getType(lhs);
      if (AST::hasType(declType))
      {
        type = parseType(declType);
        value = generateAutoCast(loc, value, type);
      }

      // Insert the value in the symbol table
      symbolTable.insert(name, value);

      // No previous value to return
      return ReturnValue();
    }

    // Other assigns (on existing variables) return the previous stored value
    // after updating the current value
    auto prev = symbolTable.lookup(name);
    symbolTable.update(name, value);
    return prev;
  }

  llvm::Expected<ReturnValue> Generator::parseCall(const ::ast::Ast& ast)
  {
    // All operations are calls, including arithmetic, comparison, casts
    // but they can be: `invoke`, `call` or `static-call`.
    assert(
      (AST::isCall(ast) || AST::isInvoke(ast) || AST::isStaticCall(ast)) &&
      "Bad node");
    auto name = AST::getID(ast);
    auto loc = getLocation(ast);

    // Get arguments
    llvm::SmallVector<::ast::Ast, 1> nodes;
    AST::getAllOperands(nodes, ast);
    llvm::SmallVector<mlir::Value, 1> args;
    for (auto node : nodes)
    {
      auto arg = parseNode(node);
      if (auto err = arg.takeError())
        return std::move(err);
      args.push_back(arg->get());
    }

    // Some calls are lowered as special nodes, do those first
    if (name == "tidy")
    {
      assert(args.size() == 1 && "Wrong number of arguments for tidy");
      builder.create<TidyOp>(getLocation(ast), args[0]);
      return ReturnValue();
    }
    else if (name == "drop")
    {
      assert(args.size() == 1 && "Wrong number of arguments for drop");
      builder.create<DropOp>(getLocation(ast), args[0]);
      return ReturnValue();
    }

    // Some function calls are static (to global functions or static members),
    // others are dynamic (with an instance of a class as the first or left
    // argument.
    // Both need a descriptor, to know where to find the function to call.
    mlir::Value descriptor;
    if (AST::isCall(ast) || AST::isInvoke(ast))
    {
      // Dynamic call: `a op b` | `a.op(b...)`
      assert(args.size() >= 1 && "Too few arguments for dynamic call");
      descriptor = args[0];
      args.erase(args.begin());
    }
    else
    {
      // Static call: `func(a, b...)` | `Class.op(a, b...)`
      auto qualType = AST::getStaticQualType(ast);
      auto type = parseType(qualType);
      auto descTy = DescriptorType::get(context, type);
      descriptor = builder.create<StaticOp>(loc, descTy, TypeAttr::get(type));
    }

    // Right now, we may not have enough type information to know which is which
    // and how many arguments there are or which types are involved. When in
    // doubt, use `unknown`.
    ValueRange argVals{args};
    auto call = builder.create<CallOp>(
      loc, unkTy, descriptor, StringAttr::get(name, context), argVals);
    return call.res();
  }

  llvm::Expected<ReturnValue> Generator::parseReturn(const ::ast::Ast& ast)
  {
    assert(AST::isReturn(ast) && "Bad node");

    // TODO: Implement returning multiple values
    ReturnValue expr;
    if (AST::hasSubs(ast))
    {
      auto node = parseNode(AST::getSingleSubNode(ast));
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

    // Get PHI nodes from block
    auto args = freeVars.getArguments(ast);
    // Place holder for arguments gathered from scope before each branch.
    llvm::SmallVector<mlir::Value, 2> vars;

    // Create local context for condition variables (valid for both if/else)
    // We use pushScope directly because we want to drop the scope before we
    // update the block arguments as the new values (PHI nodes).
    symbolTable.pushScope();

    // First node is a sequence of conditions
    // lower in the current basic block.
    auto condNode = AST::getCond(ast);
    auto condLoc = getLocation(condNode);
    auto cond = parseNode(condNode);
    if (auto err = cond.takeError())
      return std::move(err);
    // Cast to i1 if necessary
    auto castCond = generateAutoCast(condLoc, cond->get(), builder.getI1Type());

    // Neither if/else blocks have arguments.
    mlir::ValueRange empty{};

    // Create basic-blocks, conditionally branch to if/else
    auto region = builder.getInsertionBlock()->getParent();
    auto ifBB = addBlock(region);
    mlir::Block* elseBB = nullptr;
    if (AST::hasElse(ast))
      elseBB = addBlock(region);

    // We only need to add arguments to the last block due to writes on either
    // if or else blocks. The reads on both if and else blocks work out from
    // variable scope and basic-block dominance.
    auto exitBB = addBlock(region, args.size(), unkTy);
    if (AST::hasElse(ast))
    {
      builder.create<mlir::CondBranchOp>(
        condLoc, castCond, ifBB, empty, elseBB, empty);
    }
    else
    {
      // From here, exitArgs are the unmodified values (pre-condition)
      generateBBArgList(condLoc, args, vars);
      mlir::ValueRange condArgs{vars};
      builder.create<mlir::CondBranchOp>(
        condLoc, castCond, ifBB, empty, exitBB, condArgs);
    }

    {
      // Create local context for the if block variables
      SymbolScopeT if_scope{symbolTable};

      // If block (skip the block to avoid creating a new scope).
      auto ifNode = AST::skipBlock(AST::getIfBlock(ast));
      auto ifLoc = getLocation(ifNode);
      builder.setInsertionPointToEnd(ifBB);
      auto ifBlock = parseNode(ifNode);
      if (auto err = ifBlock.takeError())
        return std::move(err);

      // Branch to exit if not returned yet
      if (!hasTerminator(builder.getBlock()))
      {
        // Recreate exit arguments (from local context of all modified
        // variables)
        generateBBArgList(condLoc, args, vars);
        mlir::ValueRange ifArgs{vars};
        builder.create<mlir::BranchOp>(ifLoc, exitBB, ifArgs);
      }
    }

    // Else block
    // We don't need to lower the else part if it's empty
    if (AST::hasElse(ast))
    {
      // Create local context for the else block variables
      SymbolScopeT else_scope{symbolTable};

      // Else block (skip the block to avoid creating a new scope).
      auto elseNode = AST::skipBlock(AST::getElseBlock(ast));
      auto elseLoc = getLocation(elseNode);
      builder.setInsertionPointToEnd(elseBB);
      auto elseBlock = parseNode(elseNode);
      if (auto err = elseBlock.takeError())
        return std::move(err);

      // Branch to exit if not returned yet
      if (!hasTerminator(builder.getBlock()))
      {
        // Recreate exit arguments (from local context of all modified
        // variables)
        generateBBArgList(condLoc, args, vars);
        mlir::ValueRange elseArgs{vars};
        builder.create<mlir::BranchOp>(elseLoc, exitBB, elseArgs);
      }
    }

    // Move to exit block, where the remaining instructions will be lowered.
    builder.setInsertionPointToEnd(exitBB);

    // Pop the variable scope, to update the values from the exit block's
    // arguments (PHI nodes)
    symbolTable.popScope();
    updateSymbolTable(args, exitBB->getArguments());

    // No values to return from lexical blocks.
    return ReturnValue();
  }

  llvm::Expected<ReturnValue> Generator::parseWhileLoop(const ::ast::Ast& ast)
  {
    assert(AST::isLoop(ast) && "Bad node");
    auto loc = getLocation(ast);

    // Get PHI nodes from block
    auto args = freeVars.getArguments(ast);

    // All blocks have the same number of arguments. This is an over
    // simplification that can be optimised later. Symbols declared in the head
    // or body blocks don't survive into the exit block, and those declared in
    // the body block don't propagate to the head block through the back edge.
    // But those declared before or in the head body are visible, through
    // dominance, in the body block.
    llvm::SmallVector<mlir::Value, 2> vars;
    generateBBArgList(loc, args, vars);
    mlir::ValueRange entryArgs{vars};

    // Create the head basic-block, which will check the condition
    // and dispatch the loop to the body block or exit.
    auto region = builder.getInsertionBlock()->getParent();
    auto headBB = addBlock(region, args.size(), unkTy);
    auto bodyBB = addBlock(region, args.size(), unkTy);
    auto exitBB = addBlock(region, args.size(), unkTy);
    builder.create<mlir::BranchOp>(getLocation(ast), headBB, entryArgs);

    // Create local context for loop variables
    // We use pushScope directly because we want to drop the scope before we
    // update the block arguments as the new values (PHI nodes).
    symbolTable.pushScope();

    // Update symbol table with basic block argument
    builder.setInsertionPointToEnd(headBB);
    updateSymbolTable(args, headBB->getArguments());

    // First node is a sequence of conditions
    // lower in the head basic block, with the conditional branch.
    auto condNode = AST::getCond(ast);
    auto condLoc = getLocation(condNode);
    auto cond = parseNode(condNode);
    if (auto err = cond.takeError())
      return std::move(err);
    // Cast to i1 if necessary
    auto castCond = generateAutoCast(condLoc, cond->get(), builder.getI1Type());

    // Re-generate arguments, as condition may have changed something.
    generateBBArgList(condLoc, args, vars);
    mlir::ValueRange condArgs{vars};
    builder.create<mlir::CondBranchOp>(
      condLoc, castCond, bodyBB, condArgs, exitBB, condArgs);

    // Create local head/tail basic-block context for continue/break
    LoopFlowControl fc = {headBB, exitBB, args};
    loopScope.push(fc);

    // Update symbol table with basic block argument
    builder.setInsertionPointToEnd(bodyBB);
    updateSymbolTable(args, bodyBB->getArguments());

    // Loop body, branch back to head node which will decide exit criteria
    // Skip the block to avoid creating a new scope.
    auto bodyNode = AST::skipBlock(AST::getLoopBlock(ast));
    auto bodyLoc = getLocation(bodyNode);
    auto bodyBlock = parseNode(bodyNode);
    if (auto err = bodyBlock.takeError())
      return std::move(err);

    // No explicit terminator means back-edge to head block.
    if (!hasTerminator(builder.getBlock()))
    {
      // Re-generate arguments, as condition may have changed something.
      generateBBArgList(condLoc, args, vars);
      mlir::ValueRange bodyArgs{vars};
      builder.create<mlir::BranchOp>(bodyLoc, headBB, bodyArgs);
    }

    // Pop local loop context (break/continue/args)
    loopScope.pop();

    // Pop the variable scope, to update the values (PHI nodes)
    symbolTable.popScope();

    // Move to exit block, where the remaining instructions will be lowered
    // and rerurn the updated values from the exitBB's arguments.
    builder.setInsertionPointToEnd(exitBB);
    updateSymbolTable(args, exitBB->getArguments());

    // No values to return from lexical blocks.
    return ReturnValue();
  }

  llvm::Expected<ReturnValue> Generator::parseContinue(const ::ast::Ast& ast)
  {
    assert(AST::isContinue(ast) && "Bad node");
    auto loc = getLocation(ast);

    // Nested loops have multiple heads, we only care about the last one
    auto fc = loopScope.top();
    llvm::SmallVector<mlir::Value, 2> vars;
    generateBBArgList(loc, fc.args, vars);
    mlir::ValueRange args{vars};
    builder.create<mlir::BranchOp>(loc, fc.head, args);

    // No values to return, basic block is terminated.
    return ReturnValue();
  }

  llvm::Expected<ReturnValue> Generator::parseBreak(const ::ast::Ast& ast)
  {
    assert(AST::isBreak(ast) && "Bad node");
    auto loc = getLocation(ast);

    // Nested loops have multiple tails, we only care about the last one
    auto fc = loopScope.top();
    llvm::SmallVector<mlir::Value, 2> vars;
    generateBBArgList(loc, fc.args, vars);
    mlir::ValueRange args{vars};
    builder.create<mlir::BranchOp>(loc, fc.tail, args);

    // No values to return, basic block is terminated.
    return ReturnValue();
  }

  llvm::Expected<ReturnValue> Generator::parseNew(const ::ast::Ast& ast)
  {
    assert(AST::isNew(ast) && "Bad node");
    auto loc = getLocation(ast);

    // Class name
    auto name = AST::getID(AST::getClassTypeRef(ast));
    auto nameAttr = SymbolRefAttr::get(name, context);

    // Type to allocate
    auto type = parseType(ast);

    // Initializer list
    llvm::SmallVector<::ast::Ast, 4> nodes;
    AST::getSubNodes(nodes, AST::getClassBody(ast));
    llvm::SmallVector<mlir::Attribute, 1> fieldNames;
    llvm::SmallVector<mlir::Value, 1> fieldValues;
    for (auto node : nodes)
    {
      if (!AST::isField(node))
        continue;
      fieldNames.push_back(StringAttr::get(AST::getID(node), context));
      auto expr = parseNode(AST::getInitExpr(node));
      if (auto err = expr.takeError())
        return std::move(err);
      fieldValues.push_back(expr->get());
    }
    auto fieldNameAttr = ArrayAttr::get(fieldNames, context);
    ValueRange inits{fieldValues};

    if (AST::hasInRegion(ast))
    {
      // If there's an `inreg`, allocate object on existing region
      auto regionName = AST::getID(AST::getInRegion(ast));
      auto regionObj = symbolTable.lookup(regionName);
      auto alloc = builder.create<AllocateObjectOp>(
        loc, type, nameAttr, fieldNameAttr, inits, regionObj);
      return alloc.getResult();
    }
    else
    {
      // If not, allocate a new region
      auto alloc = builder.create<AllocateRegionOp>(
        loc, type, nameAttr, fieldNameAttr, inits);
      return alloc.getResult();
    }
  }

  llvm::Expected<ReturnValue> Generator::parseFieldRead(const ::ast::Ast& ast)
  {
    assert(AST::isMember(ast) && "Bad node");

    // Find the variable to extract from
    auto ref = AST::getLocalRef(ast);
    auto var = symbolTable.lookup(ref);

    // Get the field name
    auto field = AST::getID(ast);

    // Find the field type, if any
    auto fieldType = unkTy;
    if (auto classType = var.getType().dyn_cast<ClassType>())
      fieldType = classType.getFieldType(field);

    // Return the output of the field read op
    auto loc = getLocation(ast);
    auto op = builder.create<FieldReadOp>(loc, fieldType, var, field);
    return op.getResult();
  }

  llvm::Expected<ReturnValue>
  Generator::parseFieldWrite(const ::ast::Ast& ast, mlir::Value value)
  {
    assert(AST::isMember(ast) && "Bad node");

    // Find the variable to extract from
    auto ref = AST::getLocalRef(ast);
    auto var = symbolTable.lookup(ref);

    // Get the field name
    auto field = AST::getID(ast);

    // Find the field type
    auto fieldType = unkTy;
    if (auto classType = var.getType().dyn_cast<ClassType>())
      fieldType = classType.getFieldType(field);

    // Make sure we have the same type
    auto loc = getLocation(ast);
    if (value.getType() != fieldType)
      value = generateAutoCast(loc, value, fieldType);

    // Return the output of the field read op
    auto op = builder.create<FieldWriteOp>(loc, fieldType, var, value, field);
    return op.getResult();
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

    // Declare all arguments
    for (auto var_val : llvm::zip(args, argVals))
    {
      // Get the argument name/value
      auto name = std::get<0>(var_val);
      auto value = std::get<1>(var_val);
      // Associate the name with the alloca SSA value
      symbolTable.insert(name, value);
    }

    return func;
  }

  void Generator::updateSymbolTable(
    llvm::ArrayRef<llvm::StringRef> vars,
    llvm::ArrayRef<mlir::BlockArgument> vals)
  {
    for (auto ret_val : llvm::zip(vars, vals))
      symbolTable.update(std::get<0>(ret_val), std::get<1>(ret_val));
  }

  template<class T>
  void Generator::generateBBArgList(
    mlir::Location loc, llvm::ArrayRef<llvm::StringRef> names, T& vars)
  {
    // This creates a new list, not append to an existing one
    vars.clear();
    for (auto name : names)
    {
      auto value = symbolTable.lookup(name);
      auto cast = generateAutoCast(loc, value, unkTy);
      vars.push_back(cast);
    }
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
      type = generateType(typeName);
    return genOperation(loc, "verona.constant(" + value.str() + ")", {}, type);
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
}
