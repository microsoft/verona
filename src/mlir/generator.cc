// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "generator.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Types.h"

#include <string>

using namespace verona::parser;

/// LLVM aliases
using StructType = mlir::LLVM::LLVMStructType;
using PointerType = mlir::LLVM::LLVMPointerType;

namespace
{
  /// Helper to make sure the basic block has a terminator
  bool hasTerminator(mlir::Block* bb)
  {
    return !bb->getOperations().empty() &&
      bb->back().mightHaveTrait<mlir::OpTrait::IsTerminator>();
  }

  /// Return true if the value has a pointer type.
  bool isPointer(mlir::Value val)
  {
    return val && val.getType().isa<PointerType>();
  }

  /// Return the element type if val is a pointer.
  mlir::Type getElementType(mlir::Value val)
  {
    assert(isPointer(val) && "Bad type");
    return val.getType().dyn_cast<PointerType>().getElementType();
  }

  /// Return true if the value has a pointer to a structure type.
  bool isStructPointer(mlir::Value val)
  {
    return isPointer(val) && getElementType(val).isa<StructType>();
  }

  /// Return the element type if val is a pointer.
  mlir::Type getFieldType(StructType type, int offset)
  {
    auto field = type.getBody().begin();
    std::advance(field, offset);
    return PointerType::get(*field);
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

  Value Generator::typeConversion(Value val, Type ty)
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

    // Promote the smallest to the largest
    if (lhsSize < rhsSize)
      lhs = typeConversion(lhs, rhsType);
    else
      rhs = typeConversion(rhs, lhsType);

    return {lhs, rhs};
  }

  std::string Generator::mangleName(
    llvm::StringRef name, llvm::ArrayRef<llvm::StringRef> scope)
  {
    // FIXME: This is a hack to help running LLVM modules
    if (name == "main")
      return name.str();

    // TODO: This is inefficient but works for now
    std::string fullName;
    // Prepend the function scope (module, etc)
    for (auto s : functionScope)
    {
      fullName += s.str() + "__";
    }
    // Add the relative scope (class name, etc)
    for (auto s : scope)
    {
      fullName += s.str() + "__";
    }
    // Append the actual function's name
    fullName += name.str();
    return fullName;
  }

  std::tuple<size_t, Type, bool>
  Generator::getField(Type type, llvm::StringRef fieldName)
  {
    auto structTy = type.dyn_cast<LLVM::LLVMStructType>();
    assert(structTy && "Bad type for field access");

    auto name = structTy.getName();
    auto& fieldMap = classFields[name];
    size_t pos = 0;
    for (auto f : fieldMap.fields)
    {
      // Always recurse into all class fields
      auto classField = fieldMap.types[pos].dyn_cast<LLVM::LLVMStructType>();
      if (classField)
      {
        auto [subPos, subTy, subFound] = getField(classField, fieldName);
        pos += subPos;

        if (subFound)
          return {pos, subTy, true};
        else
          continue;
      }

      // Found the field, return position
      if (f == fieldName)
      {
        auto ty = fieldMap.types[pos];
        return {pos, ty, true};
      }

      // Not found, continue searching
      pos++;
    }

    // Not found, return false
    return {0, Type(), false};
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
      // FIXME: This may not come fully qualified, may break if two classes have
      // the same name in different contexts.
      modName = node->id.view();
    }

    // Push another scope for variables, functions and types
    SymbolScopeT var_scope(symbolTable);
    functionScope.push_back(modName);
    auto type =
      LLVM::LLVMStructType::getIdentified(builder.getContext(), modName);

    // Create an entry for the fields
    classFields.emplace(modName, FieldOffset());

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
          auto res = parseField(sub);
          if (auto err = res.takeError())
            return err;

          // Update class-field map
          auto field = nodeAs<Field>(sub);
          auto& fieldMap = classFields[modName];
          fieldMap.fields.push_back(field->location.view());
          fieldMap.types.push_back(*res);
          break;
        }
        default:
          return runtimeError("Wrong member in class");
      }
    }
    if (mlir::failed(
          type.setBody(classFields[modName].types, /*packed*/ false)))
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
      // TODO: Implement tuple for multiple arguments
      auto rhsNode = parseNode(select->args);
      if (auto err = rhsNode.takeError())
        return std::move(err);
      rhs = rhsNode->get<Value>();
    }

    // FIXME: "special case" return for now, to make it work without method
    // call. There's a bug in the current AST that doesn't create a "last" value
    // in some cases, so we add an explicit "return" to force it.
    if (select->typenames[0]->location.view() == "return")
    {
      auto thisFunc =
        dyn_cast<FuncOp>(builder.getInsertionBlock()->getParentOp());
      if (thisFunc.getType().getNumResults() > 0)
        rhs = generateAutoLoad(loc, rhs, thisFunc.getType().getResult(0));
      return rhs;
    }

    // This is either:
    //  * the LHS of a binary operator
    //  * the selector for a static/dynamic call of a class member
    if (select->expr)
    {
      auto lhsNode = parseNode(select->expr);
      if (auto err = lhsNode.takeError())
        return std::move(err);
      lhs = lhsNode->get<Value>();
    }

    // Dynamic selector, for accessing a field or calling a method
    if (isStructPointer(lhs))
    {
      auto structTy = getElementType(lhs);

      // Loading fields, we calculate the offset to load based on the field name
      auto [offset, elmTy, found] =
        getField(structTy, select->typenames[0]->location.view());
      if (found)
      {
        // Convert the address of the structure to the address of the element
        return generateGEP(loc, lhs, offset);
      }

      // FIXME: Implement dynamic dispatch of methods
      assert(false && "Dynamic method call not implemented yet");
    }

    // Typenames indicate the context and the function name
    llvm::SmallVector<llvm::StringRef, 3> scope;
    size_t end = select->typenames.size() - 1;
    for (size_t i = 0; i < end; i++)
    {
      scope.push_back(select->typenames[i]->location.view());
    }
    std::string opName =
      mangleName(select->typenames[end]->location.view(), scope);

    // Check the function table for a symbol that matches the opName
    if (auto funcOp = module->lookupSymbol<FuncOp>(opName))
    {
      // If function takes a value and rhs is a pointer (alloca), load first
      // TODO: Handle tuples
      llvm::SmallVector<Value, 1> args;
      if (rhs)
      {
        rhs = generateAutoLoad(loc, rhs, funcOp.args_begin()->getType());
        args.push_back(rhs);
      }
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
    auto lhsNode = parseNode(assign->left);
    if (auto err = lhsNode.takeError())
      return std::move(err);
    auto addr = lhsNode->get<Value>();

    // Evaluate the right hand side to get type information
    auto rhsNode = parseNode(assign->right);
    if (auto err = rhsNode.takeError())
      return std::move(err);
    auto val = rhsNode->get<Value>();

    // No address means inline let/var (incl. temps), which has no type.
    // We evaluate the RHS first (above) to get its type and create an address
    // of the same type to store in.
    if (!addr)
    {
      assert(nodeAs<Let>(assign->left) || nodeAs<Var>(assign->left));
      auto name = assign->left->location.view();

      // If the value is a pointer, we just alias the temp with the SSA address
      if (isPointer(val))
      {
        symbolTable.update(name, val);
        return val;
      }
      // Else, allocate some space to store val into it
      else
      {
        addr = generateAlloca(getLocation(ast), val.getType());
        symbolTable.update(name, addr);
      }
    }
    assert(isPointer(addr) && "Couldn't create an address for lhs in assign");

    // If both are addresses, we need to load from the RHS to be able to store
    // into the LHS
    if (isPointer(val))
      val = generateAutoLoad(val.getLoc(), val, getElementType(val));

    // If LHS and RHS types don't match, do type conversion to make them match.
    // This is specially important in literals, which still have largest types
    // themselves (I64, F64).
    auto addrTy = getElementType(addr);
    auto valTy = val.getType();
    if (addrTy != valTy)
    {
      val = typeConversion(val, addrTy);
    }

    // Load the existing value to return (most of the time unused, elided)
    auto old = generateLoad(getLocation(assign), addr);

    // Store the new value in the same address
    generateStore(getLocation(assign), addr, val);

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
        assert(type.isa<IntegerType>() && "Bad type for integer literal");
        auto op = builder.create<ConstantIntOp>(loc, val, type);
        return op->getOpResult(0);
        break;
      }
      case Kind::Float:
      {
        auto F = nodeAs<Float>(ast);
        assert(F && "Bad Node");
        auto str = F->location.view();
        auto val = llvm::APFloat(std::stod(str.data()));
        auto type = parseType(ast);
        auto floatType = type.dyn_cast<FloatType>();
        assert(floatType && "Bad type for float literal");
        auto op = builder.create<ConstantFloatOp>(loc, val, floatType);
        return op->getOpResult(0);
        break;
      }
      case Kind::Character:
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
        // FIXME: This gets the size of the host, not the target. We need a
        // target-info kind of class here to get this kinf of information, but
        // this will do for now.
        auto size = sizeof(size_t) * 8;
        // FIXME: This is possibly too early to do this conversion, but
        // helps us run lots of tests before actually implementing classes,
        // etc.
        // FIXME: Support unsigned values. The standard dialect only has
        // signless operations, so we restrict current tests to I* and avoid U*
        // integer types.
        Type type = llvm::StringSwitch<Type>(name)
                      .Case("I8", builder.getIntegerType(8))
                      .Case("I16", builder.getIntegerType(16))
                      .Case("I32", builder.getIntegerType(32))
                      .Case("I64", builder.getIntegerType(64))
                      .Case("I128", builder.getIntegerType(128))
                      .Case("ISize", builder.getIntegerType(size))
                      .Case("F32", builder.getF32Type())
                      .Case("F64", builder.getF64Type())
                      .Default(Type());
        // If type wasn't detected, it must be a class
        // The order of declaration doesn't matter, so we create empty
        // classes if they're not declared yet. Note: getIdentified below is a
        // get-or-add function.
        if (!type)
        {
          type = StructType::getIdentified(builder.getContext(), name);
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
    // TODO: Implement dynamic method calls
    auto call = builder.create<CallOp>(loc, func, args);
    // TODO: Implement multiple return values (tuples?)
    return call->getOpResult(0);
  }

  llvm::Expected<Value> Generator::generateArithmetic(
    Location loc, llvm::StringRef opName, Value lhs, Value rhs)
  {
    // FIXME: Implement all unary and binary operators
    assert(lhs && rhs && "No binary operation with less than two arguments");

    // Make sure we're dealing with values, not pointers
    // FIXME: This shouldn't be necessary at this point
    if (isPointer(lhs))
      lhs = generateLoad(loc, lhs);
    if (isPointer(rhs))
      rhs = generateLoad(loc, rhs);

    // Promote types to be the same, or ops don't work, in the end, both
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
    assert(retTy.isa<IntegerType>() && "Bad arithmetic types");
    auto op = llvm::StringSwitch<Value>(opName)
                .Case("+", builder.create<AddIOp>(loc, retTy, lhs, rhs))
                .Default({});
    assert(op && "Unknown arithmetic operator");
    return op;
  }

  Value Generator::generateAlloca(Location loc, Type ty)
  {
    PointerType pointerTy;
    Value len = generateConstant(builder.getI32Type(), 1);
    pointerTy = PointerType::get(ty);
    return builder.create<LLVM::AllocaOp>(loc, pointerTy, len);
  }

  Value Generator::generateGEP(Location loc, Value addr, int offset)
  {
    llvm::SmallVector<Value> offsetList;
    // First argument is always in context of a list
    if (isStructPointer(addr))
    {
      auto zero = generateZero(builder.getI32Type());
      offsetList.push_back(zero);
    }
    // Second argument is in context of the struct
    auto len = generateConstant(builder.getI32Type(), offset);
    offsetList.push_back(len);
    ValueRange index(offsetList);
    Type retTy = addr.getType();
    if (auto structTy = getElementType(addr).dyn_cast<StructType>())
      retTy = getFieldType(structTy, offset);
    return builder.create<LLVM::GEPOp>(loc, retTy, addr, index);
  }

  Value Generator::generateLoad(Location loc, Value addr, int offset)
  {
    if (!isa<LLVM::GEPOp>(addr.getDefiningOp()))
      addr = generateGEP(loc, addr, offset);
    else
      assert(offset == 0 && "Can't take an offset of a GEP");
    return builder.create<LLVM::LoadOp>(loc, addr);
  }

  Value
  Generator::generateAutoLoad(Location loc, Value addr, Type ty, int offset)
  {
    // If it's not an address, there's nothing to load
    if (!isPointer(addr))
      return addr;

    // If the expected type is a pointer, we want the address, not the value
    if (ty.isa<PointerType>())
      return addr;

    auto elmTy = getElementType(addr);
    assert(elmTy == ty && "Invalid pointer load");
    return generateLoad(loc, addr, offset);
  }

  void Generator::generateStore(Location loc, Value addr, Value val, int offset)
  {
    if (!isa<LLVM::GEPOp>(addr.getDefiningOp()))
      addr = generateGEP(loc, addr, offset);
    else
      assert(offset == 0 && "Can't take an offset of a GEP");
    builder.create<LLVM::StoreOp>(loc, val, addr);
  }

  Value Generator::generateConstant(Type ty, std::variant<int, double> val)
  {
    auto loc = builder.getUnknownLoc();
    if (ty.isIndex())
    {
      return builder.create<ConstantIndexOp>(loc, std::get<int>(val));
    }
    else if (auto it = ty.dyn_cast<IntegerType>())
    {
      return builder.create<ConstantIntOp>(loc, std::get<int>(val), it);
    }
    else if (auto ft = ty.dyn_cast<FloatType>())
    {
      APFloat value = APFloat(std::get<double>(val));
      return builder.create<ConstantFloatOp>(loc, value, ft);
    }

    assert(0 && "Type not supported for zero");

    // Return invalid value for release builds
    // FIXME: Attach diagnostics engine here to report problems like these.
    return Value();
  }

  Value Generator::generateZero(Type ty)
  {
    if (ty.isa<FloatType>())
      return generateConstant(ty, 0.0);
    else
      return generateConstant(ty, 0);
  }
}
