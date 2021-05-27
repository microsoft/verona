// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "consumer.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Types.h"

#include <string>

using namespace verona::parser;

namespace mlir::verona
{
  // ===================================================== Public Interface
  llvm::Expected<OwningModuleRef>
  ASTConsumer::lower(MLIRContext* context, Ast ast)
  {
    ASTConsumer con(context);
    auto err = con.consumeRootModule(ast);
    if (err)
      return std::move(err);

    return con.gen.finish();
  }

  // ===================================================== Helpers

  Location ASTConsumer::getLocation(Ast ast)
  {
    if (!ast->location.source)
      return builder().getUnknownLoc();

    auto path = ast->location.source->origin;
    auto [line, column] = ast->location.linecol();
    return mlir::FileLineColLoc::get(
      builder().getIdentifier(path), line, column);
  }

  std::string ASTConsumer::mangleName(
    llvm::StringRef name, llvm::ArrayRef<llvm::StringRef> scope)
  {
    // FIXME: This is a hack to help running LLVM modules
    if (name == "main")
      return name.str();

    // Use LLVM's raw_ostream for fast write and dump types directly
    std::string fullName;
    llvm::raw_string_ostream os(fullName);

    // Prepend the function scope (module, etc)
    for (auto s : functionScope)
    {
      os << s.str() << "__";
    }

    // Add the relative scope (class name, etc)
    for (auto s : scope)
    {
      os << s.str() << "__";
    }
    // Append the actual function's name
    os << name.str();

    return os.str();
  }

  std::tuple<size_t, Type, bool>
  ASTConsumer::getField(Type type, llvm::StringRef fieldName)
  {
    auto structTy = type.dyn_cast<StructType>();
    assert(structTy && "Bad type for field access");

    auto name = structTy.getName();
    auto& fieldMap = classFields[name];
    size_t pos = 0;
    for (auto f : fieldMap.fields)
    {
      // Found the field, return position
      if (f == fieldName)
      {
        auto ty = fieldMap.types[pos];
        return {pos, ty, true};
      }

      // Always recurse into all class fields
      auto classField = fieldMap.types[pos].dyn_cast<StructType>();
      if (classField)
      {
        auto [subPos, subTy, subFound] = getField(classField, fieldName);
        pos += subPos;

        if (subFound)
          return {pos, subTy, true};
        else
          continue;
      }

      // Not found, continue searching
      pos++;
    }

    // Not found, return false
    return {0, Type(), false};
  }

  // ===================================================== Top-Level Consumers

  llvm::Error ASTConsumer::consumeRootModule(Ast ast)
  {
    auto node = nodeAs<Class>(ast);
    assert(node && "Bad node");

    // TODO: Quick early pass to declare all functions before we descend the
    // AST.

    // Modules are just global classes
    return consumeClass(ast);
  }

  llvm::Error ASTConsumer::consumeClass(Ast ast)
  {
    auto node = nodeAs<Class>(ast);
    assert(node && "Bad node");

    StringRef modName = node->id.view();
    // The root module has no name, doesn't need to be in the context
    if (!modName.empty())
      functionScope.push_back(modName);

    // Push another scope for variables, functions and types
    SymbolScopeT var_scope(symbolTable());
    auto type = StructType::getIdentified(builder().getContext(), modName);

    // Create an entry for the fields
    classFields.emplace(modName, FieldOffset());

    // Lower members, types, functions
    for (auto sub : node->members)
    {
      switch (sub->kind())
      {
        case Kind::Class:
        {
          auto err = consumeClass(sub);
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
          auto func = consumeFunction(sub);
          if (auto err = func.takeError())
            return err;
          gen.push_back(*func);
          break;
        }
        case Kind::Field:
        {
          auto res = consumeField(sub);
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

  // ======================================================= General Consumers

  llvm::Expected<Value> ASTConsumer::consumeNode(Ast ast)
  {
    switch (ast->kind())
    {
      case Kind::Lambda:
        return consumeLambda(ast);
      case Kind::Select:
        return consumeSelect(ast);
      case Kind::Ref:
        return consumeRef(ast);
      case Kind::Assign:
        return consumeAssign(ast);
      case Kind::Let:
      case Kind::Var:
        return consumeLocalDecl(ast);
      case Kind::Oftype:
        return consumeOfType(ast);
      case Kind::Tuple:
        return consumeTuple(ast);
      case Kind::Character:
      case Kind::Int:
      case Kind::Float:
      case Kind::Hex:
      case Kind::Binary:
      case Kind::Bool:
        return consumeLiteral(ast);
      case Kind::EscapedString:
      case Kind::UnescapedString:
        return consumeString(ast);
      case Kind::Function:
      case Kind::Class:
      case Kind::Field:
        return runtimeError(
          "Top-level field " + std::string(kindname(ast->kind())) +
          " cannot be consumed through generic recursion");
      default:
        // TODO: Implement all others
        break;
    }

    return runtimeError(
      "Node " + std::string(kindname(ast->kind())) + " not implemented yet");
  }

  llvm::Expected<FuncOp> ASTConsumer::consumeFunction(Ast ast)
  {
    auto func = nodeAs<Function>(ast);
    assert(func && "Bad node");
    auto loc = getLocation(ast);

    // Find all arguments
    llvm::SmallVector<llvm::StringRef> argNames;
    llvm::SmallVector<Type> types;
    for (auto p : func->params)
    {
      auto param = nodeAs<Param>(p);
      assert(param && "Bad Node");
      argNames.push_back(param->location.view());
      types.push_back(consumeType(param->type));
      // TODO: Handle default init
    }

    // Check return type (multiple returns as tuples)
    llvm::SmallVector<Type> retTy;
    if (func->result)
    {
      retTy.push_back(consumeType(func->result));
    }

    // Declare all arguments on current scope
    SymbolScopeT var_scope(symbolTable());
    auto name = mangleName(func->name.view());
    auto def = gen.EmptyFunction(getLocation(ast), name, types, retTy);
    if (auto err = def.takeError())
      return std::move(err);
    auto& funcIR = *def;

    // Declare all arguments on current scope on a newly created stack object
    auto& entryBlock = *funcIR->getRegion(0).getBlocks().begin();
    auto argVals = entryBlock.getArguments();
    for (auto [name, val] : llvm::zip(argNames, argVals))
    {
      symbolTable().insert(name, val);
    }

    // Lower body
    auto body = func->body;
    auto last = consumeNode(body);
    if (auto err = last.takeError())
      return std::move(err);

    // Check if needs to return a value at all
    if (gen.hasTerminator(builder().getBlock()))
      return funcIR;

    // Lower return value
    // (TODO: cast type if not the same)
    bool needsReturn = !retTy.empty();

    if (needsReturn)
    {
      assert(last && "No value to return");
      builder().create<ReturnOp>(loc, *last);
    }
    else
    {
      builder().create<ReturnOp>(loc);
    }

    return funcIR;
  }

  llvm::Expected<Type> ASTConsumer::consumeField(Ast ast)
  {
    auto field = nodeAs<Field>(ast);
    assert(field && "Bad node");

    auto type = consumeType(field->type);
    // TODO: Add names to a hash so we can access for field read/write.
    // TODO: Implement initialiser

    return type;
  }

  llvm::Expected<Value> ASTConsumer::consumeLambda(Ast ast)
  {
    auto lambda = nodeAs<Lambda>(ast);
    assert(lambda && "Bad Node");

    // Blocks add lexical context
    SymbolScopeT var_scope{symbolTable()};

    Value last;
    llvm::SmallVector<Ast> nodes;
    for (auto sub : lambda->body)
    {
      auto node = consumeNode(sub);
      if (auto err = node.takeError())
        return std::move(err);
      last = *node;
    }
    return last;
  }

  llvm::Expected<Value> ASTConsumer::consumeSelect(Ast ast)
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
      auto rhsNode = consumeNode(select->args);
      if (auto err = rhsNode.takeError())
        return std::move(err);
      rhs = *rhsNode;
    }

    // FIXME: "special case" return for now, to make it work without method
    // call. There's a bug in the current AST that doesn't create a "last" value
    // in some cases, so we add an explicit "return" to force it.
    if (select->typenames[0]->location.view() == "return")
    {
      auto thisFunc =
        dyn_cast<FuncOp>(builder().getInsertionBlock()->getParentOp());
      if (thisFunc.getType().getNumResults() > 0)
        rhs = gen.AutoLoad(loc, rhs, thisFunc.getType().getResult(0));
      return rhs;
    }

    // This is either:
    //  * the LHS of a binary operator
    //  * the selector for a static/dynamic call of a class member
    if (select->expr)
    {
      auto lhsNode = consumeNode(select->expr);
      if (auto err = lhsNode.takeError())
        return std::move(err);
      lhs = *lhsNode;
    }

    // Dynamic selector, for accessing a field or calling a method
    if (auto structTy = gen.getPointedStructType(lhs))
    {
      // Loading fields, we calculate the offset to load based on the field name
      auto [offset, elmTy, found] =
        getField(structTy, select->typenames[0]->location.view());
      if (found)
      {
        // Convert the address of the structure to the address of the element
        return gen.GEP(loc, lhs, offset);
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
    if (auto funcOp = gen.lookupSymbol<FuncOp>(opName))
    {
      llvm::SmallVector<Value> args;
      // Here it's guaranteed the lhs is not a selector (handled above), so if
      // there is one, it's the first argument of a function call.
      if (lhs)
      {
        args.push_back(lhs);
      }
      if (rhs)
      {
        auto numArgs = funcOp.getNumArguments();
        // Single argument isn't wrapped in a tuple, so just push it.
        if (numArgs == 1)
        {
          assert(args.empty() && "lhs must be empty for single arg");
          // If argument is indeed a tuple, dereference the pointer
          if (gen.isStructPointer(rhs))
          {
            assert(
              funcOp.getArgument(0).getType().isa<StructType>() &&
              "Single argument type mismatch");

            // Pass a a struct, not as a pointer
            rhs = gen.GEP(loc, rhs);
          }
          rhs = gen.AutoLoad(loc, rhs);
          args.push_back(rhs);
        }
        // Multiple arguments wrap as a tuple. If the function arguments weren't
        // wrapped in a tuple, deconstruct it to get the right types for the
        // call.
        else
        {
          auto structTy = gen.getPointedStructType(rhs, /*anonymous*/ true);
          assert(
            structTy && structTy.getBody().size() == numArgs &&
            "Call to function with wrong number of operands");
          for (unsigned offset = 0, last = numArgs; offset < last; offset++)
          {
            auto ptr = gen.GEP(loc, rhs, offset);
            auto val = gen.Load(loc, ptr);
            args.push_back(val);
          }
        }
      }

      auto res = gen.Call(loc, funcOp, args);
      if (auto err = res.takeError())
        return std::move(err);
      return *res;
    }

    // If function does not exist, it's either arithmetic or an error.
    auto addrOp = dyn_cast<LLVM::AddressOfOp>(lhs.getDefiningOp());
    assert(addrOp && "Arithmetic implemented as string calls");
    // lhs has the operation name, rhs are the ops (in a tuple)
    opName = addrOp.global_name();
    return gen.Arithmetic(loc, opName, rhs);
  }

  llvm::Expected<Value> ASTConsumer::consumeRef(Ast ast)
  {
    auto ref = nodeAs<Ref>(ast);
    assert(ref && "Bad Node");
    return symbolTable().lookup(ref->location.view());
  }

  llvm::Expected<Value> ASTConsumer::consumeLocalDecl(Ast ast)
  {
    // FIXME: for now, just creates a new empty value that can be updated.
    return symbolTable().insert(ast->location.view(), Value());
  }

  llvm::Expected<Value> ASTConsumer::consumeOfType(Ast ast)
  {
    auto ofty = nodeAs<Oftype>(ast);
    assert(ofty && "Bad Node");
    auto name = ofty->expr->location.view();

    // Make sure the variable is uninitialized
    auto val = symbolTable().lookup(name, /*local scope*/ true);
    assert(!val);

    // FIXME: for now, just updates the reference's type
    auto newTy = consumeType(ofty->type);
    // FIXME: This is probably the wrong place to do this
    Value addr = gen.Alloca(getLocation(ofty), newTy);
    return symbolTable().update(name, addr);
  }

  llvm::Expected<Value> ASTConsumer::consumeAssign(Ast ast)
  {
    auto assign = nodeAs<Assign>(ast);
    assert(assign && "Bad Node");

    // lhs has to be an addressable expression (ref, let, var)
    auto lhsNode = consumeNode(assign->left);
    if (auto err = lhsNode.takeError())
      return std::move(err);
    auto addr = *lhsNode;

    // HACK: Working around the lack of type in assign, to know what the return
    // value of the function call it is.
    if (addr)
      assignTypeFromSelect = gen.getPointedType(addr);
    else
      assignTypeFromSelect = Type();

    // Evaluate the right hand side to get type information
    auto rhsNode = consumeNode(assign->right);
    if (auto err = rhsNode.takeError())
      return std::move(err);
    auto val = *rhsNode;

    // No address means inline let/var (incl. temps), which has no type.
    // We evaluate the RHS first (above) to get its type and create an address
    // of the same type to store in.
    bool needsLoad = true;
    if (!addr)
    {
      assert(nodeAs<Let>(assign->left) || nodeAs<Var>(assign->left));
      auto name = assign->left->location.view();

      // If the value is a pointer, we just alias the temp with the SSA address
      if (gen.isPointer(val))
      {
        symbolTable().update(name, val);
        return val;
      }
      // Else, allocate some space to store val into it
      else
      {
        addr = gen.Alloca(getLocation(ast), val.getType());
        symbolTable().update(name, addr);
      }
      needsLoad = false;
    }
    assert(
      gen.isPointer(addr) && "Couldn't create an address for lhs in assign");

    // If both are addresses, we need to load from the RHS to be able to store
    // into the LHS
    val = gen.AutoLoad(val.getLoc(), val);

    // If LHS and RHS types don't match, do type conversion to make them match.
    // This is specially important in literals, which still have largest types
    // themselves (I64, F64).
    auto addrTy = gen.getPointedType(addr);
    auto valTy = val.getType();
    if (addrTy != valTy)
    {
      val = gen.Convert(val, addrTy);
    }

    // Load the existing value to return (if addr existed before)
    Value old;
    if (needsLoad)
      old = gen.Load(getLocation(assign), addr);

    // Store the new value in the same address
    gen.Store(getLocation(assign), addr, val);

    // Return the previous value
    return old;
  }

  llvm::Expected<Value> ASTConsumer::consumeTuple(Ast ast)
  {
    auto tuple = nodeAs<Tuple>(ast);
    assert(tuple && "Bad Node");
    auto loc = getLocation(ast);

    // Evaluate each tuple element
    llvm::SmallVector<Value> values;
    llvm::SmallVector<Type> types;
    for (auto sub : tuple->seq)
    {
      auto node = consumeNode(sub);
      if (auto err = node.takeError())
        return std::move(err);
      values.push_back(*node);
      // FIXME: Currently, tuples and values are stored as pointers in the
      // symbol table and without explicit type on the temps we don't know if we
      // want the value of the address. For now, we assume value.
      auto actualType = node->getType();
      if (gen.isPointer(*node))
        actualType = gen.getPointedType(*node);
      types.push_back(actualType);
    }

    // Retrieve the tuple type from the assign (like select does) and allocate
    // space for its values. FIXME: Future versions of AST won't need this hack,
    // as all nodes will have types directly on them.
    auto tupleType = assignTypeFromSelect;
    if (!tupleType)
    {
      // When creating temporaries, we don't have the assign type yet, so we
      // borrow from the value definition, which should be correct by now, if we
      // don't create let expressions without explicit type definition.
      tupleType = StructType::getLiteral(builder().getContext(), types);
    }
    else
    {
      assignTypeFromSelect = Type();
    }
    auto addr = gen.Alloca(loc, tupleType);

    // Store the elements in the allocated space
    size_t index = 0;
    for (auto sub : values)
    {
      auto gep = gen.GEP(loc, addr, index++);
      auto value = gen.AutoLoad(loc, sub);
      gen.Store(loc, gep, value);
    }

    // Return the address of the tuple's storage
    return addr;
  }

  llvm::Expected<Value> ASTConsumer::consumeLiteral(Ast ast)
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
        auto type = consumeType(ast);
        assert(type.isa<IntegerType>() && "Bad type for integer literal");
        auto op = builder().create<ConstantIntOp>(loc, val, type);
        return op->getOpResult(0);
        break;
      }
      case Kind::Float:
      {
        auto F = nodeAs<Float>(ast);
        assert(F && "Bad Node");
        auto str = F->location.view();
        auto val = llvm::APFloat(std::stod(str.data()));
        auto type = consumeType(ast);
        auto floatType = type.dyn_cast<FloatType>();
        assert(floatType && "Bad type for float literal");
        auto op = builder().create<ConstantFloatOp>(loc, val, floatType);
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

  llvm::Expected<Value> ASTConsumer::consumeString(Ast ast)
  {
    std::string_view str;

    // TODO: Differentiate between escaped and unescaped
    if (auto S = nodeAs<UnescapedString>(ast))
      str = S->location.view();
    else if (auto S = nodeAs<EscapedString>(ast))
      str = S->location.view();
    else
      assert(false && "Unknown string type");

    return gen.ConstantString(str);
  }

  Type ASTConsumer::consumeType(Ast ast)
  {
    switch (ast->kind())
    {
      case Kind::Int:
        // TODO: Understand what the actual size is
        return builder().getIntegerType(64);
      case Kind::Float:
        // TODO: Understand what the actual size is
        return builder().getF64Type();
      case Kind::TypeRef:
      {
        auto R = nodeAs<TypeRef>(ast);
        assert(R && "Bad Node");
        // TODO: Implement type list
        return consumeType(R->typenames[0]);
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
                      .Case("I8", builder().getIntegerType(8))
                      .Case("I16", builder().getIntegerType(16))
                      .Case("I32", builder().getIntegerType(32))
                      .Case("I64", builder().getIntegerType(64))
                      .Case("I128", builder().getIntegerType(128))
                      .Case("ISize", builder().getIntegerType(size))
                      .Case("F32", builder().getF32Type())
                      .Case("F64", builder().getF64Type())
                      .Default(Type());
        // If type wasn't detected, it must be a class
        // The order of declaration doesn't matter, so we create empty
        // classes if they're not declared yet. Note: getIdentified below is a
        // get-or-add function.
        if (!type)
        {
          type = StructType::getIdentified(builder().getContext(), name);
        }
        assert(type && "Type not found");
        return type;
      }
      case Kind::TupleType:
      {
        auto T = nodeAs<::verona::parser::TupleType>(ast);
        assert(T && "Bad Node");
        llvm::SmallVector<Type> tuple;
        for (auto t : T->types)
          tuple.push_back(consumeType(t));
        // Tuples are represented as anonymous structures
        return StructType::getLiteral(builder().getContext(), tuple);
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
}
