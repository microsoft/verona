// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "consumer.h"

#include "error.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Types.h"
#include "parser/pass.h"

using namespace verona::parser;

namespace mlir::verona
{
  /// ASTMLIRPass - Base class for AST to MLIR passes.
  ///
  /// Passing the AST through this passes (via "pass << ast") populates the MLIR
  /// module in the generator. These passes can be chained together, if they are
  /// created with the same consumer and generator, and the same module will be
  /// updated.
  ///
  /// The objective of these passes is *just* to get AST into MLIR, not to
  /// change the AST nor the MLIR. There's a particular order in which the AST
  /// has to go through these passes, otherwise the pass will fail.
  ///
  /// To avoid declaration order issues, we have a declaration pass first, and a
  /// definition pass afterward. The ASTConsumer class takes care of doing that
  /// the right way.
  template<class Derived>
  struct ASTMLIRPass : Pass<Derived>
  {
    ASTConsumer& con;
    MLIRGenerator& gen;
    ASTMLIRPass(ASTConsumer& con, MLIRGenerator& gen) : con(con), gen(gen) {}

    /// Function scope, for mangling names. 5 because there will always be the
    /// root module, the current module and a class, at the very least.
    llvm::SmallVector<llvm::StringRef, 5> functionScope;

    /// Get builder from generator. This must be used to build any MLIR node as
    /// it keeps the context where the last operations were inserted.
    OpBuilder& builder()
    {
      return gen.getBuilder();
    }

    /// Get symbol table from generator. This must be used for all variables
    /// (user declared, temporaries, compiler generated) so that we can always
    /// refer to any declared variable from anywhere.
    SymbolTableT& symbolTable()
    {
      return gen.getSymbolTable();
    }
  };

  /// ASTDeclarations - traverses the AST and declares types and functions
  /// before they get user in definitions.
  struct ASTDeclarations : ASTMLIRPass<ASTDeclarations>
  {
    /// Stack for classes in construction. Elements are all incomplete.
    /// Classes get their fields assigned in program order, so non-top elements
    /// may have some fields. Once the class is completed, it pops off the
    /// stack and moves to the classInfo map.
    std::stack<ClassInfo> classStack;

  public:
    ASTDeclarations(ASTConsumer& con, MLIRGenerator& gen)
    : ASTMLIRPass(con, gen)
    {}

    /// Declarations for base nodes
    AST_PASS;

    /// Class declaration, sets up the current class' structure as empty.
    /// Processing fields will update structure.
    void pre(Class& node)
    {
      StringRef modName = node.id.view();
      // The root module has no name, doesn't need to be in the context
      if (!modName.empty())
        functionScope.push_back(modName);

      // Push another scope for variables, functions and types
      symbolTable().pushScope();

      // Create an entry for the class' structure type and fields
      classStack.push(ClassInfo(builder().getContext(), modName));
    }

    /// Post processing will finalise the structure.
    void post(Class& node)
    {
      StringRef modName = node.id.view();
      auto& info = classStack.top();
      assert(
        info.getType().getName() == modName && "Finishing the wrong class");

      // Finalise the structure
      info.finalize();

      // Update the map between class types and their field names
      auto key = info.key();
      con.classInfo.emplace(key, std::move(info));

      // Pop the function scope for name mangling and current class
      if (!functionScope.empty())
        functionScope.pop_back();
      classStack.pop();
      symbolTable().popScope();
    }

    /// Add field name and type to the list of class fields
    void post(Field& node)
    {
      // Update class-field map
      auto& info = classStack.top();
      info.addField(node.location.view(), con.consumeType(*node.type));
    }

    /// Parse function declaration nodes
    void post(Function& node)
    {
      auto loc = con.getLocation(node.as<NodeDef>());

      // Find all arguments
      llvm::SmallVector<Type> types;
      for (auto p : node.params)
      {
        auto param = p->as<Param>();
        types.push_back(con.consumeType(*param.type));
      }

      // Check return type (multiple returns as tuples)
      llvm::SmallVector<Type> retTy;
      if (node.result)
      {
        retTy.push_back(con.consumeType(*node.result));
      }

      // Generate the prototype
      auto name = con.mangleName(node.name.view(), functionScope);
      auto func = gen.lookupSymbol<FuncOp>(name);
      assert(!func && "Function redeclaration");
      func = gen.Proto(loc, name, types, retTy);

      // Push the function declaration into the module
      gen.push_back(func);
    }
  };

  /// ASTDefinitions - traverses the AST defining class bodies, functions,
  /// lambdas, etc. Uses the pre-declaration above to avoid declaration order
  /// issues.
  struct ASTDefinitions : ASTMLIRPass<ASTDefinitions>
  {
  private:
    Type selectTypeFromAssign;

    /// Stack of operands for calls, operations, assignments in the order
    /// they're evaluated (ex. lhs, rhs). The current AST isn't completely
    /// A-normal form, so we aren't dealing exclusively with values from a
    /// symbol table quite yet.
    std::stack<Value> operands;

    /// Push operand into stack.
    void pushOperand(Value val)
    {
      operands.push(val);
    }

    /// Take a value from the operands list. Returns Value() if list empty.
    Value takeOperand(bool last = false)
    {
      if (operands.empty())
        return Value();
      auto val = operands.top();
      operands.pop();
      if (last)
        assert(
          operands.empty() && "Mismatch on creating and consuming operands");
      return val;
    }

  public:
    ASTDefinitions(ASTConsumer& con, MLIRGenerator& gen) : ASTMLIRPass(con, gen)
    {}

    /// Declarations for base nodes
    AST_PASS;

    /// Class definition, type already exists, just keep the context up-to-date.
    void pre(Class& node)
    {
      // The root module has no name, doesn't need to be in the context
      StringRef modName = node.id.view();
      if (!modName.empty())
        functionScope.push_back(modName);

      // Push another scope for variables, functions and types
      symbolTable().pushScope();
    }

    /// Post processing will pop the scopes.
    void post(Class& node)
    {
      // Pop the variables scope
      symbolTable().popScope();

      // Pop the function scope for name mangling and current class
      if (!functionScope.empty())
        functionScope.pop_back();
    }

    /// Defines a function and creates its structure
    /// Following post(node)s will lower the body
    void pre(Function& node)
    {
      // Initialise the function's definition
      auto name = con.mangleName(node.name.view(), functionScope);
      auto func = gen.lookupSymbol<FuncOp>(name);
      assert(func && "Definition of an undeclared function");
      auto funcIR = gen.StartFunction(func);

      // Declare all arguments on a new scope
      symbolTable().pushScope();
      llvm::SmallVector<llvm::StringRef> argNames;
      for (auto p : node.params)
      {
        argNames.push_back(p->location.view());
        // TODO: Handle default init
      }
      auto& entryBlock = *funcIR->getRegion(0).getBlocks().begin();
      auto argVals = entryBlock.getArguments();
      for (auto [name, val] : llvm::zip(argNames, argVals))
      {
        symbolTable().insert(name, val);
      }
    }

    /// Closes the function, checking the return value
    void post(Function& node)
    {
      // Automatically clean up
      ScopeCleanup pop([&]() { symbolTable().popScope(); });

      // Check if needs to return a value at all
      if (gen.hasTerminator(builder().getBlock()))
        return;

      // Fetch the current function
      auto loc = con.getLocation(node);
      auto name = con.mangleName(node.name.view(), functionScope);
      auto func = gen.lookupSymbol<FuncOp>(name);
      assert(func && "Definition of an undeclared function");

      // Lower return value
      auto val = takeOperand(/*last=*/true);
      gen.Return(loc, func, val);
    }

    /// Local declarations (including temps) reserve a place on the symbol table
    /// FIXME: in the new AST, with types, the alloca will be done here
    /// FIXME: Let and Var are indistinguishable at this stage, merge them
    void pre(Let& node)
    {
      symbolTable().insert(node.location.view(), Value());
    }

    /// Local declarations reserve a place on the symbol table
    /// FIXME: in the new AST, with types, the alloca will be done here
    /// FIXME: Let and Var are indistinguishable at this stage, merge them
    void pre(Var& node)
    {
      symbolTable().insert(node.location.view(), Value());
    }

    /// Define the type of a node (this is going away on the new AST)
    void post(Oftype& node)
    {
      auto loc = con.getLocation(node);
      assert(
        node.expr->kind() == Kind::Ref && "oftype expression must be a ref");

      // Make sure the variable exists, but it's uninitialized
      auto val = con.lookup(node.expr, /*local scope*/ true);
      assert(!val && "Expression already has type");

      // Alloca the right size and update the symbol table
      auto newTy = con.consumeType(*node.type);
      Value addr = gen.Alloca(loc, newTy);
      auto name = node.expr->location.view();
      symbolTable().update(name, addr);
    }

    /// Create an integer literal, push to the operands list
    auto post(Int& node)
    {
      auto loc = con.getLocation(node);
      auto str = node.location.view();
      auto val = std::stol(str.data());
      // FIXME: The new AST will have the actual type
      auto type = builder().getIntegerType(64);
      auto op = builder().create<ConstantIntOp>(loc, val, type);
      pushOperand(op->getOpResult(0));
    }

    /// Create a float literal, push to the operands list
    auto post(Float& node)
    {
      auto loc = con.getLocation(node);
      auto str = node.location.view();
      auto val = llvm::APFloat(std::stod(str.data()));
      // FIXME: The new AST will have the actual type
      auto type = builder().getF64Type().dyn_cast<FloatType>();
      auto op = builder().create<ConstantFloatOp>(loc, val, type);
      pushOperand(op->getOpResult(0));
    }

    /// Create an escaped string literal, push to the operands list
    auto post(EscapedString& node)
    {
      // TODO: Actually implement this for real
      pushOperand(gen.ConstantString(node.location.view()));
    }

    /// Create an unescaped string literal, push to the operands list
    auto post(UnescapedString& node)
    {
      // TODO: Actually implement this for real
      pushOperand(gen.ConstantString(node.location.view()));
    }

    /// Selects (for now) can be many things so we need some checks to see how
    /// to handle it. Soon it'll be just dynamic selection and static calls and
    /// field access will have their own node types.
    void post(Select& node)
    {
      auto loc = con.getLocation(node);
      // The right-hand side of a select is always a reference (or nothing)
      auto rhs = con.lookup(node.args);

      // FIXME: "special case" return for now, to make it work without method
      // call. There's a bug in the current AST that doesn't create a "last"
      // value in some cases, so we add an explicit "return" to force it.
      if (node.typenames[0]->location.view() == "return")
      {
        assert(operands.empty());
        auto thisFunc =
          dyn_cast<FuncOp>(builder().getInsertionBlock()->getParentOp());
        auto funcTy = thisFunc.getType();
        if (funcTy.getNumResults() > 0)
        {
          assert(rhs && "Return needs value but was given none");
          rhs = gen.AutoLoad(loc, rhs, thisFunc.getType().getResult(0));
        }
        gen.Return(loc, thisFunc, rhs);
        return;
      }

      // The left-hand side of a select is always a reference (or nothing)
      auto lhs = con.lookup(node.expr);

      // Dynamic selector, for accessing a field or calling a method
      if (auto structTy = gen.getPointedStructType(lhs))
      {
        // Loading fields, we calculate the offset to load based on the field
        // name
        auto key = ClassInfo::key(structTy);
        auto& info = con.classInfo.at(key);
        auto [offset, elmTy] =
          info.getFieldType(node.typenames[0]->location.view());
        if (elmTy)
        {
          // Convert the address of the structure to the address of the element
          pushOperand(gen.GEP(loc, lhs, offset));
          return;
        }

        // FIXME: Implement dynamic dispatch of methods
        assert(false && "Dynamic method call not implemented yet");
      }

      // Typenames indicate the context and the function name
      llvm::SmallVector<llvm::StringRef, 3> scope;
      size_t end = node.typenames.size() - 1;
      for (size_t i = 0; i < end; i++)
      {
        scope.push_back(node.typenames[i]->location.view());
      }
      std::string opName = con.mangleName(
        node.typenames[end]->location.view(), functionScope, scope);

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
          // Multiple arguments wrap as a tuple. If the function arguments
          // weren't wrapped in a tuple, deconstruct it to get the right types
          // for the call.
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

        auto ret = gen.Call(loc, funcOp, args);
        if (funcOp.getType().getNumResults())
          pushOperand(ret);
        return;
      }

      // If function does not exist, it's either arithmetic or an error.
      // lhs has the operation name, rhs are the ops (in a tuple)
      auto retTy = selectTypeFromAssign;
      selectTypeFromAssign = Type();

      // FIXME: This is a work-around the current AST shape. Future versions
      // will use a special symbol, `@` to nominate foreign functions (like
      // inline MLIR/LLVM IR) and restrict those to special modules only.
      auto addrOp = dyn_cast<LLVM::AddressOfOp>(lhs.getDefiningOp());
      assert(addrOp && "Arithmetic implemented as string calls");
      opName = addrOp.global_name();
      // String was stored previously and kept as an operand, we don't need it
      // anymore.
      // FIXME: This should take the last one, some values are being left over,
      // inverstigate
      takeOperand();
      pushOperand(gen.Arithmetic(loc, opName, rhs, retTy));
    }

    void pre(Assign& node)
    {
      // FIXME: This is needed if the address has a type but the expression
      // doesn't. This happens on arithmetic, for example, where the expresion
      // is just a string and the return type is not always the same as the
      // arguments (ex. truncate/extend). Once all AST nodes have types, this
      // can be removed.
      auto lhs = con.lookup(node.left);
      if (lhs)
      {
        if (gen.isPointer(lhs))
          selectTypeFromAssign = gen.getPointedType(lhs);
        else
          selectTypeFromAssign = lhs.getType();
      }
    }

    /// Assign needs both address and value to be evaluated first, so we
    /// handle it on post. The current AST has no types on nodes, but at least
    /// either lhs or rhs must have a type, so we check them first and make
    /// sure we match.
    void post(Assign& node)
    {
      auto loc = con.getLocation(node);
      Value val;
      // Value can be a reference (and the operand list must be empty)
      if (node.right->kind() == Kind::Ref)
      {
        assert(operands.empty());
        val = con.lookup(node.right);
      }
      // Or it can be the result of an operation
      else
      {
        assert(!operands.empty());
        val = takeOperand();
      }

      // Address is always a reference or an inline let/var
      auto addr = con.lookup(node.left);

      // No address means inline let/var (incl. temps), which has no type.
      // We evaluate the RHS first (above) to get its type and create an
      // address of the same type to store in.
      if (!addr)
      {
        assert(
          node.left->kind() == Kind::Let || node.left->kind() == Kind::Var ||
          node.left->kind() == Kind::Ref);
        auto name = node.left->location.view();

        // If the value is a pointer, we just alias the temp with the SSA
        // address
        if (gen.isPointer(val))
        {
          symbolTable().update(name, val);
          return;
        }
        // Else, allocate some space to store val into it (below)
        else
        {
          addr = gen.Alloca(loc, val.getType());
          symbolTable().update(name, addr);
        }
      }
      // Either way, we must have a valid address by now
      assert(
        gen.isPointer(addr) && "Couldn't create an address for lhs in assign");

      // If both are addresses, we need to load from the RHS to be able to
      // store into the LHS. We can't just alias (like above) because both
      // addresses exist and have their own values and provenance.
      val = gen.AutoLoad(val.getLoc(), val);

      // Types of LHS and RHS must match.
      auto addrTy = gen.getPointedType(addr);
      auto valTy = val.getType();
      assert(addrTy == valTy && "Assignment types must be the same");

      // TODO: Load the existing value to return, if the value is used.
      //
      // This isn't implemented yet because we track used values with the
      // operands stack which would get out of sync if we load values that don't
      // get used.
      //
      // Once the AST has all operands as references, the load would probably go
      // to a temporary symbol and even if unused, there would be no operands
      // stack to get misaligned.

      // Store the new value in the same address
      gen.Store(loc, addr, val);
    }

    /// Creates new tuples and initialise their fields
    void post(Tuple& node)
    {
      auto loc = con.getLocation(node);

      // Evaluate each tuple element
      SmallVector<Value> values;
      SmallVector<Type> types;
      for (auto sub : node.seq)
      {
        auto val = con.lookup(sub);
        auto type = val.getType();
        // FIXME: Currently, tuples and values are stored as pointers in the
        // symbol table and without explicit type on the temps we don't know
        // if we want the value of the address.
        // For now, we assume value, but this is clearly wrong.
        if (gen.isPointer(val))
          type = gen.getPointedType(val);
        values.push_back(val);
        types.push_back(type);
      }

      // This creates a type from the declaration itself. This will be assigned
      // to a variable that we assume has the exact same type, as type inference
      // sould have occured by now.
      // If the types mismatch, the assign will bail.
      auto tupleType = StructType::getLiteral(builder().getContext(), types);
      auto addr = gen.Alloca(loc, tupleType);

      // Store the elements in the allocated space
      size_t index = 0;
      for (auto sub : values)
      {
        auto gep = gen.GEP(loc, addr, index++);
        auto value = gen.AutoLoad(loc, sub);
        gen.Store(loc, gep, value);
      }

      pushOperand(addr);
    }
  };

  // ===================================================== Public Interface

  llvm::Expected<OwningModuleRef>
  ASTConsumer::lower(MLIRContext* context, ::verona::parser::Ast ast)
  {
    ASTConsumer con(context);

    // Declaration pass
    ASTDeclarations decl(con, con.gen);
    decl.set_error(std::cerr);
    decl << ast;

    // Definition pass
    ASTDefinitions def(con, con.gen);
    def.set_error(std::cerr);
    def << ast;

    // TODO: MLIR passes, if needed

    // Return the owning module
    return con.gen.finish();
  }

  // ===================================================== Helpers

  Location ASTConsumer::getLocation(::verona::parser::NodeDef& ast)
  {
    if (!ast.location.source)
      return builder().getUnknownLoc();

    auto path = ast.location.source->origin;
    auto [line, column] = ast.location.linecol();
    return mlir::FileLineColLoc::get(
      builder().getIdentifier(path), line, column);
  }

  std::string ASTConsumer::mangleName(
    llvm::StringRef name,
    llvm::ArrayRef<llvm::StringRef> functionScope,
    llvm::ArrayRef<llvm::StringRef> callScope)
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
    for (auto s : callScope)
    {
      os << s.str() << "__";
    }
    // Append the actual function's name
    os << name.str();

    return os.str();
  }

  Value ASTConsumer::lookup(::verona::parser::Ast ast, bool lastContextOnly)
  {
    if (!ast)
      return Value();
    auto name = ast->location.view();
    return symbolTable().lookup(name, lastContextOnly);
  }

  Type ASTConsumer::consumeType(::verona::parser::NodeDef& ast)
  {
    switch (ast.kind())
    {
      case Kind::TypeRef:
      {
        auto R = ast.as<TypeRef>();
        // TODO: Implement type list
        return consumeType(R.typenames[0]->as<::verona::parser::TypeName>());
      }
      case Kind::TypeName:
      {
        auto C = ast.as<TypeName>();
        auto name = C.location.view();
        // FIXME: This gets the size of the host, not the target. We need a
        // target-info kind of class here to get this kinf of information, but
        // this will do for now.
        auto size = sizeof(size_t) * 8;
        // FIXME: This is possibly too early to do this conversion, but
        // helps us run lots of tests before actually implementing classes,
        // etc.
        // FIXME: Support unsigned values. The standard dialect only has
        // signless operations, so we restrict current tests to I* and avoid
        // U* integer types.
        Type type = llvm::StringSwitch<Type>(name)
                      .Case("I8", builder().getIntegerType(8))
                      .Case("I16", builder().getIntegerType(16))
                      .Case("I32", builder().getIntegerType(32))
                      .Case("I64", builder().getIntegerType(64))
                      .Case("I128", builder().getIntegerType(128))
                      .Case("U8", builder().getIntegerType(8))
                      .Case("U16", builder().getIntegerType(16))
                      .Case("U32", builder().getIntegerType(32))
                      .Case("U64", builder().getIntegerType(64))
                      .Case("U128", builder().getIntegerType(128))
                      .Case("ISize", builder().getIntegerType(size))
                      .Case("USize", builder().getIntegerType(size))
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
        auto T = ast.as<::verona::parser::TupleType>();
        llvm::SmallVector<Type> tuple;
        for (auto t : T.types)
          tuple.push_back(consumeType(*t));
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
