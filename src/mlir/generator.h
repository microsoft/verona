// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include "error.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "parser/ast.h"
#include "symbol.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <string>
#include <variant>

namespace mlir::verona
{
  /**
   * Return Value.
   *
   * Verona constructs (including functions, operations, statements) can return
   * 0, 1 or many values (tuples). Constructs that don't return values (like
   * lexical blocks, conditionals, loops, termination) can safely return an
   * empty "ReturnValue" with the guarantee (by the ast construction) that no
   * other construct will try to use their return value.
   *
   * Tuples in MLIR are best represented by a loose list of values:
   * https://mlir.llvm.org/docs/Rationale/Rationale/#tuple-types
   * So we keep track of all values returned along with their normalised and
   * canonicalised types (property distribution), and can access them via simply
   * taking the n-th value in the list.
   */
  class ReturnValue
  {
    using ResultTy = std::variant<Value, ModuleOp, FuncOp>;
    using ValuesTy = llvm::SmallVector<ResultTy, 1>;
    ValuesTy values;

  public:
    /// Default constructor, builds an empty return value.
    ReturnValue() {}
    /// Constructor for single valued Value
    ReturnValue(Value value)
    {
      values.push_back(value);
    }
    /// Constructor for single valued ModuleOp
    ReturnValue(ModuleOp mod)
    {
      values.push_back(mod);
    }
    /// Constructor for single valued FuncOp
    ReturnValue(FuncOp func)
    {
      values.push_back(func);
    }
    /// Multiple value constructor (not necessarily same types)
    ReturnValue(ResultRange& range)
    {
      values.insert(values.begin(), range.begin(), range.end());
    }
    /// Assignment operator for Value.
    ReturnValue& operator=(Value& value)
    {
      values.clear();
      values.push_back(value);
      return *this;
    }

    /// Returns true if this return value has exactly one value.
    bool hasValue() const
    {
      return hasValues(1);
    }

    /// Returns true if this return value has exactly `n` values.
    bool hasValues(size_t n = 1) const
    {
      return values.size() == n;
    }

    /// Access to the single value held, asserts if none or more than one.
    template<class T>
    T get() const
    {
      assert(values.size() > 0 && "Access to empty return value");
      assert(values.size() == 1 && "Direct access to multiple values");
      return std::get<T>(values[0]);
    }

    /// Access a specific value held, asserts if none or not enough values.
    template<class T>
    T get(size_t n) const
    {
      assert(values.size() > 0 && "Access to empty return value");
      assert(values.size() > n && "Not enough values");
      return std::get<T>(values[n - 1]);
    }

    /// Returns a copy of the values.
    llvm::ArrayRef<ResultTy> getAll() const
    {
      assert(values.size() > 0 && "Access to empty return value");
      return values;
    }

    /// Add elements to the list
    void push_back(Value& value)
    {
      values.push_back(value);
    }
  };

  /**
   * Field offset maps between field names, types and their relative position.
   */
  struct FieldOffset
  {
    llvm::SmallVector<llvm::StringRef, 1> fields;
    llvm::SmallVector<Type, 1> types;
  };

  /**
   * MLIR Generator.
   */
  class Generator
  {
    Generator(MLIRContext* context) : builder(context) {}

    /// MLIR module.
    OwningModuleRef module;

    /// MLIR builder.
    OpBuilder builder;

    /// Symbol tables for variables.
    SymbolTableT symbolTable;
    /// Map for each type which fields does it have.
    std::map<llvm::StringRef, FieldOffset> classFields;

    /// Function scope, for mangling names. 3 because there will always be the
    /// root module, the current module and a class, at the very least.
    llvm::SmallVector<llvm::StringRef, 3> functionScope;

    /// AST aliases
    using Ast = ::verona::parser::Ast;
    using AstPath = ::verona::parser::AstPath;
    /// Name of the root module
    constexpr static const char* rootModuleName = "__";

    // ===================================================== Helpers
    // Methods for symbols, location and other helpers for building
    // MLIR nodes.

    /// Get location of an ast node.
    Location getLocation(Ast ast);

    /// Convert (promote/demote) the value to the specified type. This
    /// automatically chooses promotion / demotion based on the types involved.
    Value typeConversion(Value val, Type ty);

    /// Promote the smallest (compatible) type and return the values to be used
    /// for arithmetic operations. If types are same, just return them, if not,
    /// return the cast operations that make them the same. If types are
    /// incompatible, assert.
    std::pair<Value, Value> typePromotion(Value lhs, Value rhs);

    /// Mangle function names. If scope is not passed, use functionScope.
    std::string mangleName(
      llvm::StringRef name, llvm::ArrayRef<llvm::StringRef> scope = {});

    /// Return the offset into the structure to load/store values into fields
    /// and the type of the field's value (if stored in a different container).
    std::tuple<size_t, Type, bool>
    getField(Type type, llvm::StringRef fieldName);

    // =================================================================
    // Parsers These methods parse the AST into MLIR constructs, then either
    // return the expected MLIR value or call the generators (see below) to do
    // that for them.

    /// Parses the top module.
    llvm::Error parseRootModule(Ast ast);

    /// Parse a class declaration
    llvm::Error parseClass(Ast ast);

    /// Generic node parser, calls other parse functions to handle each
    /// individual type.
    llvm::Expected<ReturnValue> parseNode(Ast ast);

    /// Parses a function definition.
    llvm::Expected<ReturnValue> parseFunction(Ast ast);

    /// Parses a field definition.
    llvm::Expected<Type> parseField(Ast ast);

    /// Parses a lambda (function body).
    llvm::Expected<ReturnValue> parseLambda(Ast ast);

    /// Parses a select statement.
    llvm::Expected<ReturnValue> parseSelect(Ast ast);

    /// Parses a variable reference.
    llvm::Expected<ReturnValue> parseRef(Ast ast);

    /// Parses a let/var binding.
    llvm::Expected<ReturnValue> parseLocalDecl(Ast ast);

    /// Parses a type declaration.
    llvm::Expected<ReturnValue> parseOfType(Ast ast);

    /// Parses a variable assignment.
    llvm::Expected<ReturnValue> parseAssign(Ast ast);

    /// Parses a literal.
    llvm::Expected<ReturnValue> parseLiteral(Ast ast);

    /// Parses a type.
    Type parseType(Ast ast);

    // ===================================================== MLIR Generators
    /// Generate a prototype, populating the symbol table
    llvm::Expected<FuncOp> generateProto(
      Location loc,
      llvm::StringRef name,
      llvm::ArrayRef<Type> types,
      llvm::ArrayRef<Type> retTy);
    /// Generates an empty function (with the first basic block)
    llvm::Expected<FuncOp> generateEmptyFunction(
      Location loc,
      llvm::StringRef name,
      llvm::ArrayRef<llvm::StringRef> args,
      llvm::ArrayRef<Type> types,
      llvm::ArrayRef<Type> retTy);
    /// Generates a call to a static function (FIXME: implement dynamic calls)
    llvm::Expected<Value>
    generateCall(Location loc, FuncOp func, llvm::ArrayRef<Value> args);
    /// Generates arithmetic based on param types and op names
    llvm::Expected<Value> generateArithmetic(
      Location loc, llvm::StringRef opName, Value lhs, Value rhs);
    /// Generates an alloca (stack variable)
    Value generateAlloca(Location loc, Type ty);
    /// Generates an element pointer
    Value generateGEP(Location loc, Value addr, int offset = 0);
    /// Generates a load of an address
    Value generateLoad(Location loc, Value addr, int offset = 0);
    /// Generates a load if the expected type is not a pointer and is compatible
    /// with the element type (asserts if not)
    Value generateAutoLoad(Location loc, Value addr, Type ty, int offset = 0);
    /// Generates a store into an address
    void generateStore(Location loc, Value addr, Value val, int offset = 0);
    /// Generate a constant value of a certain type
    Value generateConstant(Type ty, std::variant<int, double> val);
    /// Generate a zero initialized value of a certain type
    Value generateZero(Type ty);

  public:
    /**
     * Convert an AST into a high-level MLIR module.
     */
    static llvm::Expected<OwningModuleRef> lower(MLIRContext* context, Ast ast);
  };
}
