// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include "generator.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "parser/ast.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <stack>
#include <string>

namespace mlir::verona
{
  /**
   * Class information
   *
   * Keeps information on classes, and their respective fields (types, names).
   * This class is used for both keeping the context while building classes as
   * well as querying field types by name later in code generation.
   *
   * FIXME: Verona may play with field offset (ex. by size order), which needs
   * to be done *before* we access any field from code, ie. right after
   * declaration is finished, so the order in which they appear in the vector
   * can change but only once. MLIR doesn't allow to change the fields of a
   * declared structure, so this needs to be done before setting the
   * StructureType.
   */
  class ClassInfo
  {
    /// Bind name and type for fields. Position is given by its vector offset.
    struct Field
    {
      StringRef name;
      Type type;
    };

    /// The fields, by offset order
    SmallVector<Field> fields;

    /// The final MLIR structure type
    StructType type;

    /// Class name
    StringRef name;

    /// Reorder fields to make structures more efficient.
    void optimizeFields()
    {
      // TODO: Implement this.
    }

  public:
    /// Constructs an empty class with a name
    ClassInfo(MLIRContext* context, StringRef name) : name(name)
    {
      type = StructType::getIdentified(context, name);
    }

    /// Adds a field to the list before finalisation
    void addField(StringRef name, Type ty)
    {
      assert(!type.isInitialized() && "Can't change type after declaration");
      fields.push_back({name, ty});
    }

    /// Finalise structure and set MLIR type
    void finalize()
    {
      // First make sure we have the most optimal placing
      optimizeFields();

      // Now get the types in order and create the structure
      SmallVector<Type> types;
      for (auto field : fields)
        types.push_back(field.type);
      auto set = type.setBody(types, /*packed*/ false);
      // This really shouldn't fail
      assert(mlir::succeeded(set) && "Error setting fields to class");
    }

    /// FIXME: Find better map key than opaque pointer types
    typedef const void* KeyTy;

    /// Get the key to search maps
    KeyTy key()
    {
      return ClassInfo::key(type);
    }

    /// Get the key from some StructType to search maps
    static KeyTy key(StructType ty)
    {
      return ty.getAsOpaquePointer();
    }

    /// Get the structure type itself
    StructType getType()
    {
      return type;
    }

    /// Return the field type by name, or empty type if not found
    std::tuple<size_t, Type> getFieldType(StringRef name)
    {
      assert(type && "Can't yet determine final structure");
      size_t pos = 0;
      for (auto field : fields)
      {
        if (name == field.name)
          return {pos, field.type};
        pos++;
      }

      // Not found, return empty type
      return {pos, Type()};
    }
  };

  /**
   * AST Consumer
   */
  class ASTConsumer
  {
    /// This is temporary to make the passes work, we need to think of a better
    /// way out.
    friend struct ASTDeclarations;
    friend struct ASTDefinitions;

    ASTConsumer(MLIRContext* context) : gen(context) {}

    /// MLIR Generator
    MLIRGenerator gen;

    /// Map for each type which fields does it have.
    /// Use ClassInfo::key(structType) to get the key.
    std::unordered_map<ClassInfo::KeyTy, ClassInfo> classInfo;

    // ===================================================== Helpers

    /// Get builder from generator.
    OpBuilder& builder()
    {
      return gen.getBuilder();
    }

    /// Get symbol table from generator.
    SymbolTableT& symbolTable()
    {
      return gen.getSymbolTable();
    }

    /// Get location of an ast node.
    Location getLocation(::verona::parser::NodeDef& ast);

    /// Mangle function names. If scope is not passed, use functionScope.
    std::string mangleName(
      llvm::StringRef name,
      llvm::ArrayRef<llvm::StringRef> functionScope = {},
      llvm::ArrayRef<llvm::StringRef> callScope = {});

    /// Looks up a symbol with the ast's view.
    Value lookup(::verona::parser::Ast ast, bool lastContextOnly = false);

    /// Consumes a type.
    Type consumeType(::verona::parser::Type& ast);

  public:
    /**
     * Convert an AST into a high-level MLIR module.
     */
    static llvm::Expected<OwningModuleRef>
    lower(MLIRContext* context, ::verona::parser::Ast ast);
  };

  /*
   * Scope Cleanup helper
   *
   * Automatically invokes the callable object passed to the constructor on
   * destruction. This class is intended to provide lexically scoped cleanups,
   * for example:
   * ```c++
   * ScopedCleanup defer([&] { cleanup code here });
   * ```
   * The code in the lambda will be invoked when `defer` goes out of scope.
   */
  template<class T>
  class ScopeCleanup
  {
    /// Action to perform on destruction
    T cleanup;

  public:
    ScopeCleanup(T&& c) : cleanup(std::move(c)) {}

    /// Automatically applies the cleanup
    ~ScopeCleanup()
    {
      cleanup();
    }
  };
}
