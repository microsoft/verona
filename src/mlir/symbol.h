// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include "ast/ast.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

namespace mlir::verona
{
  /**
   * The symbol table has all declared symbols (with the original node
   * and the MLIR ounterpart) in a scope. Creating a new scope makes
   * all future insertions happen at that level, destroying it pops
   * the scope out of the stack.
   *
   * New scopes are created by creating a new local variable like:
   *   SymbolScopeT scope(symbolTable);
   * The destructor pops the scope automatically.
   *
   * We cannot use LLVM's ADT/ScopedHashTable like MLIR's Toy example because
   * that coped hash table does not allow redefinition, which is a problem when
   * declaring variables with a type only and then assigning values later.
   *
   * Also, note that because of how `lookup` returns nullptr, we can only store
   * pointers or objects that "behave like pointers" (by implementing *, ->).
   */
  template<class T>
  class ScopedTable
  {
    using MapTy = std::map<std::string, T>;
    std::vector<MapTy> stack;

  public:
    ScopedTable()
    {
      // Global scope
      pushScope();
    }

    ~ScopedTable()
    {
      // Global scope
      popScope();
      assert(stack.empty());
    }

    /// Insert entry on the last scope only
    /// Returns the inserted element
    /// Asserts if element already exist
    T insert(llvm::StringRef key, T value)
    {
      return getOrAdd(key, value, /* insert= */ true);
    }

    /// Fetch or insert the entry in the last scope
    /// Returns the inserted/fetched element
    /// If insert=true, asserts if element already exist
    T getOrAdd(llvm::StringRef key, T value, bool insert = false)
    {
      auto& frame = stack.back();
      auto res = frame.emplace(key, value);
      if (insert)
        assert(res.second && "Redeclaration");
      return std::get<1>(*res.first);
    }

    /// Return the entry if it is in the last scope
    T inScope(llvm::StringRef key)
    {
      return lookup(key, /* lastContextOnly= */ true);
    }

    /// Lookup for the entry on all scopes, from the last to first
    /// Returns the element or nullptr if none found
    /// `lastContextOnly=true` only looks up in the local scope
    T lookup(llvm::StringRef key, bool lastContextOnly = false)
    {
      for (auto it = stack.rbegin(), end = stack.rend(); it != end; it++)
      {
        auto& frame = *it;
        auto val = frame.find(key.str());
        if (val != frame.end())
          return val->second;
        if (lastContextOnly)
          return nullptr;
      }
      return nullptr;
    }

    /// Creates a new scope
    void pushScope()
    {
      stack.emplace_back();
    }

    /// Destroys the inner-most scope
    void popScope()
    {
      stack.pop_back();
    }
  };

  // FIXME: This is a hack to control scope. We can do better.
  template<class T>
  class ScopedTableScope
  {
    ScopedTable<T>& table;

  public:
    ScopedTableScope(ScopedTable<T>& table) : table(table)
    {
      table.pushScope();
    }
    ~ScopedTableScope()
    {
      table.popScope();
    }
  };

  /**
   * Variable symbols. New scopes should be created when entering classes,
   * functions, lexical blocks, lambdas, etc.
   */
  using SymbolTableT = ScopedTable<mlir::Value>;
  using SymbolScopeT = ScopedTableScope<mlir::Value>;

  /**
   * Function Symbols. New scopes should be created when entering classes
   * and sub-classes. Modules too, if we allow more than one per file.
   */
  using FunctionTableT = ScopedTable<mlir::FuncOp>;
  using FunctionScopeT = ScopedTableScope<mlir::FuncOp>;

  /**
   * Type Symbols. New scopes should be created when entering classes
   * sub-classes and functions, to be used with the 'where' keyword.
   */
  using TypeTableT = ScopedTable<mlir::Type>;
  using TypeScopeT = ScopedTableScope<mlir::Type>;

  /**
   * Basic Block Symbols. New scopes should be created when entering loops
   * to determine what is the head/exit block for 'break'/'continue'.
   */
  using BasicBlockTableT = ScopedTable<mlir::Block*>;
  using BasicBlockScopeT = ScopedTableScope<mlir::Block*>;
}
