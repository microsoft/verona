// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include <peglib.h>

namespace ast
{
  struct SymbolScope;
  struct Annotation;

  /// Ast nodes
  using AstImpl = peg::AstBase<Annotation>;
  using Ast = std::shared_ptr<AstImpl>;
  /// Weak reference to an Ast node
  using WeakAst = std::weak_ptr<AstImpl>;
  /// IDs: (variable, function, etc) names
  using Ident = std::string;
  /// Numeric tag from "name"_ operator
  using Tag = unsigned int;

  /// Annotation wrapper for AstBase with Symbol scope
  using Scope = std::shared_ptr<SymbolScope>;
  struct Annotation
  {
    Scope scope;
  };

  /// Symbol scope (ID -> Ast)
  struct SymbolScope
  {
    std::map<Ident, WeakAst> sym;
  };

  /// Create a new token (value node) with name and token value
  /// using a previous token's source location, position and length.
  Ast token(const Ast& ast, const char* name, const std::string& token);
  /// Create a new node with name and no children
  /// using a previous token's source location, position and length.
  Ast node(const Ast& ast, const char* name);
  /// Add a child to an existing node
  void push_back(Ast& ast, Ast& child);
  /// Replace prev with next node, updating prev's parent
  void replace(Ast& prev, Ast next);
  /// Remove node from its parent without invalidating existing iterators.
  void remove(Ast ast);
  /// Rename a node/token by creating a new node and replacing the old.
  void rename(Ast& ast, const char* name);
  /// Replaces node with single child with its child.
  void elide(Ast& ast);

  /// Find the closest ancestor with a specific tag, which could be itself.
  Ast get_closest(Ast ast, Tag tag);
  /// Find the closest ancestor with a scope, which could be itself.
  Ast get_scope(Ast ast);
  /// Find the closest 'expr' ancestor, which could be itself.
  Ast get_expr(Ast ast);
  /// Find the 'id' in any scope above the `ast` node. Returns empty Ast
  /// if not found.
  Ast get_def(Ast ast, Ident id);
  /// Find previous child in 'expr' parent.
  Ast get_prev_in_expr(Ast ast);
  /// Find next child in 'expr' parent.
  Ast get_next_in_expr(Ast ast);

  /// For exclusive use of 'for_each' function to avoid invalidating iterators.
  /// Returns the iteration's parent.
  Ast& iteration_parent();
  /// For exclusive use of 'for_each' function to avoid invalidating iterators.
  /// Returns the iteration's index.
  size_t& iteration_index();

  /// Apply function 'f(child, args...)' onto each child of 'ast'.
  template<typename Func, typename... Args>
  void for_each(Ast ast, Func f, Args&... args)
  {
    if (!ast)
      return;

    // This convoluted approach allows calling ast::remove on a node that is in
    // the current ast->nodes vector while maintaining the correct iteration
    // sequence.
    auto prev_parent = iteration_parent();
    auto prev_index = iteration_index();

    iteration_parent() = ast;
    iteration_index() = 0;

    while (iteration_index() < ast->nodes.size())
    {
      auto node = ast->nodes[iteration_index()];
      f(node, args...);
      iteration_index()++;
    }

    iteration_parent() = prev_parent;
    iteration_index() = prev_index;
  }
}

extern "C"
{
  /// This method is for debug only,
  /// when needing to dump ast trees inside a debugger.
  /// This is needed because `peg::ast_to_s` is templated.
  /// Needs to be in the global namespace to be found by LLDB
  void ast_dump(const ast::Ast& ast);
}
