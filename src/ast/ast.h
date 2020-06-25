// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include <peglib.h>

namespace ast
{
  struct SymbolScope;
  struct Annotation;

  using Scope = std::shared_ptr<SymbolScope>;
  using AstImpl = peg::AstBase<Annotation>;
  using Ast = std::shared_ptr<AstImpl>;
  using WeakAst = std::weak_ptr<AstImpl>;
  using Ident = std::string;
  using Tag = unsigned int;

  struct Annotation
  {
    Scope scope;
  };

  struct SymbolScope
  {
    std::map<Ident, WeakAst> sym;
  };

  Ast token(const Ast& ast, const char* name, const std::string& token);
  Ast node(const Ast& ast, const char* name);
  void push_back(Ast& ast, Ast& child);
  void replace(Ast& prev, Ast next);
  void remove(Ast ast);
  void rename(Ast& ast, const char* name);
  void elide(Ast& ast);

  Ast get_closest(Ast ast, Tag tag);
  Ast get_scope(Ast ast);
  Ast get_expr(Ast ast);
  Ast get_def(Ast ast, Ident id);
  Ast get_prev_in_expr(Ast ast);
  Ast get_next_in_expr(Ast ast);

  Ast& iteration_parent();
  size_t& iteration_index();

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
