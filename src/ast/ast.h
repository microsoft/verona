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

  Ast from(const Ast& ast, const char* name, const std::string& token);
  void replace(Ast& prev, Ast next);
  void remove(Ast ast);

  Ast get_closest(Ast ast, Tag tag);
  Ast get_scope(Ast ast);
  Ast get_expr(Ast ast);
  Ast get_def(Ast ast, Ident id);
  Ast get_prev_in_expr(Ast ast);
  Ast get_next_in_expr(Ast ast);
}
