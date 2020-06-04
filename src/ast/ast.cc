// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "ast.h"

namespace ast
{
  using namespace peg::udl;

  Ast token(const Ast& ast, const char* name, const std::string& token)
  {
    return std::make_shared<AstImpl>(
      ast->path.c_str(),
      ast->line,
      ast->column,
      name,
      token,
      ast->position,
      ast->length);
  }

  Ast node(const Ast& ast, const char* name)
  {
    std::vector<Ast> none;

    return std::make_shared<AstImpl>(
      ast->path.c_str(),
      ast->line,
      ast->column,
      name,
      none,
      ast->position,
      ast->length,
      ast->choice_count,
      ast->choice);
  }

  void push_back(Ast& ast, Ast& child)
  {
    ast->nodes.push_back(child);
    child->parent = ast;
  }

  void replace(Ast& prev, Ast next)
  {
    auto parent = prev->parent.lock();

    if (parent)
    {
      auto find = std::find(parent->nodes.begin(), parent->nodes.end(), prev);
      assert(find != parent->nodes.end());
      next->parent = parent;
      prev->parent.reset();
      *find = next;
    }

    prev = next;
  }

  void remove(Ast ast)
  {
    auto parent = ast->parent.lock();

    if (parent)
    {
      auto find = std::find(parent->nodes.begin(), parent->nodes.end(), ast);
      assert(find != parent->nodes.end());
      parent->nodes.erase(find);
      ast->parent.reset();
    }
  }

  void rename(Ast& ast, const char* name)
  {
    Ast next;

    if (ast->is_token)
    {
      next = token(ast, name, ast->token);
    }
    else
    {
      next = node(ast, name);

      for (auto& node : ast->nodes)
        push_back(next, node);

      ast->nodes.clear();
    }

    replace(ast, next);
  }

  void elide(Ast& ast)
  {
    assert(ast->nodes.size() == 1);
    auto child = ast->nodes.front();
    ast::replace(ast, child);
  }

  Ast get_closest(Ast ast, Tag tag)
  {
    while (ast && (ast->tag != tag))
      ast = ast->parent.lock();

    return ast;
  }

  Ast get_scope(Ast ast)
  {
    while (ast && !ast->scope)
      ast = ast->parent.lock();

    return ast;
  }

  Ast get_expr(Ast ast)
  {
    while (ast && (ast->tag != "expr"_))
      ast = ast->parent.lock();

    return ast;
  }

  Ast get_def(Ast ast, Ident id)
  {
    while ((ast = get_scope(ast)))
    {
      auto scope = ast->scope;
      auto find = scope->sym.find(id);

      if (find != scope->sym.end())
        return find->second.lock();

      ast = ast->parent.lock();
    }

    return {};
  }

  Ast get_prev_in_expr(Ast ast)
  {
    auto expr = get_expr(ast);

    if (expr)
    {
      auto find = std::find(expr->nodes.begin(), expr->nodes.end(), ast);
      assert(find != expr->nodes.end());

      if (find != expr->nodes.begin())
        return *(find - 1);
    }

    return {};
  }

  Ast get_next_in_expr(Ast ast)
  {
    auto expr = get_expr(ast);

    if (expr)
    {
      auto find = std::find(expr->nodes.begin(), expr->nodes.end(), ast);
      assert(find != expr->nodes.end());
      ++find;

      if (find != expr->nodes.end())
        return *find;
    }

    return {};
  }
}
