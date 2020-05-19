#include "ast.h"

namespace ast
{
  using namespace peg::udl;

  Ast from(const Ast& ast, const char* name, const std::string& token)
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

  void replace(Ast& prev, Ast next)
  {
    auto parent = prev->parent.lock();

    if (parent)
    {
      auto find = std::find(parent->nodes.begin(), parent->nodes.end(), prev);
      assert(find != parent->nodes.end());
      *find = next;
      next->parent = parent;
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
    }
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
    while (ast && (ast->tag != "expr"_) && (ast->tag != "term"_))
      ast = ast->parent.lock();

    return ast;
  }

  Ast get_def(Ast ast, Ident id)
  {
    auto pos = ast->position;

    while ((ast = get_scope(ast)))
    {
      auto scope = ast->scope;
      auto find = scope->sym.find(id);

      if (find != scope->sym.end())
      {
        auto def = find->second.lock();

        if (def->position < pos)
          return def;
      }

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
