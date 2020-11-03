// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "ast.h"

namespace
{
  auto child_pos(::ast::Ast& parent, ::ast::Ast& child)
  {
    auto pos = std::find(parent->nodes.begin(), parent->nodes.end(), child);
    assert(pos != parent->nodes.end());
    return pos;
  }
}

namespace ast
{
  // Include PEG's ""_ operator for string-switches
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

  void insert_before(Ast& child, Ast& before)
  {
    auto parent = before->parent.lock();
    assert(parent);
    parent->nodes.insert(child_pos(parent, before), child);
    child->parent = parent;
  }

  void move_back(Ast& ast, Ast& child)
  {
    auto prev = child->parent.lock();
    assert(ast);
    push_back(ast, child);
    if (prev)
      prev->nodes.erase(child_pos(prev, child));
  }

  void move_before(Ast& child, Ast& before)
  {
    auto prev = child->parent.lock();
    assert(prev);
    insert_before(child, before);
    if (prev)
      prev->nodes.erase(child_pos(prev, child));
  }

  void move_children(Ast& from, Ast& to)
  {
    for (auto node : from->nodes)
      push_back(to, node);
    from->nodes.clear();
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

      if (iteration_parent() == parent)
      {
        // If we are currently iterating over the nodes in parent, we need to
        // back up the iteration index by one, but only if the node we are
        // removing is less than or equal to the current iteration index. This
        // is to allow removing children of parent without invalidating the
        // iterator.
        size_t i = std::distance(parent->nodes.begin(), find);

        if (i <= iteration_index())
          iteration_index()--;
      }

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

  std::string hygienic_id(Ast ast, const char* prefix)
  {
    auto scope = get_closest(ast, "classdef"_);
    std::string id{prefix};
    id += "$" + std::to_string(scope->id++);
    return id;
  }

  Ast& iteration_parent()
  {
    // This is only used as a helper in ast::for_each to allow removing child
    // nodes of parent without invalidating the iterator.
    thread_local Ast parent;
    return parent;
  }

  size_t& iteration_index()
  {
    // This is only used as a helper in ast::for_each to allow removing child
    // nodes of parent without invalidating the iterator.
    thread_local size_t index;
    return index;
  }
}

void ast_dump(const ::ast::Ast& ast)
{
  std::cout << peg::ast_to_s(ast);
}
