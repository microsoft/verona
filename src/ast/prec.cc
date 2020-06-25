// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "prec.h"

using namespace peg::udl;

namespace
{
  void add_typeargs(ast::Ast& ast, ast::Ast& typeargs)
  {
    // Adds a typeargs node to ast and moves to the next node in the expr.
    // If it isn't a typeargs node, adds an empty typeargs node and does not
    // advance to the next node in the expr.
    if (typeargs && (typeargs->tag == "typeargs"_))
    {
      auto next = ast::get_next_in_expr(typeargs);
      ast::remove(typeargs);
      ast::push_back(ast, typeargs);
      typeargs = next;
    }
    else
    {
      auto emptyargs = ast::node(ast, "typeargs");
      ast::push_back(ast, emptyargs);
    }
  }

  void staticcall_rule(ast::Ast& ast, err::Errors& err)
  {
    // static-call <-
    //  typeref typargs? (lookup-typeref typeargs?)*
    //  (lookup typeargs?)? tuple?
    // (static-call qualtype function typeargs (args ...))
    assert(ast->tag == "typeref"_);
    auto typeref = ast;
    auto call = ast::node(ast, "static-call");
    ast::replace(ast, call);

    auto qualtype = ast::node(ast, "qualtype");
    ast::push_back(qualtype, typeref);
    ast::push_back(ast, qualtype);

    auto next = ast::get_next_in_expr(ast);
    add_typeargs(qualtype, next);

    // look for type lookups followed by optional typeargs
    auto def = ast::get_def(ast, typeref->token);

    while (next)
    {
      if (next->tag != "lookup"_)
        break;

      auto subdef = ast::get_def(def, next->token);

      if (
        !subdef ||
        ((subdef->tag != "typedef"_) && (subdef->tag != "classdef"_)))
      {
        break;
      }

      ast::remove(next);
      ast::rename(next, "typeref");
      ast::push_back(qualtype, next);
      next = ast::get_next_in_expr(ast);
      add_typeargs(qualtype, next);
      def = subdef;
    }

    if (next && (next->tag == "lookup"_))
    {
      // function call
      ast::remove(next);
      ast::rename(next, "function");
      ast::push_back(ast, next);
      next = ast::get_next_in_expr(ast);
      add_typeargs(ast, next);
    }
    else
    {
      // create sugar
      auto create = ast::token(ast, "function", "create");
      ast::push_back(ast, create);
      auto typeargs = ast::node(ast, "typeargs");
      ast::push_back(ast, typeargs);
    }

    if (next && (next->tag == "tuple"_))
    {
      ast::remove(next);
      ast::rename(next, "args");
      ast::push_back(ast, next);
    }
    else
    {
      auto args = ast::node(ast, "args");
      ast::push_back(ast, args);
    }
  }

  bool obj_rule(ast::Ast& ast, err::Errors& err)
  {
    if (!ast)
      return false;

    switch (ast->tag)
    {
      case "typeref"_:
      {
        staticcall_rule(ast, err);
        break;
      }

      case "lookup"_:
      case "typeargs"_:
      {
        err << ast << "Expected an expression, got " << ast->name << "."
            << err::end;
        return false;
      }
    }

    return true;
  }

  bool member_rule(ast::Ast& ast, err::Errors& err)
  {
    // member <- obj (lookup / typeargs tuple? / tuple)*
    // lookup can't distinguish types from fields, so invocation is either
    // calling a method or calling `apply` on a field.
    if (!obj_rule(ast, err))
      return false;

    auto next = ast::get_next_in_expr(ast);

    while (next)
    {
      switch (next->tag)
      {
        case "lookup"_:
        {
          // (member obj lookup)
          auto obj = ast;
          auto member = ast::node(next, "member");
          ast::replace(ast, member);
          ast::push_back(ast, obj);
          ast::remove(next);
          ast::push_back(ast, next);
          next = ast::get_next_in_expr(ast);
          break;
        }

        case "typeargs"_:
        {
          // (invoke obj typeargs args)
          auto obj = ast;
          auto invoke = ast::node(next, "invoke");
          ast::replace(ast, invoke);
          ast::push_back(ast, obj);
          ast::remove(next);
          ast::push_back(ast, next);
          next = ast::get_next_in_expr(ast);

          if (next && (next->tag == "tuple"_))
          {
            ast::remove(next);
            ast::rename(next, "args");
            ast::push_back(ast, next);
          }
          else
          {
            // zero argument invoke
            auto args = ast::node(ast, "args");
            ast::push_back(ast, args);
          }
          break;
        }

        case "tuple"_:
        {
          // (invoke obj typeargs args)
          auto obj = ast;
          auto invoke = ast::node(next, "invoke");
          ast::replace(ast, invoke);
          ast::push_back(ast, obj);
          auto typeargs = ast::node(ast, "typeargs");
          ast::push_back(ast, typeargs);
          ast::remove(next);
          ast::rename(next, "args");
          ast::push_back(ast, next);
          break;
        }

        default:
          return true;
      }
    }

    return true;
  }

  bool prefix_rule(ast::Ast& ast, err::Errors& err)
  {
    // (call function typeargs obj (args))
    if (ast->tag != "op"_)
      return member_rule(ast, err);

    auto op = ast;
    auto call = ast::node(ast, "call");
    ast::replace(ast, call);
    ast::rename(op, "function");
    ast::push_back(ast, op);
    auto next = ast::get_next_in_expr(ast);
    add_typeargs(ast, next);

    if (!next)
    {
      err << ast << "Expected an argument to this prefix function." << err::end;
      return false;
    }

    if (!prefix_rule(next, err))
      return false;

    ast::remove(next);

    if ((next->tag == "tuple"_) && !next->nodes.empty())
    {
      auto obj = next->nodes.front();
      ast::remove(obj);
      ast::push_back(ast, obj);
      ast::rename(next, "args");
      ast::push_back(ast, next);
    }
    else
    {
      ast::push_back(ast, next);
      auto args = ast::node(ast, "args");
      ast::push_back(ast, args);
    }

    return true;
  }

  bool infix_rule(ast::Ast& ast, err::Errors& err)
  {
    // (call function typeargs lhs (args rhs))
    // Infix operators are left associative.
    if (!prefix_rule(ast, err))
      return false;

    auto next = ast::get_next_in_expr(ast);
    std::string op;

    while (next)
    {
      if (next->tag == "assign"_)
        return true;

      if (next->tag != "op"_)
      {
        err << next << "Expected an infix operator here." << err::end;
        return false;
      }

      if (op.empty())
      {
        op = next->token;
      }
      else if (op != next->token)
      {
        err << next << "Use parentheses to establish precedence between " << op
            << " and " << next->token << " infix operators." << err::end;
      }

      auto lhs = ast;
      auto call = ast::node(ast, "call");
      ast::replace(ast, call);
      ast::remove(next);
      ast::rename(next, "function");
      ast::push_back(ast, next);
      next = ast::get_next_in_expr(ast);
      add_typeargs(ast, next);
      ast::push_back(ast, lhs);
      auto args = ast::node(ast, "args");
      ast::push_back(ast, args);

      if (!next)
      {
        err << ast << "Expected an expression after this infix operator."
            << err::end;
        return false;
      }

      if (!prefix_rule(next, err))
        return false;

      ast::remove(next);
      ast::push_back(args, next);
      next = ast::get_next_in_expr(ast);
    }

    return true;
  }

  bool assign_rule(ast::Ast& ast, err::Errors& err)
  {
    // (assign lhs rhs)
    // Assignment is right associative.
    if (!infix_rule(ast, err))
      return false;

    auto next = ast::get_next_in_expr(ast);

    if (!next)
      return true;

    if (next->tag != "assign"_)
    {
      err << next << "Expected an assignment or the end of an expression here."
          << err::end;
      return false;
    }

    auto lhs = ast;
    auto assign = ast::node(ast, "assign");
    ast::replace(ast, assign);
    ast::push_back(ast, lhs);
    ast::remove(next);
    next = ast::get_next_in_expr(ast);

    if (!next)
    {
      err << ast << "Expected an expression after this assignment." << err::end;
      return false;
    }

    if (!assign_rule(next, err))
      return false;

    ast::remove(next);
    ast::push_back(ast, next);
    return true;
  }

  void controlflow_rule(ast::Ast& ast, err::Errors& err)
  {
    switch (ast->tag)
    {
      case "break"_:
      case "continue"_:
        break;

      case "return"_:
      case "yield"_:
      {
        auto control = ast::node(ast, ast->name.c_str());
        ast::replace(ast, control);
        auto next = ast::get_next_in_expr(ast);

        if (next && assign_rule(next, err))
        {
          ast::remove(next);
          ast::push_back(ast, next);
        }
        break;
      }

      default:
      {
        assign_rule(ast, err);
        break;
      }
    }
  }

  void interp_rule(ast::Ast& ast)
  {
    auto parent = ast->parent.lock();

    if (!parent || (parent->tag != "interp_string"_))
      return;

    // (call (function 'string') (typeargs) obj (args))
    auto call = ast::node(ast, "call");
    auto fun = ast::token(ast, "function", "string");
    ast::push_back(call, fun);
    auto typeargs = ast::node(ast, "typeargs");
    ast::push_back(call, typeargs);
    auto obj = ast->nodes.front();
    ast::remove(obj);
    ast::push_back(call, obj);
    auto args = ast::node(ast, "args");
    ast::push_back(call, args);
    ast::push_back(ast, call);
  }
}

namespace prec
{
  void build(ast::Ast& ast, err::Errors& err)
  {
    // static-call <-
    //  typeref typargs? (lookup-typeref typeargs?)* (lookup typeargs?)? tuple?
    // obj <-
    //  let / localref / tuple / new / lambda / literal / blockexpr /
    //  static-call
    // member <- obj (lookup / typeargs tuple? / tuple)*
    // prefix-call <- (op typeargs? prefix-call) / member
    // infix-call <- prefix-call (op typeargs? prefix-call)*
    // assign <- infix-call ('=' assign)?
    // expr <- break / continue / return assign? / yield assign? / assign
    switch (ast->tag)
    {
      case "expr"_:
      {
        if (ast->nodes.empty())
        {
          ast::remove(ast);
          return;
        }

        controlflow_rule(ast->nodes.front(), err);
        interp_rule(ast);

        if (ast->nodes.size() == 1)
        {
          // Elide expr nodes that have a single element.
          ast::elide(ast);
          build(ast, err);
          return;
        }
        break;
      }

      case "tuple"_:
      {
        if (ast->nodes.size() == 1)
        {
          // Elide tuple nodes that have a single expression in them.
          ast::elide(ast);
          build(ast, err);
          return;
        }
        break;
      }
    }

    ast::for_each(ast, build, err);
  }
}
