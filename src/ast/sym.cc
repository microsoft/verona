// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "sym.h"

using namespace peg::udl;

namespace
{
  void add_scope(ast::Ast& ast)
  {
    ast->scope = std::make_shared<ast::SymbolScope>();
  }

  void add_symbol(ast::Ident id, ast::Ast& ast, err::Errors& err)
  {
    auto enclosing = ast::get_scope(ast);

    if (!enclosing)
    {
      err << ast << id << " has no enclosing scope." << err::end;
      return;
    }

    auto scope = enclosing->scope;
    auto find = scope->sym.find(id);

    if (find != scope->sym.end())
    {
      auto prev = find->second.lock();
      err << ast << id << " is already defined." << prev
          << "Previous definition of " << id << " is here." << err::end;
    }

    scope->sym.emplace(id, ast);
  }

  void resolve_ref(ast::Ast& ast, err::Errors& err)
  {
    assert(ast->tag == "ref"_);
    auto def = ast::get_def(ast, ast->token);

    if (!def)
    {
      ast::rename(ast, "op");
    }
    else
    {
      switch (def->tag)
      {
        case "typedef"_:
        case "typeparam"_:
        {
          ast::rename(ast, "typeref");
          break;
        }

        case "field"_:
        case "function"_:
        {
          // could be a member ref if implicit self access is allowed
          ast::rename(ast, "op");
          break;
        }

        case "namedparam"_:
        case "local"_:
        {
          // TODO: use before def
          ast::rename(ast, "localref");
          break;
        }

        default:
        {
          assert(0);
        }
      }
    }
  }

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

  template<typename T>
  void for_each(ast::Ast ast, err::Errors& err, T f)
  {
    if (!ast)
      return;

    for (decltype(ast->nodes.size()) i = 0; i < ast->nodes.size(); i++)
    {
      auto node = ast->nodes[i];
      f(node, err);
    }
  }

  void elide(ast::Ast& ast, err::Errors& err)
  {
    assert(ast->nodes.size() == 1);
    auto child = ast->nodes[0];
    ast::replace(ast, child);
  }

  void only_atom(ast::Ast& ast, err::Errors& err)
  {
    auto expr = ast::get_expr(ast);

    if (!expr || (expr->nodes.size() > 1))
    {
      err << ast << ast->name << " must be the only element of an expression."
          << err::end;
    }
  }

  void first_atom(ast::Ast& ast, err::Errors& err)
  {
    auto expr = ast::get_expr(ast);

    if (!expr || (expr->nodes[0] != ast))
    {
      err << ast << ast->name << " must be the first element of an expression."
          << err::end;
    }
  }

  void not_in_pattern(ast::Ast& ast, err::Errors& err)
  {
    if (ast::get_closest(ast, "pattern"_))
      err << ast << ast->token << " cannot appear in a pattern." << err::end;
  }

  void last_in_block(ast::Ast& ast, err::Errors& err)
  {
    auto block = ast::get_closest(ast, "block"_);

    if (block)
    {
      auto seq = block->nodes[0];
      auto expr = seq->nodes.back();

      if (expr == ast::get_expr(ast))
        return;
    }

    err << ast << ast->token << " must be the last expression in a block."
        << err::end;
  }

  size_t hex(const std::string& src, size_t& i, size_t len)
  {
    size_t r = 0;

    for (size_t count = 0; count < len; count++)
    {
      if (i >= (src.size() - 1))
        return r;

      auto c = src[i + 1];

      if ((c >= '0') && (c <= '9'))
        c = c - '0';
      else if ((c >= 'A') && (c <= 'F'))
        c = c - 'A' + 10;
      else if ((c >= 'a') && (c <= 'f'))
        c = c - 'a' + 10;
      else
        return r;

      i++;
      r = (r << 4) + c;
    }

    return r;
  }

  std::string utf8(size_t v)
  {
    std::string s;

    if (v <= 0x7f)
    {
      s.push_back(v & 0x7f);
    }
    else if (v <= 0x7FF)
    {
      s.push_back(0xc0 | (v >> 6));
      s.push_back(0x80 | (v & 0x3f));
    }
    else if (v <= 0xffff)
    {
      s.push_back(0xe0 | (v >> 12));
      s.push_back(0x80 | ((v >> 6) & 0x3f));
      s.push_back(0x80 | (v & 0x3f));
    }
    else if (v < 0x10ffff)
    {
      s.push_back(0xf0 | (v >> 18));
      s.push_back(0x80 | ((v >> 12) & 0x3f));
      s.push_back(0x80 | ((v >> 6) & 0x3f));
      s.push_back(0x80 | (v & 0x3f));
    }

    return s;
  }

  std::string escape(const std::string& src)
  {
    std::string dst;

    for (size_t i = 0; i < src.size(); i++)
    {
      auto c = src[i];

      if (c != '\\')
      {
        dst.push_back(c);
        continue;
      }

      auto n = src[++i];

      switch (n)
      {
        case 'a':
        {
          dst.push_back('\a');
          break;
        }

        case 'b':
        {
          dst.push_back('\b');
          break;
        }

        case 'e':
        {
          dst.push_back('\e');
          break;
        }

        case 'f':
        {
          dst.push_back('\f');
          break;
        }

        case 'n':
        {
          dst.push_back('\n');
          break;
        }

        case 'r':
        {
          dst.push_back('\r');
          break;
        }

        case 't':
        {
          dst.push_back('\t');
          break;
        }

        case 'v':
        {
          dst.push_back('\v');
          break;
        }

        case '\\':
        {
          dst.push_back('\\');
          break;
        }

        case '0':
        {
          dst.push_back('\0');
          break;
        }

        case 'x':
        {
          dst.append(utf8(hex(src, i, 2)));
          break;
        }

        case 'u':
        {
          dst.append(utf8(hex(src, i, 4)));
          break;
        }

        case 'U':
        {
          dst.append(utf8(hex(src, i, 6)));
          break;
        }

        default:
        {
          dst.push_back(n);
          break;
        }
      }
    }

    return dst;
  }

  void mangle_indent(ast::Ast& ast)
  {
    if (ast->nodes.empty())
      return;

    // Remove a leading blank line if one exists.
    auto node = ast->nodes.front();

    if (node->tag == "quote"_)
    {
      auto pos = node->token.find_first_not_of(" \f\r\t\v");

      if ((pos != std::string::npos) && (node->token[pos] == '\n'))
      {
        if (pos == (node->token.size() - 1))
        {
          ast::remove(node);

          if (ast->nodes.empty())
            return;
        }
        else
        {
          auto trim = node->token.substr(pos + 1);
          auto quote = ast::token(node, "quote", trim);
          ast::replace(node, quote);
        }
      }
    }

    // Remove a trailing blank line if one exists.
    node = ast->nodes.back();

    if (node->tag == "quote"_)
    {
      auto pos = node->token.find_last_not_of(" \f\r\t\v");

      if ((pos != std::string::npos) && (node->token[pos] == '\n'))
      {
        if (pos == 0)
        {
          ast::remove(node);
 
          if (ast->nodes.empty())
            return;
        }
        else
        {
          auto trim = node->token.substr(0, pos);
          auto quote = ast::token(node, "quote", trim);
          ast::replace(node, quote);
        }
      }
    }

    // If the first node is not a quote, the string starts with an unquote that
    // is not indented, so we need no indentation mangling.
    if (ast->nodes[0]->tag != "quote"_)
      return;

    // Calculate indentation.
    size_t indent = std::numeric_limits<size_t>::max();

    for (size_t i = 0; i < ast->nodes.size(); i++)
    {
      node = ast->nodes[i];

      if (node->tag != "quote"_)
        continue;

      // Start from the first character only on the first node. Otherwise start
      // from right after the first newline.
      size_t prev = 0;

      if (i > 0)
      {
        prev = node->token.find("\n");

        if (prev != std::string::npos)
          prev++;
      }

      while (prev != std::string::npos)
      {
        auto pos = node->token.find_first_not_of(" \f\r\t\v", prev);

        // Entirely blank lines don't influence indentation.
        if (pos == std::string::npos)
          break;

        auto len = pos - prev;

        if (len < indent)
          indent = len;

        prev = node->token.find("\n", pos);

        if (prev != std::string::npos)
          prev++;
      }
    }

    // Trim indentation.
    if (indent > 0)
    {
      for (size_t i = 0; i < ast->nodes.size(); i++)
      {
        node = ast->nodes[i];

        if (node->tag != "quote"_)
          continue;

        // Start from the first character only on the first node. Otherwise
        // start from right after the first newline.
        std::string s;
        size_t prev = 0;

        if (i > 0)
        {
          prev = node->token.find("\n");

          if (prev == std::string::npos)
            continue;

          prev++;
          s.append(node->token.substr(0, prev));
        }

        while (prev != std::string::npos)
        {
          auto pos = node->token.find("\n", prev);

          if (pos != std::string::npos)
            pos++;

          if ((prev + indent) < node->token.size())
            s.append(node->token.substr(prev + indent, pos - indent));

          prev = pos;
        }

        auto quote = ast::token(node, "quote", s);
        ast::replace(node, quote);
      }
    }
  }
}

namespace sym
{
  void scope(ast::Ast& ast, err::Errors& err)
  {
    switch (ast->tag)
    {
      case "module"_:
      case "lambda"_:
      {
        add_scope(ast);
        break;
      }

      case "new"_:
      {
        if (ast->nodes.size() == 0)
          err << ast << "new requires type name or body" << err::end;

        add_scope(ast);
        break;
      }

      case "typedef"_:
      {
        add_symbol(ast->nodes[1]->token, ast, err);
        add_scope(ast);
        not_in_pattern(ast, err);
        break;
      }

      case "typeparam"_:
      {
        add_symbol(ast->nodes[0]->token, ast, err);
        break;
      }

      case "field"_:
      {
        add_symbol(ast->nodes[0]->token, ast, err);
        break;
      }

      case "function"_:
      {
        // A missing function name is sugar for "apply"
        if (ast->nodes[0]->nodes.size() == 0)
        {
          ast->nodes[0]->nodes.push_back(
            ast::token(ast->nodes[0], "id", "apply"));
        }

        auto node = ast->nodes[0]->nodes[0];
        add_symbol(node->token, ast, err);
        add_scope(ast);
        break;
      }

      case "namedparam"_:
      {
        add_symbol(ast->nodes[0]->token, ast, err);
        break;
      }

      case "term"_:
      {
        ast::rename(ast, "expr");
        scope(ast, err);
        return;
      }

      case "blockexpr"_:
      {
        elide(ast, err);
        scope(ast, err);
        return;
      }

      case "block"_:
      case "when"_:
      case "while"_:
      case "match"_:
      case "case"_:
      case "if"_:
      case "for"_:
      {
        add_scope(ast);
        not_in_pattern(ast, err);
        break;
      }

      case "atom"_:
      {
        elide(ast, err);

        if (ast->tag == "id"_)
          ast::rename(ast, "ref");

        scope(ast, err);
        return;
      }

      case "break"_:
      case "continue"_:
      {
        only_atom(ast, err);
        not_in_pattern(ast, err);
        last_in_block(ast, err);
        break;
      }

      case "return"_:
      case "yield"_:
      {
        first_atom(ast, err);
        not_in_pattern(ast, err);
        last_in_block(ast, err);
        break;
      }

      case "let"_:
      {
        first_atom(ast, err);

        for_each(ast, err, [](auto& node, auto& err) {
          if (node->tag == "id"_)
          {
            ast::rename(node, "local");
            add_symbol(node->token, node, err);
          }
        });
        break;
      }

      case "sym"_:
      {
        if (ast->token == ".")
        {
          auto next = ast::get_next_in_expr(ast);

          if (next && (next->tag == "atom"_))
          {
            auto id = next->nodes[0];

            if ((id->tag == "id"_) || (id->tag == "sym"_))
            {
              auto lookup = ast::token(id, "lookup", id->token);
              ast::replace(ast, lookup);
              ast::remove(next);
              break;
            }
          }

          err << ast << "lookup must be followed by a symbol or an identifier."
              << err::end;
        }
        else if (ast->token == "=")
        {
          ast::rename(ast, "assign");
        }
        else
        {
          ast::rename(ast, "op");
        }
        break;
      }

      case "string"_:
      {
        auto s = escape(ast->token);
        auto e = ast::token(ast, "string", s);
        ast::replace(ast, e);
        break;
      }

      case "interp_string"_:
      {
        mangle_indent(ast);
        break;
      }

      case "quote"_:
      {
        // Quote elements of an interpolated string are not escaped.
        ast::rename(ast, "string");
        break;
      }

      case "unquote"_:
      {
        if (ast->nodes[0]->tag == "%word"_)
        {
          auto ref = ast->nodes[0];
          auto expr = ast::node(ast, "expr");
          ast::replace(ast, expr);
          ast::rename(ref, "ref");
          ast::push_back(ast, ref);
        }
        else
        {
          elide(ast, err);
        }

        scope(ast, err);
        return;
      }
    }

    for_each(ast, err, scope);
  }

  void references(ast::Ast& ast, err::Errors& err)
  {
    switch (ast->tag)
    {
      case "ref"_:
      {
        resolve_ref(ast, err);
        break;
      }
    }

    for_each(ast, err, references);
  }

  void prec_staticcall(ast::Ast& ast, err::Errors& err)
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
    assert(def && (def->tag == "typedef"_));

    while (next)
    {
      if (next->tag != "lookup"_)
        break;

      auto subdef = ast::get_def(def, next->token);

      if (!subdef || (subdef->tag != "typedef"_))
        break;

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
      ast::push_back(call, next);
      next = ast::get_next_in_expr(ast);
      add_typeargs(call, next);
    }
    else
    {
      // create sugar
      auto create = ast::token(call, "function", "create");
      ast::push_back(call, create);
      auto typeargs = ast::node(ast, "typeargs");
      ast::push_back(call, typeargs);
    }

    if (next && (next->tag == "tuple"_))
    {
      ast::remove(next);
      ast::rename(next, "args");
      ast::push_back(call, next);
    }
    else
    {
      auto args = ast::node(call, "args");
      ast::push_back(call, args);
    }
  }

  void prec_obj(ast::Ast& ast, err::Errors& err)
  {
    if (ast && (ast->tag == "typeref"_))
      prec_staticcall(ast, err);
  }

  void prec_member(ast::Ast& ast, err::Errors& err)
  {
    // member <- obj (lookup / typeargs tuple? / tuple)*
    // lookup can't distinguish types from fields, so invocation is either
    // calling a method or calling `apply` on a field.
    prec_obj(ast, err);
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
          return;
      }
    }
  }

  void prec_prefix(ast::Ast& ast, err::Errors& err)
  {
    // (call function typeargs obj (args))
    if (ast && (ast->tag == "op"_))
    {
      auto op = ast;
      auto call = ast::node(ast, "call");
      ast::replace(ast, call);
      ast::rename(op, "function");
      ast::push_back(ast, op);
      auto next = ast::get_next_in_expr(ast);
      add_typeargs(ast, next);
      prec_prefix(next, err);

      if (!next)
      {
        err << ast << "Expected an argument to this prefix function."
            << err::end;
        return;
      }

      ast::remove(next);
      ast::push_back(ast, next);
      auto args = ast::node(ast, "args");
      ast::push_back(ast, args);
    }
    else
    {
      prec_member(ast, err);
    }
  }

  void prec_infix(ast::Ast& ast, err::Errors& err)
  {
    // (call function typeargs lhs (args rhs))
    // Infix operators are left associative.
    prec_prefix(ast, err);
    auto next = ast::get_next_in_expr(ast);
    std::string op;

    while (next)
    {
      if (next->tag == "assign"_)
        return;

      if (next->tag != "op"_)
      {
        err << next << "Expected an infix operator here." << err::end;
        return;
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
        return;
      }

      prec_prefix(next, err);

      if (next)
      {
        ast::remove(next);
        ast::push_back(args, next);
        next = ast::get_next_in_expr(ast);
      }
    }
  }

  void prec_assign(ast::Ast& ast, err::Errors& err)
  {
    // (assign lhs rhs)
    // Assignment is right associative.
    prec_infix(ast, err);
    auto next = ast::get_next_in_expr(ast);

    if (!next)
      return;

    if (next->tag != "assign"_)
    {
      err << next << "Expected an assignment or the end of an expression here."
          << err::end;
      return;
    }

    auto lhs = ast;
    auto assign = ast::node(ast, "assign");
    ast::replace(ast, assign);
    ast::push_back(ast, lhs);
    ast::remove(next);
    next = ast::get_next_in_expr(ast);
    prec_assign(next, err);

    if (!next)
    {
      err << ast << "Expected an expression after this assignment." << err::end;
      return;
    }

    ast::remove(next);
    ast::push_back(ast, next);
  }

  void precedence(ast::Ast& ast, err::Errors& err)
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
          return;

        auto expr = ast;
        ast = ast->nodes[0];

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
            prec_assign(next, err);
            ast::remove(next);
            ast::push_back(ast, next);
            break;
          }

          default:
          {
            prec_assign(ast, err);
            break;
          }
        }

        auto parent = expr->parent.lock();

        if (parent && (parent->tag == "interp_string"_))
        {
          // (call (function 'string') (typeargs) obj (args))
          auto call = ast::node(expr, "call");
          auto fun = ast::token(expr, "function", "string");
          ast::push_back(call, fun);
          auto typeargs = ast::node(expr, "typeargs");
          ast::push_back(call, typeargs);
          auto obj = expr->nodes[0];
          ast::remove(obj);
          ast::push_back(call, obj);
          auto args = ast::node(expr, "args");
          ast::push_back(call, args);
          ast::push_back(expr, call);
        }

        if (expr->nodes.size() == 1)
        {
          elide(expr, err);
          precedence(expr, err);
          return;
        }
        break;
      }

      case "tuple"_:
      {
        if (ast->nodes.size() == 1)
        {
          elide(ast, err);
          precedence(ast, err);
          return;
        }
        break;
      }
    }

    for_each(ast, err, precedence);
  }
}
