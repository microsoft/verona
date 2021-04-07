// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "parser.h"

#include "dnf.h"
#include "escaping.h"
#include "ident.h"
#include "lookup.h"
#include "path.h"

#include <cassert>
#include <set>

namespace verona::parser
{
  constexpr auto ext = "verona";

  enum Result
  {
    Skip,
    Success,
    Error,
  };

  struct Parse
  {
    Source source;
    size_t pos;
    size_t la;
    Token previous;
    std::vector<Token> lookahead;

    AstPath symbol_stack;

    Ident ident;
    Location name_apply;

    Result final_result;
    std::vector<std::string> imports;
    std::string stdlib;
    std::ostream& out;

    struct SymbolPush
    {
      Parse& parser;

      SymbolPush(Parse& parser) : parser(parser) {}

      ~SymbolPush()
      {
        parser.pop();
      }
    };

    Parse(const std::string& stdlib, std::ostream& out)
    : pos(0), la(0), final_result(Success), stdlib(stdlib), out(out)
    {
      name_apply = ident("apply");
    }

    ~Parse()
    {
      assert(symbol_stack.size() == 0);
    }

    void start(Source& src)
    {
      source = src;
      pos = 0;
      la = 0;
      previous = {};
      lookahead.clear();
    }

    std::ostream& error()
    {
      final_result = Error;
      return out << "--------" << std::endl;
    }

    SymbolPush push(Ast node)
    {
      assert(node->symbol_table() != nullptr);
      symbol_stack.push_back(node);
      return SymbolPush(*this);
    }

    void pop()
    {
      symbol_stack.pop_back();
    }

    void set_sym(SymbolTable* st, const Location& id, Ast node)
    {
      auto prev = st->set(id, node);

      if (prev)
      {
        auto& loc = node->location;

        error() << loc << "There is a previous definition of \"" << id.view()
                << "\"" << text(loc) << prev.value()
                << "The previous definition is here" << text(prev.value());
      }
    }

    void set_sym(const Location& id, Ast node)
    {
      assert(symbol_stack.size() > 0);
      set_sym(symbol_stack.back()->symbol_table(), id, node);
    }

    void set_sym_parent(const Location& id, Ast node)
    {
      assert(symbol_stack.size() > 1);
      set_sym(symbol_stack[symbol_stack.size() - 2]->symbol_table(), id, node);
    }

    Node<Ref> ref(const Location& loc)
    {
      auto ref = std::make_shared<Ref>();
      ref->location = loc;
      return ref;
    }

    Location loc()
    {
      if (lookahead.size() > 0)
        return lookahead[0].location;
      else
        return previous.location;
    }

    text line()
    {
      return text(loc());
    }

    bool peek(const TokenKind& kind, const char* text = nullptr)
    {
      if (la >= lookahead.size())
        lookahead.push_back(lex(source, pos));

      assert(la < lookahead.size());

      if (lookahead[la].kind == kind)
      {
        if (!text || (lookahead[la].location == text))
        {
          next();
          return true;
        }
      }

      return false;
    }

    void next()
    {
      la++;
    }

    void rewind()
    {
      la = 0;
    }

    Token take()
    {
      assert(la == 0);

      if (lookahead.size() == 0)
        return lex(source, pos);

      previous = lookahead.front();
      lookahead.erase(lookahead.begin());
      return previous;
    }

    bool has(TokenKind kind, const char* text = nullptr)
    {
      assert(la == 0);

      if (peek(kind, text))
      {
        rewind();
        take();
        return true;
      }

      return false;
    }

    bool is_localref(const Location& id)
    {
      if (look_up_local(symbol_stack, id))
        return true;

      return false;
    }

    bool peek_delimited(TokenKind kind, TokenKind terminator)
    {
      // Look for `kind`, skipping over balanced () [] {}.
      while (!peek(TokenKind::End))
      {
        if (peek(kind))
          return true;

        if (peek(terminator))
          return false;

        if (peek(TokenKind::LParen))
        {
          peek_delimited(TokenKind::RParen, TokenKind::End);
        }
        else if (peek(TokenKind::LSquare))
        {
          peek_delimited(TokenKind::RSquare, TokenKind::End);
        }
        else if (peek(TokenKind::LBrace))
        {
          peek_delimited(TokenKind::RBrace, TokenKind::End);
        }
        else
        {
          next();
        }
      }

      return false;
    }

    void restart_before(const std::initializer_list<TokenKind>& kinds)
    {
      // Skip over balanced () [] {}
      while (!has(TokenKind::End))
      {
        for (auto& kind : kinds)
        {
          if (peek(kind))
          {
            rewind();
            return;
          }
        }

        if (has(TokenKind::LParen))
        {
          restart_before(TokenKind::RParen);
        }
        else if (has(TokenKind::LSquare))
        {
          restart_before(TokenKind::RSquare);
        }
        else if (has(TokenKind::LBrace))
        {
          restart_before(TokenKind::RBrace);
        }
        else
        {
          take();
        }
      }
    }

    void restart_after(const std::initializer_list<TokenKind>& kinds)
    {
      restart_before(kinds);
      take();
    }

    void restart_before(TokenKind kind)
    {
      restart_before({kind});
    }

    void restart_after(TokenKind kind)
    {
      restart_after({kind});
    }

    Result optident(Location& id)
    {
      if (!has(TokenKind::Ident))
        return Skip;

      id = previous.location;
      return Success;
    }

    Result optwhen(Node<Expr>& expr)
    {
      // when <- 'when' postfix lambda
      if (!has(TokenKind::When))
        return Skip;

      Result r = Success;
      auto when = std::make_shared<When>();
      auto st = push(when);
      when->location = previous.location;
      expr = when;

      if (optpostfix(when->waitfor) != Success)
      {
        error() << loc() << "Expected a when condition" << line();
        r = Error;
      }

      if (optlambda(when->behaviour) != Success)
      {
        error() << loc() << "Expected a when body" << line();
        r = Error;
      }

      return r;
    }

    Result opttry(Node<Expr>& expr)
    {
      // try <- 'try' lambda 'catch' '{' lambda* '}'
      if (!has(TokenKind::Try))
        return Skip;

      Result r = Success;
      auto tr = std::make_shared<Try>();
      auto st = push(tr);
      tr->location = previous.location;
      expr = tr;

      if (optlambda(tr->body) != Success)
      {
        error() << loc() << "Expected a try block" << line();
        r = Error;
      }

      auto& body = tr->body->as<Lambda>();

      if (!body.typeparams.empty())
      {
        error() << body.typeparams.front()->location
                << "A try block can't have type parameters"
                << text(body.typeparams.front()->location);
        r = Error;
      }

      if (!body.params.empty())
      {
        error() << body.params.front()->location
                << "A try block can't have parameters"
                << text(body.params.front()->location);
        r = Error;
      }

      if (!has(TokenKind::Catch))
      {
        error() << loc() << "Expected a catch block" << line();
        return Error;
      }

      if (!has(TokenKind::LBrace))
      {
        error() << loc() << "Expected a {" << line();
        return Error;
      }

      while (true)
      {
        Node<Expr> clause;
        Result r2;

        if ((r2 = optlambda(clause)) == Skip)
          break;

        tr->catches.push_back(clause);

        if (r2 == Error)
          r = Error;
      }

      if (!has(TokenKind::RBrace))
      {
        error() << loc() << "Expected a }" << line();
        return Error;
      }

      return r;
    }

    Result optmatch(Node<Expr>& expr)
    {
      // match <- 'match' postfix '{' lambda* '}'
      if (!has(TokenKind::Match))
        return Skip;

      Result r = Success;
      auto match = std::make_shared<Match>();
      auto st = push(match);
      match->location = previous.location;
      expr = match;

      if (optpostfix(match->test) != Success)
      {
        error() << loc() << "Expected a match test-expression" << line();
        r = Error;
      }

      if (!has(TokenKind::LBrace))
      {
        error() << loc() << "Expected { to start match cases" << line();
        return Error;
      }

      while (!has(TokenKind::RBrace))
      {
        if (has(TokenKind::End))
        {
          error() << loc() << "Expected a case or } to end match cases"
                  << line();
          r = Error;
          break;
        }

        Node<Expr> clause;
        Result r2 = optlambda(clause);

        if (r2 == Skip)
          break;

        match->cases.push_back(clause);

        if (r2 == Error)
          r = Error;
      }

      return r;
    }

    Result opttuple(Node<Expr>& expr)
    {
      // tuple <- '(' (expr (',' expr)*)? ')'
      if (!has(TokenKind::LParen))
        return Skip;

      auto tup = std::make_shared<Tuple>();
      tup->location = previous.location;
      expr = tup;

      if (has(TokenKind::RParen))
        return Success;

      Result r = Success;

      do
      {
        Node<Expr> elem;
        Result r2;

        if ((r2 = optexpr(elem)) == Skip)
          break;

        if (r2 == Error)
        {
          error() << loc() << "Expected an expression" << line();
          restart_before({TokenKind::Comma, TokenKind::RParen});
          r = Error;
        }

        tup->seq.push_back(elem);
      } while (has(TokenKind::Comma));

      if (!has(TokenKind::RParen))
      {
        error() << loc() << "Expected , or )" << line();
        r = Error;
      }

      return r;
    }

    Result optlambda(Node<Expr>& expr)
    {
      // lambda <-
      //  '{' (typeparams? (param (',' param)*)? '=>')? (expr ';'*)* '}'
      if (!has(TokenKind::LBrace))
        return Skip;

      auto lambda = std::make_shared<Lambda>();
      auto st = push(lambda);
      lambda->location = previous.location;
      expr = lambda;

      Result r = opttypeparams(lambda->typeparams);
      bool has_params = true;

      if (r == Skip)
      {
        has_params = peek_delimited(TokenKind::FatArrow, TokenKind::RBrace);
        r = Success;
        rewind();
      }

      if (has_params)
      {
        if (optparamlist(lambda->params, TokenKind::FatArrow) == Error)
          r = Error;

        if (!has(TokenKind::FatArrow))
        {
          error() << loc() << "Expected =>" << line();
          r = Error;
        }
      }

      while (!has(TokenKind::RBrace))
      {
        if (has(TokenKind::End))
        {
          error() << lambda->location << "Unexpected EOF in lambda body"
                  << line();
          return Error;
        }

        Node<Expr> expr;
        Result r2 = optexpr(expr);

        if (r2 == Skip)
          break;

        lambda->body.push_back(expr);

        if (r2 == Error)
          r = Error;

        while (has(TokenKind::Semicolon))
          ;
      }

      return r;
    }

    Result optref(Node<Expr>& expr)
    {
      // ref <- [local] ident oftype?
      if (!peek(TokenKind::Ident))
        return Skip;

      bool local = is_localref(lookahead[la - 1].location);
      rewind();

      if (!local)
        return Skip;

      if (!has(TokenKind::Ident))
        return Skip;

      auto ref = std::make_shared<Ref>();
      ref->location = previous.location;
      expr = ref;
      return Success;
    }

    Result optconstant(Node<Expr>& expr)
    {
      // constant <-
      //  escapedstring / unescapedstring / character /
      //  float / int / hex / binary / 'true' / 'false'
      if (has(TokenKind::EscapedString))
      {
        auto con = std::make_shared<EscapedString>();
        con->location = previous.location;
        expr = con;
      }
      else if (has(TokenKind::UnescapedString))
      {
        auto con = std::make_shared<UnescapedString>();
        con->location = previous.location;
        expr = con;
      }
      else if (has(TokenKind::Character))
      {
        auto con = std::make_shared<Character>();
        con->location = previous.location;
        expr = con;
      }
      else if (has(TokenKind::Int))
      {
        auto con = std::make_shared<Int>();
        con->location = previous.location;
        expr = con;
      }
      else if (has(TokenKind::Float))
      {
        auto con = std::make_shared<Float>();
        con->location = previous.location;
        expr = con;
      }
      else if (has(TokenKind::Hex))
      {
        auto con = std::make_shared<Hex>();
        con->location = previous.location;
        expr = con;
      }
      else if (has(TokenKind::Binary))
      {
        auto con = std::make_shared<Binary>();
        con->location = previous.location;
        expr = con;
      }
      else if (has(TokenKind::Bool))
      {
        auto con = std::make_shared<Bool>();
        con->location = previous.location;
        expr = con;
      }
      else
      {
        return Skip;
      }

      return Success;
    }

    Result objectliteral(Node<Expr>& expr)
    {
      // new <- 'new' (typebody / type typebody) ('@' ident)?
      Result r = Success;
      auto obj = std::make_shared<ObjectLiteral>();
      auto st = push(obj);
      obj->location = previous.location;
      expr = obj;

      if (has(TokenKind::Symbol, "@"))
      {
        if (optident(obj->in) != Success)
        {
          error() << loc() << "Expected an identifier" << line();
          r = Error;
        }
      }

      bool inherits = !peek(TokenKind::LBrace);
      rewind();

      if (inherits)
      {
        if (typeexpr(obj->inherits) == Error)
          r = Error;

        if (checkinherit(obj->inherits) == Error)
          r = Error;
      }

      if (typebody(obj->members) != Success)
        r = Error;

      return r;
    }

    Result optnew(Node<Expr>& expr)
    {
      // new <- 'new' ('@' ident)? (tuple / typebody / type typebody)
      if (!has(TokenKind::New))
        return Skip;

      bool ctor = peek(TokenKind::LParen) ||
        (peek(TokenKind::Symbol, "@") && peek(TokenKind::Ident) &&
         peek(TokenKind::LParen));
      rewind();

      if (!ctor)
        return objectliteral(expr);

      // ctor <- 'new' tuple ('@' ident)?
      Result r = Success;
      auto n = std::make_shared<New>();
      n->location = previous.location;
      expr = n;

      if (has(TokenKind::Symbol, "@"))
      {
        if (optident(n->in) != Success)
        {
          error() << loc() << "Expected an identifier" << line();
          r = Error;
        }
      }

      if (opttuple(n->args) != Success)
        r = Error;

      return r;
    }

    Result optatom(Node<Expr>& expr)
    {
      // atom <- tuple / constant / new / when / try / match / lambda
      Result r;

      if ((r = opttuple(expr)) != Skip)
        return r;

      if ((r = optconstant(expr)) != Skip)
        return r;

      if ((r = optnew(expr)) != Skip)
        return r;

      if ((r = optwhen(expr)) != Skip)
        return r;

      if ((r = opttry(expr)) != Skip)
        return r;

      if ((r = optmatch(expr)) != Skip)
        return r;

      if ((r = optlambda(expr)) != Skip)
        return r;

      return Skip;
    }

    Result opttypeargs(List<Type>& typeargs)
    {
      // typeargs <- '[' type (',' type)* ']'
      if (!has(TokenKind::LSquare))
        return Skip;

      Result r = Success;

      do
      {
        Node<Type> arg;

        if (typeexpr(arg) != Success)
        {
          error() << loc() << "Expected a type argument" << line();
          restart_before({TokenKind::Comma, TokenKind::RSquare});
          r = Error;
        }

        typeargs.push_back(arg);
      } while (has(TokenKind::Comma));

      if (!has(TokenKind::RSquare))
      {
        error() << loc() << "Expected , or ]" << line();
        r = Error;
      }

      return r;
    }

    Result optselector(Node<Expr>& expr)
    {
      // selector <- name typeargs? ('::' name typeargs?)*
      bool ok = peek(TokenKind::Ident) || peek(TokenKind::Symbol);
      rewind();

      if (!ok)
        return Skip;

      Result r = Success;

      auto sel = std::make_shared<Select>();
      sel->location = previous.location;
      sel->expr = expr;
      expr = sel;

      do
      {
        if (!has(TokenKind::Ident) && !has(TokenKind::Symbol))
        {
          error() << loc() << "Expected a selector name" << line();
          return Error;
        }

        auto name = std::make_shared<TypeName>();
        name->location = previous.location;
        sel->typenames.push_back(name);

        if (opttypeargs(name->typeargs) == Error)
          r = Error;
      } while (has(TokenKind::DoubleColon));

      return r;
    }

    Result optselect(Node<Expr>& expr)
    {
      // select <- '.' selector tuple?
      if (!has(TokenKind::Dot))
        return Skip;

      Result r = Success;

      if (optselector(expr) != Success)
      {
        error() << loc() << "Expected a selector" << line();
        r = Error;
      }

      if (opttuple(expr->as<Select>().args) == Error)
        r = Error;

      return r;
    }

    Result optapplysugar(Node<Expr>& expr)
    {
      // applysugar <- ref typeargs? tuple?
      Result r;

      if ((r = optref(expr)) == Skip)
        return r;

      bool ok = peek(TokenKind::LSquare) || peek(TokenKind::LParen);
      rewind();

      if (!ok)
        return r;

      auto sel = std::make_shared<Select>();
      sel->expr = expr;
      sel->location = expr->location;
      expr = sel;

      auto name = std::make_shared<TypeName>();
      name->location = name_apply;
      sel->typenames.push_back(name);

      if (opttypeargs(name->typeargs) == Error)
        r = Error;

      if (opttuple(sel->args) == Error)
        r = Error;

      return r;
    }

    Result optpostfixstart(Node<Expr>& expr)
    {
      // postfixstart <- atom / applysugar
      Result r;

      if ((r = optatom(expr)) != Skip)
        return r;

      if ((r = optapplysugar(expr)) != Skip)
        return r;

      return Skip;
    }

    Result optpostfix(Node<Expr>& expr)
    {
      // postfix <- postfixstart select*
      Result r;
      Result r2;

      if ((r = optpostfixstart(expr)) == Skip)
        return Skip;

      while ((r2 = optselect(expr)) != Skip)
      {
        if (r2 == Error)
          r = Error;
      }

      return r;
    }

    Result optinfix(Node<Expr>& expr)
    {
      // infix <- (postfix / selector)+
      Result r = Success;
      Result r2;
      Node<Expr> next;
      Node<Select> sel;

      while (true)
      {
        if ((r2 = optpostfix(next)) != Skip)
        {
          if (!expr)
          {
            // This is the first element in an expression.
            expr = next;
          }
          else if ((expr->kind() == Kind::Select) && !expr->as<Select>().args)
          {
            // This is the right-hand side of an infix operator.
            expr->as<Select>().args = next;
          }
          else
          {
            // Adjacency means `expr.apply(next)`
            auto sel = std::make_shared<Select>();
            sel->location = expr->location;
            sel->expr = expr;

            auto name = std::make_shared<TypeName>();
            name->location = name_apply;
            sel->typenames.push_back(name);

            sel->args = next;
            expr = sel;
          }
        }
        else if ((r2 = optselector(expr)) != Skip)
        {
          // This is an infix operator.
          if (r2 == Error)
            r = Error;
        }
        else
        {
          break;
        }
      }

      if (!expr)
        return Skip;

      return r;
    }

    Result optlet(Node<Expr>& expr)
    {
      if (!has(TokenKind::Let))
        return Skip;

      if (!has(TokenKind::Ident))
      {
        error() << loc() << "Expected an identifier" << line();
        return Error;
      }

      auto let = std::make_shared<Let>();
      let->location = previous.location;
      set_sym(let->location, let);
      expr = let;
      return Success;
    }

    Result optvar(Node<Expr>& expr)
    {
      if (!has(TokenKind::Var))
        return Skip;

      if (!has(TokenKind::Ident))
      {
        error() << loc() << "Expected an identifier" << line();
        return Error;
      }

      auto var = std::make_shared<Var>();
      var->location = previous.location;
      set_sym(var->location, var);
      expr = var;
      return Success;
    }

    Result optthrow(Node<Expr>& expr)
    {
      if (!has(TokenKind::Throw))
        return Skip;

      Result r = Success;
      auto thr = std::make_shared<Throw>();
      thr->location = previous.location;
      expr = thr;

      if ((r = optexpr(thr->expr)) == Skip)
      {
        error() << loc() << "Expected a throw expression" << line();
        r = Error;
      }

      return r;
    }

    Result optexprstart(Node<Expr>& expr)
    {
      // exprstart <- decl / throw / infix
      Result r;

      if ((r = optlet(expr)) != Skip)
        return r;

      if ((r = optvar(expr)) != Skip)
        return r;

      if ((r = optthrow(expr)) != Skip)
        return r;

      if ((r = optinfix(expr)) != Skip)
        return r;

      return Skip;
    }

    Result optexpr(Node<Expr>& expr)
    {
      // expr <- exprstart oftype? ('=' expr)?
      Result r;

      if ((r = optexprstart(expr)) == Skip)
        return Skip;

      if (peek(TokenKind::Colon))
      {
        rewind();
        auto ot = std::make_shared<Oftype>();
        ot->expr = expr;
        expr = ot;

        if (oftype(ot->type) != Success)
          r = Error;
      }

      if (has(TokenKind::Equals))
      {
        auto asgn = std::make_shared<Assign>();
        asgn->location = previous.location;
        asgn->left = expr;
        expr = asgn;

        if (optexpr(asgn->right) != Success)
        {
          error() << loc() << "Expected an expression on the right-hand side"
                  << line();
          r = Error;
        }
      }

      return r;
    }

    Result initexpr(Node<Expr>& expr)
    {
      // initexpr <- '=' expr
      if (!has(TokenKind::Equals))
        return Skip;

      Result r;

      // Encode an initexpr as a zero-argument lambda
      auto lambda = std::make_shared<Lambda>();
      auto st = push(lambda);
      lambda->location = previous.location;
      expr = lambda;

      Node<Expr> init;

      if ((r = optexpr(init)) != Skip)
      {
        lambda->body.push_back(init);
      }
      else
      {
        error() << loc() << "Expected an initialiser expression" << line();
        r = Error;
      }

      return r;
    }

    Result opttupletype(Node<Type>& type)
    {
      // tupletype <- '(' (type (',' type)*)? ')'
      if (!has(TokenKind::LParen))
        return Skip;

      auto tup = std::make_shared<TupleType>();
      tup->location = previous.location;
      type = tup;

      if (has(TokenKind::RParen))
        return Success;

      Result r = Success;

      do
      {
        Node<Type> elem;

        if (typeexpr(elem) != Success)
        {
          r = Error;
          restart_before({TokenKind::Comma, TokenKind::RParen});
        }

        tup->types.push_back(elem);
      } while (has(TokenKind::Comma));

      if (!has(TokenKind::RParen))
      {
        error() << loc() << "Expected )" << line();
        r = Error;
      }

      if (tup->types.size() == 1)
        type = tup->types.front();

      return r;
    }

    Result optmodulename(Node<TypeName>& name)
    {
      if (!has(TokenKind::EscapedString))
        return Skip;

      Result r = Success;

      name = std::make_shared<ModuleName>();
      name->location = previous.location;

      // Look for a module relative to the current source file first.
      auto base = path::to_directory(escapedstring(name->location.view()));
      auto relative = path::join(source->origin, base);
      auto std = path::join(stdlib, base);
      auto find = path::canonical(relative);

      // Otherwise, look for a module relative to the standard library.
      if (find.empty())
        find = path::canonical(std);

      if (!find.empty())
      {
        auto it = std::find(imports.begin(), imports.end(), find);
        size_t i = it - imports.begin();

        if (it == imports.end())
        {
          i = imports.size();
          imports.push_back(find);
        }

        name->location = ident("$module-" + std::to_string(i));
      }
      else
      {
        auto& out = error() << name->location << "Couldn't locate module \""
                            << base << "\"" << text(name->location);
        out << "Tried " << relative << std::endl;
        out << "Tried " << std << std::endl;
        r = Error;
      }

      if (opttypeargs(name->typeargs) == Error)
        r = Error;

      return r;
    }

    Result opttyperef(Node<Type>& type)
    {
      // typename <- ident typeargs?
      // modulename <- string typeargs?
      // typeref <- (modulename / typename) ('::' typename)*
      if (
        !peek(TokenKind::Ident) && !peek(TokenKind::EscapedString) &&
        !peek(TokenKind::UnescapedString))
        return Skip;

      rewind();
      auto typeref = std::make_shared<TypeRef>();
      type = typeref;

      Result r = Success;

      // A typeref can start with a module name.
      Node<TypeName> name;

      if (optmodulename(name) != Skip)
      {
        typeref->location = name->location;
        typeref->typenames.push_back(name);

        if (!has(TokenKind::DoubleColon))
          return r;
      }

      do
      {
        if (!has(TokenKind::Ident))
        {
          error() << loc() << "Expected a type identifier" << line();
          return Error;
        }

        auto name = std::make_shared<TypeName>();
        name->location = previous.location;

        typeref->location = name->location;
        typeref->typenames.push_back(name);

        if (opttypeargs(name->typeargs) == Error)
          r = Error;
      } while (has(TokenKind::DoubleColon));

      return r;
    }

    Result opttypelist(Node<Type>& type)
    {
      bool ok = peek(TokenKind::Ident) && peek(TokenKind::Ellipsis);
      rewind();

      if (!ok)
        return Skip;

      auto tl = std::make_shared<TypeList>();
      type = tl;

      if (!has(TokenKind::Ident))
        return Error;

      tl->location = previous.location;

      if (!has(TokenKind::Ellipsis))
        return Error;

      return Success;
    }

    Result optcaptype(Node<Type>& type)
    {
      // captype <-
      //  'iso' / 'mut' / 'imm' / 'Self' / tupletype / typelist / typeref
      if (has(TokenKind::Iso))
      {
        auto cap = std::make_shared<Iso>();
        cap->location = previous.location;
        type = cap;
        return Success;
      }

      if (has(TokenKind::Mut))
      {
        auto cap = std::make_shared<Mut>();
        cap->location = previous.location;
        type = cap;
        return Success;
      }

      if (has(TokenKind::Imm))
      {
        auto cap = std::make_shared<Imm>();
        cap->location = previous.location;
        type = cap;
        return Success;
      }

      if (has(TokenKind::Self))
      {
        auto self = std::make_shared<Self>();
        self->location = previous.location;
        type = self;
        return Success;
      }

      Result r;

      if ((r = opttupletype(type)) != Skip)
        return r;

      if ((r = opttypelist(type)) != Skip)
        return r;

      if ((r = opttyperef(type)) != Skip)
        return r;

      return Skip;
    }

    Result optviewtype(Node<Type>& type)
    {
      // viewtype <- captype (('~>' / '<~') captype)*
      Result r;

      if ((r = optcaptype(type)) == Skip)
        return r;

      Node<TypePair> pair;

      while (true)
      {
        if (has(TokenKind::Symbol, "~>"))
          pair = std::make_shared<ViewType>();
        else if (has(TokenKind::Symbol, "<~"))
          pair = std::make_shared<ExtractType>();
        else
          break;

        pair->location = previous.location;
        pair->left = type;
        type = pair;

        Result r2;

        if ((r2 = optcaptype(pair->right)) != Success)
        {
          error() << loc() << "Expected a type" << line();
          r = Error;
          break;
        }
      }

      return r;
    }

    Result optfunctiontype(Node<Type>& type)
    {
      // functiontype <- viewtype ('->' functiontype)?
      // Right associative.
      Result r;

      if ((r = optviewtype(type)) != Success)
        return r;

      if (!has(TokenKind::Symbol, "->"))
        return Success;

      auto functype = std::make_shared<FunctionType>();
      functype->location = previous.location;
      functype->left = type;
      type = functype;

      return optfunctiontype(functype->right);
    }

    Result optisecttype(Node<Type>& type)
    {
      // isecttype <- functiontype ('&' functiontype)*
      Result r = Success;

      if ((r = optfunctiontype(type)) != Success)
        return r;

      while (has(TokenKind::Symbol, "&"))
      {
        Node<Type> next;
        Result r2;

        if ((r2 = optfunctiontype(next)) != Success)
        {
          error() << loc() << "Expected a type" << line();
          r = Error;
        }

        if (r2 != Skip)
          type = dnf::conjunction(type, next);
      }

      return r;
    }

    Result optthrowtype(Node<Type>& type)
    {
      // throwtype <- 'throw'? isecttype
      bool throwing = has(TokenKind::Throw);
      Result r;

      if ((r = optisecttype(type)) == Skip)
        return Skip;

      if (throwing)
        type = dnf::throwtype(type);

      return r;
    }

    Result optuniontype(Node<Type>& type)
    {
      // uniontype <- throwtype ('|' throwtype)*
      Result r = Success;

      if ((r = optthrowtype(type)) != Success)
        return r;

      while (has(TokenKind::Symbol, "|"))
      {
        Node<Type> next;
        Result r2;

        if ((r2 = optthrowtype(next)) != Success)
        {
          error() << loc() << "Expected a type" << line();
          r = Error;
        }

        if (r2 != Skip)
          type = dnf::disjunction(type, next);
      }

      return r;
    }

    Result typeexpr(Node<Type>& type)
    {
      // typeexpr <- uniontype
      if (optuniontype(type) != Success)
      {
        error() << loc() << "Expected a type" << line();
        return Error;
      }

      return Success;
    }

    Result inittype(Node<Type>& type)
    {
      // inittype <- '=' type
      if (!has(TokenKind::Equals))
        return Skip;

      if (typeexpr(type) != Success)
        return Error;

      return Success;
    }

    Result oftype(Node<Type>& type)
    {
      if (!has(TokenKind::Colon))
        return Skip;

      return typeexpr(type);
    }

    Result optparam(Node<Expr>& param)
    {
      if (peek(TokenKind::Ident))
      {
        bool isparam = peek(TokenKind::Colon) || peek(TokenKind::Equals) ||
          peek(TokenKind::Comma) || peek(TokenKind::FatArrow) ||
          peek(TokenKind::RParen);
        rewind();

        if (isparam)
        {
          Result r = Success;
          has(TokenKind::Ident);
          auto p = std::make_shared<Param>();
          p->location = previous.location;

          if (oftype(p->type) == Error)
            r = Error;

          if (initexpr(p->init) == Error)
            r = Error;

          set_sym(p->location, p);
          param = p;
          return r;
        }
      }

      return optexpr(param);
    }

    Result optparamlist(List<Expr>& params, TokenKind terminator)
    {
      Result r = Success;
      Result r2;
      Node<Expr> param;

      do
      {
        if ((r2 = optparam(param)) == Skip)
          break;

        params.push_back(param);

        if (r2 == Error)
        {
          error() << loc() << "Expected a parameter" << line();
          r = Error;
          restart_before({TokenKind::Comma, terminator});
        }
      } while (has(TokenKind::Comma));

      return r;
    }

    Result optparams(List<Expr>& params)
    {
      if (!has(TokenKind::LParen))
        return Skip;

      Result r = optparamlist(params, TokenKind::RParen);

      if (!has(TokenKind::RParen))
      {
        error() << loc() << "Expected )" << line();
        r = Error;
      }

      return r;
    }

    Result optfield(List<Member>& members)
    {
      // field <- ident oftype initexpr ';'
      if (!has(TokenKind::Ident))
        return Skip;

      auto field = std::make_shared<Field>();
      field->location = previous.location;
      Result r = Success;

      if (oftype(field->type) == Error)
        r = Error;

      if (initexpr(field->init) == Error)
        r = Error;

      if (!has(TokenKind::Semicolon))
      {
        error() << loc() << "Expected ;" << line();
        r = Error;
      }

      members.push_back(field);
      set_sym(field->location, field);
      return r;
    }

    Result optfunction(List<Member>& members)
    {
      // function <- (ident / symbol)? typeparams? params oftype? (block / ';')
      bool ok = peek(TokenKind::Symbol) ||
        (peek(TokenKind::Ident) &&
         (peek(TokenKind::LSquare) || peek(TokenKind::LParen))) ||
        (peek(TokenKind::LSquare) || peek(TokenKind::LParen));

      rewind();

      if (!ok)
        return Skip;

      auto func = std::make_shared<Function>();
      auto st = push(func);
      Result r = Success;

      if (has(TokenKind::Ident) || has(TokenKind::Symbol))
      {
        func->location = previous.location;
        func->name = previous.location;
      }
      else
      {
        // Replace an empy name with 'apply'.
        func->location = lookahead.front().location;
        func->name = name_apply;
      }

      if (opttypeparams(func->typeparams) == Error)
        r = Error;

      if (optparams(func->params) != Success)
        r = Error;

      for (auto& param : func->params)
      {
        if (param->kind() != Kind::Param)
        {
          error() << param->location << "Function parameters can't be patterns"
                  << text(param->location);
        }
        else
        {
          auto& p = param->as<Param>();

          if (!p.type)
          {
            error() << param->location << "Function parameters must have types"
                    << text(param->location);
          }
        }
      }

      if (oftype(func->result) == Error)
        r = Error;

      Result r2;

      if ((r2 = optlambda(func->body)) != Skip)
      {
        if (r2 == Error)
          r = Error;
      }
      else if (!has(TokenKind::Semicolon))
      {
        error() << loc() << "Expected a lambda or ;" << line();
        r = Error;
      }

      members.push_back(func);
      set_sym_parent(func->name, func);
      return r;
    }

    Result opttypeparam(Node<TypeParam>& tp)
    {
      // typeparam <- ident oftype inittype
      if (!has(TokenKind::Ident))
        return Skip;

      Result r = Success;
      auto loc = previous.location;

      if (has(TokenKind::Ellipsis))
        tp = std::make_shared<TypeParamList>();
      else
        tp = std::make_shared<TypeParam>();

      tp->location = loc;

      if (oftype(tp->type) == Error)
        r = Error;

      if (inittype(tp->init) == Error)
        r = Error;

      set_sym(tp->location, tp);
      return r;
    }

    Result opttypeparams(List<TypeParam>& typeparams)
    {
      // typeparams <- ('[' typeparam (',' typeparam)* ']')?
      if (!has(TokenKind::LSquare))
        return Skip;

      Result r = Success;

      do
      {
        Node<TypeParam> tp;
        Result r2;

        if ((r2 = opttypeparam(tp)) != Success)
        {
          error() << loc() << "Expected a type parameter" << line();
          r = Error;
          restart_before({TokenKind::Comma, TokenKind::RSquare});
        }

        if (r2 != Skip)
          typeparams.push_back(tp);
      } while (has(TokenKind::Comma));

      if (!has(TokenKind::RSquare))
      {
        error() << loc() << "Expected , or ]" << line();
        r = Error;
      }

      return r;
    }

    Result checkinherit(Node<Type>& inherit)
    {
      if (!inherit)
        return Skip;

      Result r = Success;

      if (inherit->kind() == Kind::IsectType)
      {
        auto& isect = inherit->as<IsectType>();

        for (auto& type : isect.types)
        {
          if (checkinherit(type) == Error)
            r = Error;
        }
      }
      else if (inherit->kind() != Kind::TypeRef)
      {
        error() << inherit->location << "A type can't inherit from a "
                << kindname(inherit->kind()) << text(inherit->location);
        r = Error;
      }

      return r;
    }

    Result entity(Entity& ent)
    {
      // entity <- typeparams oftype
      Result r = Success;

      if (opttypeparams(ent.typeparams) == Error)
        r = Error;

      if (oftype(ent.inherits) == Error)
        r = Error;

      if (checkinherit(ent.inherits) == Error)
        r = Error;

      return r;
    }

    Result namedentity(NamedEntity& ent)
    {
      // namedentity <- ident entity
      Result r = Success;

      if (optident(ent.id) == Skip)
      {
        error() << loc() << "Expected an identifier" << line();
        r = Error;
      }

      if (entity(ent) == Error)
        r = Error;

      return r;
    }

    Result optusing(List<Member>& members)
    {
      // using <- 'using' typeref ';'
      if (!has(TokenKind::Using))
        return Skip;

      auto use = std::make_shared<Using>();
      use->location = previous.location;

      Result r;

      if ((r = opttyperef(use->type)) != Success)
      {
        if (r == Skip)
          error() << loc() << "Expected a type reference" << line();

        r = Error;
      }

      if (!has(TokenKind::Semicolon))
      {
        error() << loc() << "Expected ;" << line();
        r = Error;
      }

      members.push_back(use);
      symbol_stack.back()->symbol_table()->use.push_back(use);
      return r;
    }

    Result typealias(List<Member>& members)
    {
      // typealias <- 'type' ident typeparams? '=' type ';'
      if (!has(TokenKind::Type))
        return Skip;

      auto alias = std::make_shared<TypeAlias>();
      alias->location = previous.location;
      Result r = Success;

      if (has(TokenKind::Ident))
      {
        alias->id = previous.location;
      }
      else
      {
        error() << loc() << "Expected an identifier" << line();
        r = Error;
      }

      if (opttypeparams(alias->typeparams) == Error)
        r = Error;

      if (!has(TokenKind::Equals))
      {
        error() << loc() << "Expected =" << line();
        r = Error;
      }

      if (typeexpr(alias->type) == Error)
        r = Error;

      if (!has(TokenKind::Semicolon))
      {
        error() << loc() << "Expected ;" << line();
        r = Error;
      }

      members.push_back(alias);
      set_sym(alias->id, alias);
      return r;
    }

    Result interface(List<Member>& members)
    {
      // interface <- 'interface' namedentity typebody
      if (!has(TokenKind::Interface))
        return Skip;

      auto iface = std::make_shared<Interface>();
      auto st = push(iface);
      iface->location = previous.location;
      Result r = Success;

      if (namedentity(*iface) == Error)
        r = Error;

      if (typebody(iface->members) == Error)
        r = Error;

      members.push_back(iface);
      set_sym_parent(iface->id, iface);
      return r;
    }

    Result classdef(List<Member>& members)
    {
      // classdef <- 'class' namedentity typebody
      if (!has(TokenKind::Class))
        return Skip;

      auto cls = std::make_shared<Class>();
      auto st = push(cls);
      cls->location = previous.location;
      Result r = Success;

      if (namedentity(*cls) == Error)
        r = Error;

      if (typebody(cls->members) == Error)
        r = Error;

      members.push_back(cls);
      set_sym_parent(cls->id, cls);
      return r;
    }

    Result optmoduledef(Node<Module>& module)
    {
      // moduledef <- 'module' entity ';'
      if (!has(TokenKind::Module))
        return Skip;

      auto mod = std::make_shared<Module>();
      mod->location = previous.location;
      Result r = Success;

      if (entity(*mod) == Error)
        r = Error;

      if (!has(TokenKind::Semicolon))
      {
        error() << loc() << "Expected ;" << line();
        r = Error;
      }

      if (!module)
      {
        module = mod;
      }
      else
      {
        error() << mod->location << "The module has already been defined"
                << text(mod->location) << module->location
                << "The previous definition is here" << text(module->location);
        r = Error;
      }

      return r;
    }

    Result optmember(List<Member>& members)
    {
      // member <-
      //  classdef / interface / typealias / using / field / function
      Result r;

      if ((r = classdef(members)) != Skip)
        return r;

      if ((r = interface(members)) != Skip)
        return r;

      if ((r = typealias(members)) != Skip)
        return r;

      if ((r = optusing(members)) != Skip)
        return r;

      if ((r = optfunction(members)) != Skip)
        return r;

      if ((r = optfield(members)) != Skip)
        return r;

      return Skip;
    }

    Result typebody(List<Member>& members)
    {
      // typebody <- '{' member* '}'
      Result r = Success;

      if (!has(TokenKind::LBrace))
      {
        error() << loc() << "Expected {" << line();
        r = Error;
      }

      if (has(TokenKind::RBrace))
        return r;

      while (!has(TokenKind::RBrace))
      {
        if (has(TokenKind::End))
        {
          error() << loc() << "Expected }" << line();
          return Error;
        }

        Result r2;

        if ((r2 = optmember(members)) == Skip)
        {
          error() << loc()
                  << "Expected a class, interface, type alias, field, "
                     "or function"
                  << line();

          restart_before({TokenKind::RBrace,
                          TokenKind::Class,
                          TokenKind::Interface,
                          TokenKind::Type,
                          TokenKind::Ident,
                          TokenKind::Symbol,
                          TokenKind::LSquare,
                          TokenKind::LParen});
        }

        if (r2 == Error)
          r = Error;
      }

      return r;
    }

    Result sourcefile(
      const std::string& file, Node<Class>& module, Node<Module>& moduledef)
    {
      auto source = load_source(file);

      if (!source)
      {
        error() << "Couldn't read file " << file << std::endl;
        return Error;
      }

      start(source);

      // module <- (moduledef / member)*
      while (!has(TokenKind::End))
      {
        Result r;

        if ((r = optmoduledef(moduledef)) == Skip)
          r = optmember(module->members);

        if (r == Skip)
        {
          error() << loc()
                  << "Expected a module, class, interface, type alias, field, "
                     "or function"
                  << line();

          restart_before({TokenKind::Module,
                          TokenKind::Class,
                          TokenKind::Interface,
                          TokenKind::Type,
                          TokenKind::Ident,
                          TokenKind::Symbol,
                          TokenKind::LSquare,
                          TokenKind::LParen});
        }
      }

      return final_result;
    }

    // `path` can't be a reference, because `imports` may be modified during
    // parsing.
    Result
    module(const std::string path, size_t module_index, Node<Class>& program)
    {
      auto modulename = ident("$module-" + std::to_string(module_index));

      if (look_in(symbol_stack.front(), modulename))
        return final_result;

      Node<Module> moduledef;
      auto r = Success;

      auto module = std::make_shared<Class>();
      auto st = push(module);
      module->id = modulename;
      program->members.push_back(module);
      set_sym_parent(module->id, module);

      if (!path::is_directory(path))
      {
        // This is only for testing.
        r = sourcefile(path, module, moduledef);
      }
      else
      {
        auto files = path::files(path);
        size_t count = 0;

        for (auto& file : files)
        {
          if (ext != path::extension(file))
            continue;

          auto filename = path::join(path, file);
          count++;

          if (sourcefile(filename, module, moduledef) == Error)
            r = Error;
        }

        if (!count)
        {
          error() << "No " << ext << " files found in " << path << std::endl;
          r = Error;
        }
      }

      if (moduledef)
      {
        module->typeparams = std::move(moduledef->typeparams);
        module->inherits = moduledef->inherits;
      }

      return r;
    }
  };

  std::pair<bool, Ast>
  parse(const std::string& path, const std::string& stdlib, std::ostream& out)
  {
    Parse parse(stdlib, out);
    auto program = std::make_shared<Class>();
    auto st = parse.push(program);
    parse.imports.push_back(path::canonical(path));

    for (size_t i = 0; i < parse.imports.size(); i++)
      parse.module(parse.imports[i], i, program);

    return {parse.final_result == Success, program};
  }
}
