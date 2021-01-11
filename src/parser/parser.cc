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
    Location name_has_value;
    Location name_next;

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
      name_has_value = ident("has_value");
      name_next = ident("next");
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

    void set_sym(const Location& id, Ast node, SymbolTable& st)
    {
      auto find = st.map.find(id);

      if (find != st.map.end())
      {
        auto& loc = node->location;
        auto& prev = find->second->location;

        error() << loc << "There is a previous definition of \"" << id.view()
                << "\"" << text(loc) << prev
                << "The previous definition is here" << text(prev);
        return;
      }

      st.map.emplace(id, node);
    }

    void set_sym(const Location& id, Ast node)
    {
      assert(symbol_stack.size() > 0);
      auto st = symbol_stack.back()->symbol_table();
      set_sym(id, node, *st);
    }

    void set_sym_parent(const Location& id, Ast node)
    {
      assert(symbol_stack.size() > 1);
      auto st = symbol_stack[symbol_stack.size() - 2]->symbol_table();
      set_sym(id, node, *st);
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

    bool is_blockexpr(Node<Expr>& expr)
    {
      switch (expr->kind())
      {
        case Kind::Block:
        case Kind::When:
        case Kind::While:
        case Kind::Match:
        case Kind::If:
        case Kind::Lambda:
          return true;

        case Kind::Prefix:
          return expr->as<Prefix>().block;

        case Kind::Infix:
          return expr->as<Infix>().block;

        default:
          return false;
      }
    }

    void peek_matching(TokenKind kind)
    {
      // Skip over balanced () [] {}
      while (!peek(TokenKind::End))
      {
        if (peek(kind))
          return;

        if (peek(TokenKind::LParen))
        {
          peek_matching(TokenKind::RParen);
        }
        else if (peek(TokenKind::LSquare))
        {
          peek_matching(TokenKind::RSquare);
        }
        else if (peek(TokenKind::LBrace))
        {
          peek_matching(TokenKind::RBrace);
        }
        else
        {
          next();
        }
      }
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
      // when <- 'when' tuple block
      if (!has(TokenKind::When))
        return Skip;

      Result r = Success;
      auto when = std::make_shared<When>();
      auto st = push(when);
      when->location = previous.location;
      expr = when;

      if (opttuple(when->waitfor) != Success)
      {
        error() << loc() << "Expected a when condition" << line();
        r = Error;
      }

      if (optblock(when->behaviour) != Success)
      {
        error() << loc() << "Expected a when body" << line();
        r = Error;
      }

      return r;
    }

    Result forcondition(Node<Expr>& state, Node<Expr>& iter)
    {
      Result r = Success;

      if (!has(TokenKind::LParen))
      {
        error() << loc() << "Expected ( to start a for-loop condition"
                << line();
        r = Error;
      }

      if (optexpr(state) != Success)
      {
        error() << loc() << "Expected for-loop state" << line();
        r = Error;
      }

      if (!has(TokenKind::In))
      {
        error() << loc() << "Expected 'in'" << line();
        r = Error;
      }

      if (optexpr(iter) != Success)
      {
        error() << loc() << "Expected for-loop iterator" << line();
        r = Error;
      }

      if (!has(TokenKind::RParen))
      {
        error() << loc() << "Expected ) to end a for-loop condition" << line();
        r = Error;
      }

      return r;
    }

    Result optfor(Node<Expr>& expr)
    {
      // for <- 'for' '(' expr 'in' expr ')' block
      if (!has(TokenKind::For))
        return Skip;

      Result r = Success;

      auto blk = std::make_shared<Block>();
      blk->location = previous.location;
      auto st0 = push(blk);
      expr = blk;

      auto wh = std::make_shared<While>();
      auto st1 = push(wh);
      wh->location = previous.location;

      Node<Expr> state;
      Node<Expr> iter;

      if (forcondition(state, iter) != Success)
        r = Error;

      if (optblock(wh->body) != Success)
        r = Error;

      if (r != Success)
        return r;

      // Rewrite as a while loop.

      // init = (assign (let (ref $0)) $iter)
      auto init = std::make_shared<Assign>();
      init->location = iter->location;
      auto id = ident();

      auto decl = std::make_shared<Let>();
      decl->location = id;
      set_sym_parent(id, decl);
      init->left = decl;
      init->right = iter;

      // cond = (tuple (apply (select (ref $0) (ident has_value)) (tuple)))
      auto cond = std::make_shared<Tuple>();
      auto select_has_value = std::make_shared<Select>();
      select_has_value->expr = ref(id);
      select_has_value->member = name_has_value;
      auto apply_has_value = std::make_shared<Apply>();
      apply_has_value->expr = select_has_value;
      apply_has_value->args = std::make_shared<Tuple>();
      cond->seq.push_back(apply_has_value);

      // begin = (assign $state (apply (ref $0) (tuple))
      auto apply = std::make_shared<Apply>();
      apply->expr = ref(id);
      apply->args = std::make_shared<Tuple>();

      auto begin = std::make_shared<Assign>();
      begin->location = wh->body->location;
      begin->left = state;
      begin->right = apply;

      // end = (apply (select (ref $0) (ident next)) (tuple))
      auto select_next = std::make_shared<Select>();
      select_next->location = wh->body->location;
      select_next->expr = ref(id);
      select_next->member = name_next;

      auto end = std::make_shared<Apply>();
      end->location = wh->body->location;
      end->expr = select_next;
      end->args = std::make_shared<Tuple>();

      // (block init wh)
      blk->seq.push_back(init);
      blk->seq.push_back(wh);

      wh->cond = cond;
      auto& body = wh->body->as<Block>();
      body.seq.insert(body.seq.begin(), begin);
      body.seq.push_back(end);

      return Success;
    }

    Result optwhile(Node<Expr>& expr)
    {
      // while <- 'while' tuple block
      if (!has(TokenKind::While))
        return Skip;

      auto wh = std::make_shared<While>();
      auto st = push(wh);
      wh->location = previous.location;
      expr = wh;

      Result r = Success;

      if (opttuple(wh->cond) != Success)
      {
        error() << loc() << "Expected while-loop condition" << line();
        r = Error;
      }

      if (optblock(wh->body) != Success)
      {
        error() << loc() << "Expected while-loop body" << line();
        r = Error;
      }

      return r;
    }

    Result matchcase(Node<Case>& expr)
    {
      // case <- expr ('if' expr)? '=>' expr
      Result r = Success;
      expr = std::make_shared<Case>();
      auto st = push(expr);

      if (optexpr(expr->pattern) != Success)
      {
        error() << loc() << "Expected a case pattern" << line();
        r = Error;
      }

      if (has(TokenKind::If))
      {
        if (optexpr(expr->guard) != Success)
        {
          error() << loc() << "Expected a guard expression" << line();
          r = Error;
        }
      }

      if (has(TokenKind::FatArrow))
      {
        expr->location = previous.location;
      }
      else
      {
        error() << loc() << "Expected =>" << line();
        r = Error;
      }

      if (optexpr(expr->body) != Success)
      {
        error() << loc() << "Expected a case expression" << line();
        r = Error;
      }

      return r;
    }

    Result optmatch(Node<Expr>& expr)
    {
      // match <- 'match' tuple '{' case* '}'
      if (!has(TokenKind::Match))
        return Skip;

      Result r = Success;
      auto match = std::make_shared<Match>();
      auto st = push(match);
      match->location = previous.location;
      expr = match;

      if (opttuple(match->cond) != Success)
      {
        error() << loc() << "Expected a match condition" << line();
        r = Error;
      }

      if (!has(TokenKind::LBrace))
      {
        error() << loc() << "Expected { to start match cases" << line();
        return Error;
      }

      while (true)
      {
        Node<Case> ca;
        Result r2 = matchcase(ca);

        if (r2 == Skip)
          break;

        match->cases.push_back(ca);

        if (r2 == Error)
          r = Error;
      }

      if (!has(TokenKind::RBrace))
      {
        error() << loc() << "Expected a case or } to end match cases" << line();
        r = Error;
      }

      return r;
    }

    Result optif(Node<Expr>& expr)
    {
      // if <- 'if' tuple block ('else' block)?
      if (!has(TokenKind::If))
        return Skip;

      Result r = Success;
      auto cond = std::make_shared<If>();
      auto st = push(cond);
      cond->location = previous.location;
      expr = cond;

      if (opttuple(cond->cond) != Success)
      {
        error() << loc() << "Expected an if condition" << line();
        r = Error;
      }

      if (optblock(cond->on_true) != Success)
      {
        error() << loc() << "Expected a true branch block" << line();
        r = Error;
      }

      if (!has(TokenKind::Else))
        return r;

      if (optblock(cond->on_false) != Success)
      {
        error() << loc() << "Expected a false branch block" << line();
        r = Error;
      }

      return r;
    }

    Result opttuple(Node<Expr>& expr)
    {
      // tuple <- '(' expr* ')'
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

        if (optexpr(elem) != Success)
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

      if (oftype(tup->type) == Error)
        r = Error;

      // TODO: collapse if we're not looking for a lambda signature
      // if (!tup->type && (tup->seq.size() == 1))
      //   expr = tup->seq.front();

      return r;
    }

    Result optblock(Node<Expr>& expr)
    {
      // block <- '{' ('}' / (preblock / expr ';')* controlflow? '}')
      if (!has(TokenKind::LBrace))
        return Skip;

      auto blk = std::make_shared<Block>();
      auto st = push(blk);
      blk->location = previous.location;
      expr = blk;

      if (has(TokenKind::RBrace))
        return Success;

      Result r = Success;
      Node<Expr> elem;
      bool check_controlflow = false;

      do
      {
        Result r2 = optexpr(elem);

        if (r2 == Skip)
        {
          check_controlflow = true;
          break;
        }

        blk->seq.push_back(elem);

        if (r2 == Error)
          r = Error;
      } while (is_blockexpr(elem) || has(TokenKind::Semicolon));

      if (check_controlflow)
      {
        // Swallow a trailing ; if there is one.
        Result r2 = optcontrolflow(elem);
        has(TokenKind::Semicolon);

        if (r2 != Skip)
          blk->seq.push_back(elem);

        if (r2 == Error)
          r = Error;
      }

      if (!has(TokenKind::RBrace))
      {
        error() << loc() << "Expected an expression or }" << line();
        r = Error;
      }

      return r;
    }

    void reftoparam(Node<Expr>& e, Node<Param>& param)
    {
      auto& ref = e->as<Ref>();
      param->location = ref.location;
      param->type = ref.type;
      set_sym(param->location, param);
    }

    Result optlambda(Node<Expr>& expr)
    {
      // lambda <- (signature / ident) '=>' expr
      // This can also return a Tuple node.
      bool ok = peek(TokenKind::LSquare) || peek(TokenKind::LParen) ||
        (peek(TokenKind::Ident) && peek(TokenKind::FatArrow));
      rewind();

      if (!ok)
        return Skip;

      Result r = Success;
      Node<Lambda> lambda;

      if (peek(TokenKind::LSquare))
      {
        rewind();
        lambda = std::make_shared<Lambda>();
        auto st = push(lambda);

        if (signature(lambda->signature) != Success)
          r = Error;
      }
      else if ((r = opttuple(expr)) != Skip)
      {
        // Return a tuple instead of a lambda.
        if (!peek(TokenKind::Throws) && !peek(TokenKind::FatArrow))
          return r;

        rewind();
        lambda = std::make_shared<Lambda>();
        auto st = push(lambda);

        auto tup = expr->as<Tuple>();
        auto sig = std::make_shared<Signature>();
        sig->location = tup.location;
        sig->result = tup.type;
        lambda->signature = sig;

        // Each element must be: ref / (assign ref expr)
        for (auto& e : tup.seq)
        {
          auto param = std::make_shared<Param>();
          sig->params.push_back(param);

          if (e->kind() == Kind::Ref)
          {
            reftoparam(e, param);
            continue;
          }

          if (e->kind() == Kind::Assign)
          {
            auto& asgn = e->as<Assign>();

            if (asgn.left->kind() == Kind::Ref)
            {
              reftoparam(asgn.left, param);
              param->init = asgn.right;
              continue;
            }
          }

          error() << loc() << "Expected a lambda parameter" << line();
          r = Error;
        }

        if (optthrows(sig->throws) == Error)
          r = Error;
      }
      else if (peek(TokenKind::Ident) && peek(TokenKind::FatArrow))
      {
        rewind();
        has(TokenKind::Ident);

        lambda = std::make_shared<Lambda>();
        auto st = push(lambda);

        auto param = std::make_shared<Param>();
        param->location = previous.location;
        set_sym(param->location, param);

        auto sig = std::make_shared<Signature>();
        sig->location = previous.location;
        sig->params.push_back(param);

        lambda->signature = sig;
        r = Success;
      }

      auto st = push(lambda);
      expr = lambda;

      if (!has(TokenKind::FatArrow))
      {
        error() << loc() << "Expected =>" << line();
        r = Error;
      }

      lambda->location = previous.location;

      if (optexpr(lambda->body) != Success)
      {
        error() << loc() << "Expected a lambda body" << line();
        r = Error;
      }

      return r;
    }

    Result optblockexpr(Node<Expr>& expr)
    {
      // blockexpr <- block / when / if / match / while / for / lambda
      // This can also return a Tuple node, because optlambda may do so.
      Result r;

      if ((r = optblock(expr)) != Skip)
        return r;

      if ((r = optwhen(expr)) != Skip)
        return r;

      if ((r = optif(expr)) != Skip)
        return r;

      if ((r = optmatch(expr)) != Skip)
        return r;

      if ((r = optwhile(expr)) != Skip)
        return r;

      if ((r = optfor(expr)) != Skip)
        return r;

      if ((r = optlambda(expr)) != Skip)
        return r;

      return Skip;
    }

    Result optbreak(Node<Expr>& expr)
    {
      // break <- 'break'
      if (!has(TokenKind::Break))
        return Skip;

      auto brk = std::make_shared<Break>();
      brk->location = previous.location;
      expr = brk;
      return Success;
    }

    Result optcontinue(Node<Expr>& expr)
    {
      // continue <- 'continue'
      if (!has(TokenKind::Continue))
        return Skip;

      auto cont = std::make_shared<Continue>();
      cont->location = previous.location;
      expr = cont;
      return Success;
    }

    Result optreturn(Node<Expr>& expr)
    {
      // return <- 'return' expr?
      if (!has(TokenKind::Return))
        return Skip;

      auto ret = std::make_shared<Return>();
      ret->location = previous.location;
      expr = ret;

      if (optexpr(ret->expr) == Error)
        return Error;

      return Success;
    }

    Result optyield(Node<Expr>& expr)
    {
      // yield <- 'yield' expr
      if (!has(TokenKind::Yield))
        return Skip;

      auto yield = std::make_shared<Yield>();
      yield->location = previous.location;
      expr = yield;

      if (optexpr(yield->expr) == Error)
        return Error;

      return Success;
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

      if (oftype(ref->type) == Error)
        return Error;

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
      else if (has(TokenKind::True))
      {
        auto con = std::make_shared<True>();
        con->location = previous.location;
        expr = con;
      }
      else if (has(TokenKind::False))
      {
        auto con = std::make_shared<False>();
        con->location = previous.location;
        expr = con;
      }
      else
      {
        return Skip;
      }

      return Success;
    }

    template<typename Decl>
    Result declelem(Node<Expr>& decl)
    {
      // declelem <- (ident / '(' declelem (',' declelem)* ')') oftype?
      if (has(TokenKind::Ident))
      {
        auto elem = std::make_shared<Decl>();
        elem->location = previous.location;
        set_sym(elem->location, elem);
        decl = elem;

        if (oftype(elem->type) == Error)
          return Error;

        return Success;
      }

      if (!has(TokenKind::LParen))
      {
        error() << loc() << "Expected an identifier or (" << line();
        return Error;
      }

      auto tup = std::make_shared<Tuple>();
      tup->location = previous.location;
      decl = tup;
      Result r = Success;

      do
      {
        Node<Expr> elem;

        if (declelem<Decl>(elem) == Error)
        {
          restart_before({TokenKind::Comma, TokenKind::RParen});
          r = Error;
        }
      } while (has(TokenKind::Comma));

      if (!has(TokenKind::RParen))
      {
        error() << loc() << "Expected a )" << line();
        r = Error;
      }

      if (oftype(tup->type) == Error)
        r = Error;

      return r;
    }

    Result optdecl(Node<Expr>& expr)
    {
      // decl <- ('let' / 'var') declelem
      if (has(TokenKind::Let))
        return declelem<Let>(expr);

      if (has(TokenKind::Var))
        return declelem<Var>(expr);

      return Skip;
    }

    Result optnew(Node<Expr>& expr)
    {
      // new <- 'new' (tuple / typebody / type typebody) ('in' ident)?
      if (!has(TokenKind::New))
        return Skip;

      Result r = Success;

      if (peek(TokenKind::LParen))
      {
        rewind();
        auto n = std::make_shared<New>();
        n->location = previous.location;
        expr = n;

        if (opttuple(n->args) != Success)
          r = Error;

        if (has(TokenKind::In))
        {
          if (optident(n->in) != Success)
          {
            error() << loc() << "Expected an identifier" << line();
            r = Error;
          }
        }

        return r;
      }

      auto obj = std::make_shared<ObjectLiteral>();
      auto st = push(obj);
      obj->location = previous.location;
      expr = obj;

      if (!peek(TokenKind::LBrace))
      {
        if (typeexpr(obj->inherits) == Error)
          r = Error;

        if (checkinherit(obj->inherits) == Error)
          r = Error;
      }
      else
      {
        rewind();
      }

      if (typebody(obj->members) != Success)
        r = Error;

      if (has(TokenKind::In))
      {
        if (optident(obj->in) != Success)
        {
          error() << loc() << "Expected an identifier" << line();
          r = Error;
        }
      }

      return r;
    }

    Result optstaticref(Node<Expr>& expr)
    {
      // staticname <- (ident / symbol) typeargs?
      // staticref <- [nonlocal] staticname ('::' staticname)*
      if (!peek(TokenKind::Ident) && !peek(TokenKind::Symbol))
        return Skip;

      bool local = is_localref(lookahead[la - 1].location);
      rewind();

      if (local)
        return Skip;

      auto stat = std::make_shared<StaticRef>();
      expr = stat;

      Result r = Success;

      do
      {
        if (!has(TokenKind::Ident) && !has(TokenKind::Symbol))
        {
          error() << loc() << "Expected an identifier or symbol" << line();
          r = Error;
          break;
        }

        // Use the location of the last ident or symbol.
        stat->location = previous.location;

        auto name = std::make_shared<TypeName>();
        name->location = previous.location;
        stat->typenames.push_back(name);

        if (opttypeargs(name->typeargs) == Error)
          r = Error;
      } while (has(TokenKind::DoubleColon));

      return r;
    }

    Result optatom(Node<Expr>& expr)
    {
      // atom <- staticref / ref / constant / new / tuple
      Result r;

      if ((r = optstaticref(expr)) != Skip)
        return r;

      if ((r = optref(expr)) != Skip)
        return r;

      if ((r = optconstant(expr)) != Skip)
        return r;

      if ((r = optdecl(expr)) != Skip)
        return r;

      if ((r = optnew(expr)) != Skip)
        return r;

      // We will have already found a tuple when trying to find a lambda, so
      // this will always Skip.
      if ((r = opttuple(expr)) != Skip)
        return r;

      return Skip;
    }

    Result optselect(Node<Expr>& expr)
    {
      // select <- expr '.' (ident / symbol)
      if (!has(TokenKind::Dot))
        return Skip;

      auto sel = std::make_shared<Select>();
      sel->location = previous.location;
      sel->expr = expr;
      expr = sel;

      if (has(TokenKind::Ident) || has(TokenKind::Symbol))
      {
        sel->member = previous.location;
        return Success;
      }

      error() << loc() << "Expected an identifier or a symbol" << line();
      return Error;
    }

    Result opttypeargs(List<Type>& typeargs)
    {
      // typeargs <- '[' (expr (',' expr)*)?) ']'
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

    Result optspecialise(Node<Expr>& expr)
    {
      // specialise <- expr typeargs
      if (!peek(TokenKind::LSquare))
        return Skip;

      rewind();
      auto spec = std::make_shared<Specialise>();
      spec->location = previous.location;
      spec->expr = expr;
      expr = spec;

      if (opttypeargs(spec->typeargs) != Success)
        return Error;

      return Success;
    }

    Result optapply(Node<Expr>& expr)
    {
      // apply <- expr tuple
      if (!peek(TokenKind::LParen))
        return Skip;

      rewind();
      auto app = std::make_shared<Apply>();
      app->expr = expr;
      expr = app;

      Result r = Success;

      if (opttuple(app->args) != Success)
        r = Error;

      app->location = app->args->location;
      return r;
    }

    Result onepostfix(Node<Expr>& expr)
    {
      Result r;

      if ((r = optselect(expr)) != Skip)
        return r;

      if ((r = optspecialise(expr)) != Skip)
        return r;

      if ((r = optapply(expr)) != Skip)
        return r;

      return Skip;
    }

    Result optpostorblock(Node<Expr>& expr)
    {
      // postfix <- atom ('.' (ident / symbol) / typeargs / tuple)*
      Result r;

      // If we already have an error, we're done.
      if ((r = optblockexpr(expr)) == Error)
        return Error;

      // If we got something other than a tuple, we're done.
      if ((r == Success) && (expr->kind() != Kind::Tuple))
        return Success;

      if (r == Skip)
      {
        // If we haven't already got a tuple, try to read an atom.
        if ((r = optatom(expr)) != Success)
          return r;
      }

      while (true)
      {
        Result r2 = onepostfix(expr);

        if (r2 == Skip)
          break;

        if (r2 == Error)
          r = Error;
      }

      return r;
    }

    Result optprefix(Node<Expr>& expr)
    {
      // prefix <- staticref prefix / postfix
      // preblock <- staticref preblock / blockexpr
      List<Expr> list;
      Node<Expr> last;
      Result r = Success;

      do
      {
        Node<Expr> next;
        Result r2;

        if ((r2 = optpostorblock(next)) != Success)
        {
          if (r2 == Skip)
            break;

          r = Error;
        }

        if (last)
          list.push_back(last);

        last = next;
      } while (last->kind() == Kind::StaticRef);

      if (!last)
        return Skip;

      auto block = is_blockexpr(last);

      while (list.size() > 0)
      {
        auto pre = std::make_shared<Prefix>();
        pre->op = list.back();
        pre->location = pre->op->location;
        pre->expr = last;
        pre->block = block;

        list.pop_back();
        last = pre;
      }

      expr = last;
      return r;
    }

    Result optinfix(Node<Expr>& expr)
    {
      // infix <- prefix (staticref prefix / postfix)*
      // inblock <- preblock / infix (staticref preblock / blockexpr)?
      Result r = Success;

      if ((r = optprefix(expr)) != Success)
        return r;

      if (is_blockexpr(expr))
        return r;

      Node<Expr> prev;

      while (true)
      {
        Node<Expr> next;
        Result r2;

        if ((r2 = optstaticref(next)) != Skip)
        {
          if (r2 == Error)
            r = Error;

          // TODO: how do we check precedence over staticref?
          // if (
          //   prev &&
          //   ((next->kind() != prev->kind()) ||
          //    (next->location != prev->location)))
          // {
          //   error() << next->location
          //           << "Use parentheses to indicate precedence"
          //           << text(next->location);
          // }

          prev = next;
          Node<Expr> rhs;

          if ((r2 = optprefix(rhs)) != Success)
          {
            if (r2 == Skip)
              return r;

            error() << loc() << "Expected an expression after an infix operator"
                    << line();
            r = Error;
          }

          auto inf = std::make_shared<Infix>();
          inf->location = next->location;
          inf->op = next;
          inf->left = expr;
          inf->right = rhs;
          inf->block = is_blockexpr(rhs);
          expr = inf;

          if (inf->block)
            return r;
        }
        else if ((r2 = optpostorblock(next)) != Skip)
        {
          // We have a postfix, use adjacency to mean apply.
          if (r2 == Error)
            r = Error;

          auto apply = std::make_shared<Apply>();
          apply->location = next->location;
          apply->expr = expr;
          apply->args = next;
          expr = apply;

          if (is_blockexpr(next))
            return r;
        }
        else
        {
          return r;
        }
      }
    }

    Result optexpr(Node<Expr>& expr)
    {
      // expr <- inblock / infix ('=' expr)?
      Result r;

      if ((r = optinfix(expr)) == Skip)
        return Skip;

      if (is_blockexpr(expr))
        return r;

      if (!has(TokenKind::Equals))
        return r;

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

      return r;
    }

    Result optcontrolflow(Node<Expr>& expr)
    {
      // controlflow <- break / continue / return / yield / expr
      Result r;

      if ((r = optbreak(expr)) != Skip)
        return r;

      if ((r = optcontinue(expr)) != Skip)
        return r;

      if ((r = optreturn(expr)) != Skip)
        return r;

      if ((r = optyield(expr)) != Skip)
        return r;

      if ((r = optexpr(expr)) != Skip)
        return r;

      return Skip;
    }

    Result initexpr(Node<Expr>& expr)
    {
      // initexpr <- '=' expr
      if (!has(TokenKind::Equals))
        return Skip;

      Result r;

      auto blk = std::make_shared<Block>();
      auto st = push(blk);
      blk->location = previous.location;
      expr = blk;

      Node<Expr> init;

      if ((r = optexpr(init)) != Skip)
      {
        blk->seq.push_back(init);
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
      auto find = path::canonical(path::join(source->origin, base));

      // Otherwise, look for a module relative to the standard library.
      if (find.empty())
        find = path::canonical(path::join(stdlib, base));

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
        error() << name->location << "Couldn't locate module \"" << base << "\""
                << text(name->location);
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

    Result optcaptype(Node<Type>& type)
    {
      // captype <- 'iso' / 'mut' / 'imm' / typeref
      Result r = Success;

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

      return opttyperef(type);
    }

    Result optviewtype(Node<Type>& type)
    {
      // viewtype <- (captype ('~>' / '<~'))* (captype / tupletype)
      // Left associative.
      Result r = Success;

      if ((r = opttupletype(type)) != Skip)
        return r;

      if ((r = optcaptype(type)) != Success)
        return r;

      Node<Type>& next = type;

      while (true)
      {
        if (has(TokenKind::Symbol, "~>"))
        {
          auto view = std::make_shared<ViewType>();
          view->location = previous.location;
          view->left = type;
          type = view;
          next = view->right;
        }
        else if (has(TokenKind::Symbol, "<~"))
        {
          auto extract = std::make_shared<ExtractType>();
          extract->location = previous.location;
          extract->left = type;
          type = extract;
          next = extract->right;
        }
        else
        {
          return r;
        }

        Result r2;

        if ((r2 = opttupletype(next)) == Success)
          return r;

        if (r2 == Error)
          return Error;

        if (optcaptype(next) != Success)
          r = Error;
      }
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

      if (!has(TokenKind::Symbol, "&"))
        return r;

      do
      {
        auto amploc = previous.location;
        Node<Type> next;
        Result r2;

        if ((r2 = optfunctiontype(next)) != Success)
        {
          error() << loc() << "Expected a type" << line();
          r = Error;
        }

        if (r2 != Skip)
          type = dnf::intersect(type, next, amploc);
      } while (has(TokenKind::Symbol, "&"));

      return r;
    }

    Result optuniontype(Node<Type>& type)
    {
      // uniontype <- isecttype ('|' isecttype)*
      Result r = Success;

      if ((r = optisecttype(type)) != Success)
        return r;

      if (!has(TokenKind::Symbol, "|"))
        return r;

      auto un = std::make_shared<UnionType>();
      un->location = previous.location;
      un->types.push_back(type);
      type = un;

      do
      {
        Node<Type> elem;

        if (optisecttype(elem) != Success)
        {
          error() << loc() << "Expected a type" << line();
          r = Error;
        }

        if (elem->kind() == Kind::UnionType)
        {
          auto& rhs = elem->as<UnionType>().types;
          un->types.insert(un->types.end(), rhs.begin(), rhs.end());
        }
        else
        {
          un->types.push_back(elem);
        }
      } while (has(TokenKind::Symbol, "|"));

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

      return typeexpr(type);
    }

    Result oftype(Node<Type>& type)
    {
      if (!has(TokenKind::Colon))
        return Skip;

      return typeexpr(type);
    }

    Result optparam(Node<Param>& param)
    {
      if (!has(TokenKind::Ident))
        return Skip;

      Result r = Success;
      param = std::make_shared<Param>();
      param->location = previous.location;

      if (oftype(param->type) == Error)
        r = Error;

      if (initexpr(param->init) == Error)
        r = Error;

      set_sym(param->location, param);
      return r;
    }

    Result signature(Node<Signature>& sig)
    {
      // sig <- typeparams params oftype ('throws' type)?
      Result r = Success;
      sig = std::make_shared<Signature>();

      if (typeparams(sig->typeparams) == Error)
        r = Error;

      if (has(TokenKind::LParen))
      {
        sig->location = previous.location;

        if (!has(TokenKind::RParen))
        {
          do
          {
            Node<Param> param;
            Result r2;

            if ((r2 = optparam(param)) != Success)
            {
              error() << loc() << "Expected a parameter" << line();
              r = Error;
              restart_before({TokenKind::Comma, TokenKind::RParen});
            }

            if (r2 != Skip)
              sig->params.push_back(param);
          } while (has(TokenKind::Comma));

          if (!has(TokenKind::RParen))
          {
            error() << loc() << "Expected , or )" << line();
            return Error;
          }
        }
      }
      else
      {
        error() << loc() << "Expected (" << line();
        r = Error;
      }

      if (oftype(sig->result) == Error)
        r = Error;

      if (optthrows(sig->throws) == Error)
        r = Error;

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

    Result function(Function& func)
    {
      // function <- (ident / symbol)? signature (block / ';')
      Result r = Success;

      if (has(TokenKind::Ident) || has(TokenKind::Symbol))
      {
        func.location = previous.location;
        func.name = previous.location;
      }
      else
      {
        // Replace an empy name with 'apply'.
        func.name = name_apply;
      }

      if (signature(func.signature) == Error)
        r = Error;

      for (auto& param : func.signature->params)
      {
        if (!param->type)
        {
          error() << param->location << "Function parameters must have types"
                  << text(param->location);
        }
      }

      if (!func.location.source)
        func.location = func.signature->location;

      Result r2;

      if ((r2 = optblock(func.body)) != Skip)
      {
        if (r2 == Error)
          r = Error;
      }
      else if (!has(TokenKind::Semicolon))
      {
        error() << loc() << "Expected a block or ;" << line();
        r = Error;
      }

      return r;
    }

    Result optmethod(List<Member>& members)
    {
      // method <- function
      bool ok = peek(TokenKind::Symbol) ||
        (peek(TokenKind::Ident) &&
         (peek(TokenKind::LSquare) || peek(TokenKind::LParen))) ||
        (peek(TokenKind::LSquare) || peek(TokenKind::LParen));

      rewind();

      if (!ok)
        return Skip;

      auto method = std::make_shared<Method>();
      auto st = push(method);
      Result r = Success;

      if (function(*method) != Success)
        r = Error;

      members.push_back(method);
      set_sym_parent(method->name, method);
      return r;
    }

    Result optstaticfunction(List<Member>& members)
    {
      // optstaticfunction <- 'static' function
      if (!has(TokenKind::Static))
        return Skip;

      auto func = std::make_shared<Function>();
      auto st = push(func);
      Result r = Success;

      if (function(*func) != Success)
        r = Error;

      members.push_back(func);
      set_sym_parent(func->name, func);
      return r;
    }

    Result optthrows(Node<Type>& type)
    {
      if (!has(TokenKind::Throws))
        return Skip;

      return typeexpr(type);
    }

    Result opttypeparam(Node<TypeParam>& tp)
    {
      // typeparam <- ident oftype inittype
      if (!has(TokenKind::Ident))
        return Skip;

      Result r = Success;
      tp = std::make_shared<TypeParam>();
      tp->location = previous.location;

      if (oftype(tp->type) == Error)
        r = Error;

      if (inittype(tp->init) == Error)
        r = Error;

      set_sym(tp->location, tp);
      return r;
    }

    Result typeparams(List<TypeParam>& typeparams)
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

      if (typeparams(ent.typeparams) == Error)
        r = Error;

      if (oftype(ent.inherits) == Error)
        r = Error;

      if (checkinherit(ent.inherits) == Error)
        r = Error;

      return r;
    }

    Result namedentity(NamedEntity& ent)
    {
      // namedentity <- ident? entity
      Result r = Success;

      if (optident(ent.id) == Skip)
        ent.id = ident();

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
      // typealias <- 'type' namedentity '=' type ';'
      if (!has(TokenKind::Type))
        return Skip;

      auto alias = std::make_shared<TypeAlias>();
      alias->location = previous.location;
      Result r = Success;

      if (namedentity(*alias) == Error)
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
      //  classdef / interface / typealias / using / field / method / function
      Result r;

      if ((r = classdef(members)) != Skip)
        return r;

      if ((r = interface(members)) != Skip)
        return r;

      if ((r = typealias(members)) != Skip)
        return r;

      if ((r = optusing(members)) != Skip)
        return r;

      if ((r = optstaticfunction(members)) != Skip)
        return r;

      if ((r = optmethod(members)) != Skip)
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
                     "method, or function"
                  << line();

          restart_before({TokenKind::RBrace,
                          TokenKind::Class,
                          TokenKind::Interface,
                          TokenKind::Type,
                          TokenKind::Ident,
                          TokenKind::Symbol,
                          TokenKind::Static,
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
                     "method, or function"
                  << line();

          restart_before({TokenKind::Module,
                          TokenKind::Class,
                          TokenKind::Interface,
                          TokenKind::Type,
                          TokenKind::Ident,
                          TokenKind::Symbol,
                          TokenKind::Static,
                          TokenKind::LSquare,
                          TokenKind::LParen});
        }
      }

      return final_result;
    }

    Result
    module(const std::string& path, size_t module_index, Node<Class>& program)
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

        if (files.empty())
        {
          error() << "No " << ext << " files found in " << path << std::endl;
          r = Error;
        }
        else
        {
          for (auto& file : files)
          {
            if (ext != path::extension(file))
              continue;

            auto filename = path::join(path, file);

            if (sourcefile(filename, module, moduledef) == Error)
              r = Error;
          }
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
