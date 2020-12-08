// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "parser.h"

#include "escaping.h"
#include "path.h"

#include <cstring>
#include <iostream>
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

    List<NodeDef> symbol_stack;

    Source rewrite;
    size_t hygienic;
    Token token_apply;
    Token token_has_value;
    Token token_next;

    Result final_result;
    std::set<std::string> imports;
    std::string stdlib;

    struct SymbolPush
    {
      Parse& parser;

      SymbolPush(Parse& parser) : parser(parser) {}

      ~SymbolPush()
      {
        parser.pop();
      }
    };

    Parse(const std::string& stdlib)
    : pos(0), la(0), hygienic(0), final_result(Success), stdlib(stdlib)
    {
      rewrite = std::make_shared<SourceDef>();
      token_apply = ident("apply");
      token_has_value = ident("has_value");
      token_next = ident("next");
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
      return std::cerr;
    }

    SymbolPush push(Node<NodeDef> node)
    {
      assert(node->symbol_table() != nullptr);
      symbol_stack.push_back(node);
      return SymbolPush(*this);
    }

    void pop()
    {
      symbol_stack.pop_back();
    }

    Node<NodeDef> get_sym(const ID& id)
    {
      for (int i = symbol_stack.size() - 1; i >= 0; i--)
      {
        auto st = symbol_stack[i]->symbol_table();
        assert(st != nullptr);
        auto find = st->map.find(id);

        if (find != st->map.end())
          return find->second;
      }

      return {};
    }

    void set_sym(const ID& id, Node<NodeDef> node, SymbolTable& st)
    {
      auto find = st.map.find(id);

      if (find != st.map.end())
      {
        error() << id << "There is a previous definition of \"" << id.view()
                << "\"" << text(id) << find->first
                << "The previous definition is here" << text(find->first);
        return;
      }

      st.map.emplace(id, node);
    }

    void set_sym(const ID& id, Node<NodeDef> node)
    {
      assert(symbol_stack.size() > 0);
      auto st = symbol_stack.back()->symbol_table();
      set_sym(id, node, *st);
    }

    void set_sym_parent(const ID& id, Node<NodeDef> node)
    {
      assert(symbol_stack.size() > 1);
      auto st = symbol_stack[symbol_stack.size() - 2]->symbol_table();
      set_sym(id, node, *st);
    }

    Token ident(const char* text = "")
    {
      auto len = strlen(text);

      if (len == 0)
      {
        auto h = "$" + std::to_string(hygienic++);
        auto pos = rewrite->contents.size();
        rewrite->contents.append(h);
        len = h.size();
        return {TokenKind::Ident, {rewrite, pos, pos + len - 1}};
      }

      auto pos = rewrite->contents.find(text);

      if (pos == std::string::npos)
      {
        pos = rewrite->contents.size();
        rewrite->contents.append(text);
      }

      return {TokenKind::Ident, {rewrite, pos, pos + len - 1}};
    }

    Token ident(const std::string& s)
    {
      return ident(s.c_str());
    }

    Node<Ref> ref(const Location& loc)
    {
      auto ref = std::make_shared<Ref>();
      ref->location = loc;
      return ref;
    }

    Node<Constant> constant(const Token& tok)
    {
      if (
        (tok.kind != TokenKind::EscapedString) &&
        (tok.kind != TokenKind::UnescapedString) &&
        (tok.kind != TokenKind::Character) && (tok.kind != TokenKind::Float) &&
        (tok.kind != TokenKind::Int) && (tok.kind != TokenKind::Hex) &&
        (tok.kind != TokenKind::Binary) && (tok.kind != TokenKind::True) &&
        (tok.kind != TokenKind::False))
      {
        return {};
      }

      auto con = std::make_shared<Constant>();
      con->location = tok.location;
      con->value = tok;
      return con;
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

    bool is_localref(const ID& id)
    {
      auto def = get_sym(id);

      return def &&
        ((def->kind() == Kind::Param) || (def->kind() == Kind::Ref));
    }

    bool is_localref(Node<Expr>& expr)
    {
      if (expr->kind() != Kind::Ref)
        return false;

      auto& ref = expr->as<Ref>();
      return is_localref(ref.location);
    }

    bool is_op(Node<Expr>& expr)
    {
      return (expr->kind() == Kind::StaticRef) ||
        (expr->kind() == Kind::SymRef) ||
        ((expr->kind() == Kind::Ref) && !is_localref(expr));
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
        case Kind::Preblock:
        case Kind::Inblock:
          return true;

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

    Result optident(ID& id)
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
      expr = blk;

      auto wh = std::make_shared<While>();
      auto st = push(wh);
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
      decl->decl = ref(id.location);
      set_sym(id.location, decl->decl);
      init->left = decl;
      init->right = iter;

      // cond = (tuple (apply (select (ref $0) (ident has_value)) (tuple)))
      auto cond = std::make_shared<Tuple>();
      auto select_has_value = std::make_shared<Select>();
      select_has_value->expr = ref(id.location);
      select_has_value->member = token_has_value;
      auto apply_has_value = std::make_shared<Apply>();
      apply_has_value->expr = select_has_value;
      apply_has_value->args = std::make_shared<Tuple>();
      cond->seq.push_back(apply_has_value);

      // begin = (assign $state (apply (ref $0) (tuple))
      auto apply = std::make_shared<Apply>();
      apply->expr = ref(id.location);
      apply->args = std::make_shared<Tuple>();

      auto begin = std::make_shared<Assign>();
      begin->location = wh->body->location;
      begin->left = state;
      begin->right = apply;

      // end = (apply (select (ref $0) (ident next)) (tuple))
      auto select_next = std::make_shared<Select>();
      select_next->location = wh->body->location;
      select_next->expr = ref(id.location);
      select_next->member = token_next;

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
      param->id = ref.location;
      param->type = ref.type;
      set_sym(param->id, param);
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
        param->id = previous.location;
        set_sym(param->id, param);

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
      // ref <- ident (':' type)?
      if (!has(TokenKind::Ident))
        return Skip;

      auto ref = std::make_shared<Ref>();
      ref->location = previous.location;
      expr = ref;

      if (oftype(ref->type) == Error)
        return Error;

      return Success;
    }

    Result optlocalref(Node<Expr>& expr, bool wantlocal = true)
    {
      if (!peek(TokenKind::Ident))
        return Skip;

      bool local = is_localref(lookahead[la - 1].location);
      rewind();

      if (local != wantlocal)
        return Skip;

      return optref(expr);
    }

    Result optnonlocalref(Node<Expr>& expr)
    {
      return optlocalref(expr, false);
    }

    Result optsymref(Node<Expr>& expr)
    {
      // symref <- symbol
      if (!has(TokenKind::Symbol))
        return Skip;

      auto ref = std::make_shared<SymRef>();
      ref->location = previous.location;
      expr = ref;
      return Success;
    }

    Result optop(Node<Expr>& expr)
    {
      Result r;

      if ((r = optstaticref(expr)) != Skip)
        return r;

      if ((r = optnonlocalref(expr)) != Skip)
        return r;

      if ((r = optsymref(expr)) != Skip)
        return r;

      return Skip;
    }

    Result optconstant(Node<Expr>& expr)
    {
      // constant <-
      //  escapedstring / unescapedstring / character /
      //  float / int / hex / binary / 'true' / 'false'
      if (
        !has(TokenKind::EscapedString) && !has(TokenKind::UnescapedString) &&
        !has(TokenKind::Character) && !has(TokenKind::Float) &&
        !has(TokenKind::Int) && !has(TokenKind::Hex) &&
        !has(TokenKind::Binary) && !has(TokenKind::True) &&
        !has(TokenKind::False))
      {
        return Skip;
      }

      expr = constant(previous);
      return Success;
    }

    Result declelem(Node<Expr>& decl)
    {
      // declelem <- ref / '(' declelem (',' declelem)* ')' oftype?
      Result r;

      if ((r = optref(decl)) != Skip)
      {
        // If this is successful, the ref location is also the ident location.
        if (r == Success)
          set_sym(decl->location, decl);

        return r;
      }

      if (!has(TokenKind::LParen))
      {
        error() << loc() << "Expected a ref or (" << line();
        return Error;
      }

      auto tup = std::make_shared<Tuple>();

      do
      {
        Node<Expr> elem;

        if (declelem(elem) != Success)
          return Error;

        tup->seq.push_back(elem);
      } while (has(TokenKind::Comma));

      if (!has(TokenKind::RParen))
      {
        error() << loc() << "Expected a )" << line();
        return Error;
      }

      if (oftype(tup->type) == Error)
        return Error;

      decl = tup;
      return Success;
    }

    Result optdecl(Node<Expr>& expr)
    {
      // decl <- ('let' / 'var') declelem
      Node<Let> let;

      if (has(TokenKind::Let))
        let = std::make_shared<Let>();
      else if (has(TokenKind::Var))
        let = std::make_shared<Var>();
      else
        return Skip;

      expr = let;
      return declelem(let->decl);
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
      // staticname <- ident typeargs?
      // staticref <- staticname ('::' staticname)* ('::' ident / symbol)
      // This can also return a Specialise node.
      if (!peek(TokenKind::Ident))
        return Skip;

      if (peek(TokenKind::LSquare))
        peek_matching(TokenKind::RSquare);

      bool ok = peek(TokenKind::DoubleColon);
      rewind();

      if (!ok)
        return Skip;

      auto stat = std::make_shared<StaticRef>();
      auto typeref = std::make_shared<TypeRef>();
      stat->path = typeref;
      expr = stat;

      Result r = Success;

      while (true)
      {
        // Use the location of the last DoubleColon.
        stat->location = previous.location;
        bool more = false;

        if (peek(TokenKind::Ident))
        {
          // Figure out if we have another DoubleColon.
          if (peek(TokenKind::LSquare))
            peek_matching(TokenKind::RSquare);

          more = peek(TokenKind::DoubleColon);
          rewind();
        }

        if (!has(TokenKind::Ident) && !has(TokenKind::Symbol))
        {
          error() << loc() << "Expected an identifier or symbol" << line();
          r = Error;
          break;
        }

        if (!more)
        {
          stat->ref = previous;
          break;
        }

        auto name = std::make_shared<TypeName>();
        name->location = previous.location;
        name->value = previous;
        typeref->typenames.push_back(name);

        if (opttypeargs(name->typeargs) == Error)
          r = Error;

        if (!has(TokenKind::DoubleColon))
        {
          error() << loc() << "Expected ::" << line();
          r = Error;
          break;
        }
      };

      return r;
    }

    Result optatom(Node<Expr>& expr)
    {
      // atom <- staticref / ref / symref / constant / new / tuple
      Result r;

      if ((r = optstaticref(expr)) != Skip)
        return r;

      if ((r = optref(expr)) != Skip)
        return r;

      if ((r = optsymref(expr)) != Skip)
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
        sel->member = previous;
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

    template<typename T>
    void buildpre(List<Expr>& list, Node<Expr>& last)
    {
      while (list.size() > 0)
      {
        auto pre = std::make_shared<T>();
        pre->op = list.back();
        pre->location = pre->op->location;
        pre->expr = last;

        list.pop_back();
        last = pre;
      }
    }

    Result optprefix(Node<Expr>& expr)
    {
      // op <- staticref / nonlocalref / symref
      // prefix <- op prefix / postfix
      // preblock <- op preblock / blockexpr
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
      } while (is_op(last));

      if (!last)
        return Skip;

      if (is_blockexpr(last))
        buildpre<Preblock>(list, last);
      else
        buildpre<Prefix>(list, last);

      expr = last;
      return r;
    }

    Result optinfix(Node<Expr>& expr)
    {
      // infix <- prefix (op prefix / postfix)*
      // inblock <- preblock / infix (op preblock / blockexpr)?
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

        if ((r2 = optop(next)) != Skip)
        {
          if (r2 == Error)
            r = Error;

          if (
            prev &&
            ((next->kind() != prev->kind()) ||
             (next->location != prev->location)))
          {
            error() << next->location
                    << "Use parentheses to indicate precedence"
                    << text(next->location);
          }

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

          Node<Infix> inf;

          if (is_blockexpr(rhs))
            inf = std::make_shared<Inblock>();
          else
            inf = std::make_shared<Infix>();

          inf->location = next->location;
          inf->op = next;
          inf->left = expr;
          inf->right = rhs;
          expr = inf;

          if (is_blockexpr(rhs))
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

      if ((r = optexpr(expr)) == Skip)
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

      return r;
    }

    Result optmodulename(Node<TypeName>& name)
    {
      if (!has(TokenKind::EscapedString))
        return Skip;

      Result r = Success;

      name = std::make_shared<ModuleName>();
      name->location = previous.location;
      name->value = previous;

      // Look for a module relative to the current source file first.
      auto base =
        path::to_directory(escapedstring(name->value.location.view()));
      auto find = path::canonical(path::join(source->origin, base));

      // Otherwise, look for a module relative to the standard library.
      if (find.empty())
        find = path::canonical(path::join(stdlib, base));

      if (!find.empty())
      {
        imports.insert(find);
      }
      else
      {
        error() << name->value.location << "Couldn't locate module \"" << base
                << "\"" << text(name->value.location);
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
      typeref->location = previous.location;
      type = typeref;

      Result r = Success;

      // A typeref can start with a module name.
      Node<TypeName> name;

      if (optmodulename(name) != Skip)
      {
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
        name->value = previous;
        typeref->typenames.push_back(name);

        if (opttypeargs(name->typeargs) == Error)
          r = Error;
      } while (has(TokenKind::DoubleColon));

      return r;
    }

    Result optviewtype(Node<Type>& type)
    {
      // viewtype <- (typeref ('~>' / '<~'))* (typeref / tupletype)
      // Left associative.
      Result r = Success;

      if ((r = opttupletype(type)) != Skip)
        return r;

      if ((r = opttyperef(type)) != Success)
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

        if (opttyperef(next) != Success)
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

      auto isect = std::make_shared<IsectType>();
      isect->location = previous.location;
      isect->types.push_back(type);
      type = isect;

      do
      {
        Node<Type> elem;

        if (optfunctiontype(elem) != Success)
        {
          error() << loc() << "Expected a type" << line();
          r = Error;
        }

        isect->types.push_back(elem);
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

        un->types.push_back(elem);
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

    Result parameter(Param& param)
    {
      Result r = Success;

      if (optident(param.id) != Success)
      {
        error() << loc() << "Expected a parameter name" << line();
        r = Error;
      }

      param.location = param.id;

      if (oftype(param.type) == Error)
        r = Error;

      if (initexpr(param.init) == Error)
        r = Error;

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
            auto param = std::make_shared<Param>();
            sig->params.push_back(param);

            if (parameter(*param) != Success)
              return Error;

            set_sym(param->id, param);
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
      field->id = previous.location;
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
      set_sym(field->id, field);
      return r;
    }

    Result function(Function& func)
    {
      // function <- (ident / symbol)? signature (block / ';')
      Result r = Success;

      if (has(TokenKind::Ident) || has(TokenKind::Symbol))
      {
        func.location = previous.location;
        func.name = previous;
      }
      else
      {
        // Replace an empy name with 'apply'.
        func.name = token_apply;
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
      set_sym_parent(method->name.location, method);
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
      set_sym_parent(func->name.location, func);
      return r;
    }

    Result optthrows(Node<Type>& type)
    {
      if (!has(TokenKind::Throws))
        return Skip;

      return typeexpr(type);
    }

    Result typeparam(Node<TypeParam>& tp)
    {
      // typeparam <- ident oftype inittype
      Result r = Success;
      tp = std::make_shared<TypeParam>();

      if (optident(tp->id) != Success)
      {
        error() << loc() << "Expected a type parameter name" << line();
        r = Error;
      }

      if (oftype(tp->type) == Error)
        r = Error;

      if (inittype(tp->init) == Error)
        r = Error;

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

        if (typeparam(tp) == Error)
        {
          r = Error;
          restart_before({TokenKind::Comma, TokenKind::RSquare});
        }

        typeparams.push_back(tp);
      } while (has(TokenKind::Comma));

      if (!has(TokenKind::RSquare))
      {
        error() << loc() << "Expected , or ]" << line();
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

      return r;
    }

    Result namedentity(NamedEntity& ent)
    {
      // namedentity <- ident? entity
      Result r = Success;

      if (optident(ent.id) == Skip)
        ent.id = ident().location;

      if (entity(ent) == Error)
        r = Error;

      return r;
    }

    Result optopen(List<Member>& members)
    {
      // open <- 'open' type ';'
      if (!has(TokenKind::Open))
        return Skip;

      auto open = std::make_shared<Open>();
      open->location = previous.location;

      Result r = Success;

      if (typeexpr(open->type) == Error)
        r = Error;

      if (!has(TokenKind::Semicolon))
      {
        error() << loc() << "Expected ;" << line();
        r = Error;
      }

      members.push_back(open);
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
      //  classdef / interface / typealias / open / field / method / function
      Result r;

      if ((r = classdef(members)) != Skip)
        return r;

      if ((r = interface(members)) != Skip)
        return r;

      if ((r = typealias(members)) != Skip)
        return r;

      if ((r = optopen(members)) != Skip)
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

    Result module(const std::string& path, Node<Class>& program)
    {
      auto modulename = ident(path).location;

      if (get_sym(modulename))
        return final_result;

      Node<Module> moduledef;
      auto r = Success;

      auto module = std::make_shared<Class>();
      auto st0 = push(program);
      auto st1 = push(module);
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

        for (auto& tp : module->typeparams)
          set_sym(tp->id, tp);
      }

      // Reset hygienic for the next module.
      hygienic = 0;
      return r;
    }
  };

  std::pair<bool, Node<NodeDef>>
  parse(const std::string& path, const std::string& stdlib)
  {
    Parse parse(stdlib);
    auto program = std::make_shared<Class>();
    parse.imports.insert(path::canonical(path));

    while (!parse.imports.empty())
    {
      auto module = parse.imports.begin();
      parse.module(*module, program);
      parse.imports.erase(module);
    }

    return {parse.final_result == Success, program};
  }
}
