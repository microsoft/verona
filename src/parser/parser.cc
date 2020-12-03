// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "parser.h"

#include "path.h"

#include <cstring>
#include <deque>
#include <iostream>

namespace verona::parser
{
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
    Token previous;
    std::vector<Token> lookahead;
    size_t la;

    List<NodeDef> symbol_stack;

    Source rewrite;
    size_t hygienic;
    Token token_apply;
    Token token_has_value;
    Token token_next;

    Result final_result;

    struct SymbolPush
    {
      Parse& parser;

      SymbolPush(Parse& parser) : parser(parser) {}

      ~SymbolPush()
      {
        parser.pop();
      }
    };

    Parse(Source& source)
    : source(source), pos(0), la(0), hygienic(0), final_result(Success)
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

    Node<NodeDef> get_sym(ID id)
    {
      for (int i = symbol_stack.size() - 1; i >= 0; i--)
      {
        auto st = symbol_stack[i]->symbol_table();
        assert(st != nullptr);
        auto find = st->map.find(id);

        if (find != st->map.end())
          return find->second;
      }

      return nullptr;
    }

    void set_sym(ID id, Node<NodeDef> node, SymbolTable& st)
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

    void set_sym(ID id, Node<NodeDef> node)
    {
      assert(symbol_stack.size() > 0);
      auto st = symbol_stack.back()->symbol_table();
      set_sym(id, node, *st);
    }

    void set_sym_parent(ID id, Node<NodeDef> node)
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

    Node<Ref> ref(Location loc)
    {
      auto ref = std::make_shared<Ref>();
      ref->location = loc;
      return ref;
    }

    Location loc()
    {
      assert(lookahead.size() > 0);
      return lookahead[0].location;
    }

    text line()
    {
      return text(loc());
    }

    bool peek(TokenKind kind, const char* text = nullptr)
    {
      if (la >= lookahead.size())
        lookahead.push_back(lex(source, pos));

      assert(la < lookahead.size());

      if (lookahead[la].kind == kind)
      {
        if (!text || (lookahead[la].location == text))
        {
          la++;
          return true;
        }
      }

      return false;
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

    void rewind()
    {
      la = 0;
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

    bool is_localref(ID id)
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
      return ((expr->kind() == Kind::Ref) && !is_localref(expr)) ||
        (expr->kind() == Kind::SymRef);
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

    void
    restart_before_token_or_pair(TokenKind tok, TokenKind left, TokenKind right)
    {
      // This assumes we've already consumed the first `left` and we're looking
      // for a balanced `right` to restart before. If we encounter `tok` at
      // depth 1, restart before that instead.
      size_t depth = 1;

      while (true)
      {
        if ((depth == 1) && peek(tok))
        {
          rewind();
          return;
        }

        if (peek(left))
        {
          depth++;
        }
        else if (peek(right))
        {
          depth--;

          if (depth == 0)
          {
            rewind();
            return;
          }
        }

        take();
      }
    }

    void restart_after_parens()
    {
      restart_before_token_or_pair(
        TokenKind::End, TokenKind::LParen, TokenKind::RParen);
      take();
    }

    void restart_after_squares()
    {
      restart_before_token_or_pair(
        TokenKind::End, TokenKind::LSquare, TokenKind::RSquare);
      take();
    }

    void restart_after_braces()
    {
      restart_before_token_or_pair(
        TokenKind::End, TokenKind::LBrace, TokenKind::RBrace);
      take();
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
      // TODO: shouldn't use elses here
      if (!has(TokenKind::LParen))
      {
        error() << loc() << "Expected ( to start a for-loop condition"
                << line();
        return Error;
      }
      else if (optexpr(state) != Success)
      {
        error() << loc() << "Expected for-loop state" << line();
      }
      else if (!has(TokenKind::In))
      {
        error() << loc() << "Expected 'in'" << line();
      }
      else if (optexpr(iter) != Success)
      {
        error() << loc() << "Expected for-loop iterator" << line();
      }
      else if (!has(TokenKind::RParen))
      {
        error() << loc() << "Expected ) to end a for-loop condition" << line();
      }
      else
      {
        return Success;
      }

      restart_after_parens();
      return Error;
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
      {
        error() << loc() << "Expected for-loop body" << line();
        r = Error;
      }

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
        restart_after_braces();
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
          restart_before_token_or_pair(
            TokenKind::Comma, TokenKind::LParen, TokenKind::RParen);
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
      auto lambda = std::make_shared<Lambda>();

      if (peek(TokenKind::LSquare))
      {
        rewind();
        auto st = push(lambda);

        if (signature(lambda->signature) != Success)
          return Error;
      }
      else if (opttuple(expr) == Success)
      {
        if (
          !peek(TokenKind::Throws) && !peek(TokenKind::Where) &&
          !peek(TokenKind::FatArrow))
        {
          // Return a successful tuple instead of a lambda.
          return Success;
        }

        rewind();
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
          return Error;
        }

        if (optthrows(sig->throws) == Error)
          return Error;

        if (optconstraints(sig->constraints) == Error)
          return Error;
      }
      else if (peek(TokenKind::Ident) && peek(TokenKind::FatArrow))
      {
        rewind();
        has(TokenKind::Ident);

        auto st = push(lambda);
        auto param = std::make_shared<Param>();
        param->location = previous.location;
        param->id = previous.location;
        set_sym(param->id, param);

        auto sig = std::make_shared<Signature>();
        sig->location = previous.location;
        sig->params.push_back(param);

        lambda->signature = sig;
      }
      else
      {
        rewind();
        return Skip;
      }

      auto st = push(lambda);
      expr = lambda;

      if (!has(TokenKind::FatArrow))
      {
        error() << loc() << "Expected =>" << line();
        return Error;
      }

      lambda->location = previous.location;

      if (optexpr(lambda->body) != Success)
      {
        error() << loc() << "Expected a lambda body" << line();
        return Error;
      }

      return Success;
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

    Result optconstant(Node<Expr>& expr)
    {
      // constant <-
      //  escapedstring / unescapedstring / character /
      //  float / int / hex / binary / 'true' / 'false'
      if (
        !has(TokenKind::Character) && !has(TokenKind::Float) &&
        !has(TokenKind::Int) && !has(TokenKind::Hex) &&
        !has(TokenKind::Binary) && !has(TokenKind::True) &&
        !has(TokenKind::False))
      {
        return Skip;
      }

      auto con = std::make_shared<Constant>();
      con->location = previous.location;
      con->value = previous;
      expr = con;
      return Success;
    }

    Result optstring(Node<Expr>& expr)
    {
      // string <- escapedstring / unescapedstring
      if (!has(TokenKind::EscapedString) && !has(TokenKind::UnescapedString))
        return Skip;

      auto con = std::make_shared<Constant>();
      con->location = previous.location;
      con->value = previous;
      expr = con;
      return Success;
    }

    Result optconcatelem(Node<Expr>& expr)
    {
      Result r;

      if ((r = optstring(expr)) != Skip)
        return r;

      if ((r = optlocalref(expr)) != Skip)
        return r;

      if ((r = opttuple(expr)) != Skip)
        return r;

      return Skip;
    }

    Result optconcat(Node<Expr>& expr)
    {
      // concat <- string (string / localref / tuple)*
      Result r;

      if ((r = optstring(expr)) != Success)
        return r;

      auto concat = std::make_shared<Concat>();
      concat->list.push_back(expr);
      expr = concat;

      while (true)
      {
        Node<Expr> elem;
        r = optconcatelem(elem);

        if (r == Skip)
        {
          if (concat->list.size() == 1)
            expr = concat->list.front();

          return Success;
        }

        concat->list.push_back(elem);
      }
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

      if (peek(TokenKind::LParen))
      {
        rewind();
        auto n = std::make_shared<New>();
        n->location = previous.location;
        expr = n;

        if (opttuple(n->args) != Success)
          return Error;

        if (has(TokenKind::In))
        {
          if (optident(n->in) != Success)
            return Error;
        }

        return Success;
      }

      auto obj = std::make_shared<ObjectLiteral>();
      auto st = push(obj);
      obj->location = previous.location;
      expr = obj;

      if (!peek(TokenKind::LBrace))
      {
        if (typeexpr(obj->inherits) == Error)
          return Error;
      }
      else
      {
        rewind();
      }

      if (typebody(obj->members) != Success)
        return Error;

      if (has(TokenKind::In))
      {
        if (optident(obj->in) != Success)
          return Error;
      }

      return Success;
    }

    Result optstaticref(Node<Expr>& expr)
    {
      // staticref <- typeref ('::' ident / symbol)
      // This can also return a Specialise node.
      bool ok = peek(TokenKind::Ident) &&
        (peek(TokenKind::LSquare) || peek(TokenKind::DoubleColon));
      rewind();

      if (!ok)
        return Skip;

      Node<Type> typeref;
      Result r;

      if ((r = opttyperef(typeref)) != Success)
        return r;

      Token trailing;

      if (has(TokenKind::DoubleColon))
      {
        if (!has(TokenKind::Symbol))
        {
          error() << loc() << "Expected symbol" << line();
          return Error;
        }

        trailing = previous;
      }

      TypeRef& t = typeref->as<TypeRef>();

      if (!trailing.location.source && (t.typenames.size() == 1))
      {
        // Turn this into a Specialise node instead.
        auto& name = t.typenames.front();
        auto spec = std::make_shared<Specialise>();
        spec->location = name->location;
        spec->expr = ref(name->id);
        spec->typeargs = name->typeargs;
        expr = spec;
      }
      else
      {
        auto stat = std::make_shared<StaticRef>();
        stat->path = typeref;
        stat->ref = trailing;
        expr = stat;

        // If we have no trailing symbol, use the last entry as the trailing
        // ident and turn its typeargs into a Specialise node.
        if (!trailing.location.source)
        {
          auto last = t.typenames.back();
          t.typenames.pop_back();
          stat->ref = {TokenKind::Ident, last->id};

          auto spec = std::make_shared<Specialise>();
          spec->location = last->location;
          spec->expr = stat;
          spec->typeargs = last->typeargs;
          expr = spec;
        }
      }

      return Success;
    }

    Result optatom(Node<Expr>& expr)
    {
      // atom <- staticref / ref / symref / constant / new / tuple
      // This can also return a Specialise, as staticref can do so.
      Result r;

      if ((r = optstaticref(expr)) != Skip)
        return r;

      if ((r = optref(expr)) != Skip)
        return r;

      if ((r = optsymref(expr)) != Skip)
        return r;

      if ((r = optconstant(expr)) != Skip)
        return r;

      if ((r = optconcat(expr)) != Skip)
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

      if (!has(TokenKind::Ident) && !has(TokenKind::Symbol))
      {
        error() << loc() << "Expected an identifier or a symbol" << line();
        return Error;
      }

      auto sel = std::make_shared<Select>();
      sel->location = previous.location;
      sel->expr = expr;
      sel->member = previous;
      expr = sel;
      return Success;
    }

    Result opttypeargs(List<Expr>& typeargs)
    {
      // typeargs <- '[' (expr (',' expr)*)?) ']'
      if (!has(TokenKind::LSquare))
        return Skip;

      do
      {
        Node<Expr> arg;

        if (optexpr(arg) != Success)
        {
          error() << loc() << "Expected a type argument" << line();
          return Error;
        }

        typeargs.push_back(arg);
      } while (has(TokenKind::Comma));

      if (!has(TokenKind::RSquare))
      {
        error() << loc() << "Expected , or ]" << line();
        return Error;
      }

      return Success;
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

      if (opttuple(app->args) != Success)
        return Error;

      app->location = app->args->location;
      return Success;
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

      // postfix <- atom ('.' (ident / symbol) / typeargs / tuple)*
      while (true)
      {
        switch (onepostfix(expr))
        {
          case Success:
            break;

          case Error:
            return Error;

          case Skip:
            return Success;
        }
      }

      return Skip;
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
      // op <- nonlocalref / symref
      // prefix <- op prefix / postfix
      // preblock <- op preblock / blockexpr
      List<Expr> list;
      Node<Expr> last;
      Result r;

      do
      {
        Node<Expr> next;

        if ((r = optpostorblock(next)) != Success)
          break;

        if (last)
          list.push_back(last);

        last = next;
      } while (is_op(last));

      if (r == Error)
        return Error;

      if (!last)
        return Skip;

      if (is_blockexpr(last))
        buildpre<Preblock>(list, last);
      else
        buildpre<Prefix>(list, last);

      expr = last;
      return Success;
    }

    Result optinfix(Node<Expr>& expr)
    {
      // infix <- prefix (op prefix)*
      // inblock <- preblock / infix (op preblock)?
      Result r;

      if ((r = optprefix(expr)) != Success)
        return r;

      if (is_blockexpr(expr))
        return Success;

      Node<Expr> prev;

      while (true)
      {
        Node<Expr> op;
        Node<Expr> rhs;

        // Fetch a nonlocalref or a symref.
        if ((r = optnonlocalref(op)) == Skip)
          r = optsymref(op);

        if (r == Error)
          return Error;

        if (r == Skip)
          return Success;

        if (
          prev &&
          ((op->kind() != prev->kind()) || (op->location != prev->location)))
        {
          error() << op->location << "Use parentheses to indicate precedence"
                  << text(op->location);
        }

        prev = op;

        if (optprefix(rhs) != Success)
        {
          error() << loc() << "Expected an expression after an infix operator"
                  << line();
          return Error;
        }

        if (is_blockexpr(rhs))
        {
          auto inf = std::make_shared<Inblock>();
          inf->location = op->location;
          inf->op = op;
          inf->left = expr;
          inf->right = rhs;
          expr = inf;
          return Success;
        }

        auto inf = std::make_shared<Infix>();
        inf->location = op->location;
        inf->op = op;
        inf->left = expr;
        inf->right = rhs;
        expr = inf;
      }
    }

    Result optexpr(Node<Expr>& expr)
    {
      // expr <- inblock / infix ('=' expr)?
      Result r;

      if ((r = optinfix(expr)) != Success)
        return r;

      if (is_blockexpr(expr))
        return Success;

      if (!has(TokenKind::Equals))
        return Success;

      auto asgn = std::make_shared<Assign>();
      asgn->location = previous.location;
      asgn->left = expr;
      expr = asgn;

      return optexpr(asgn->right);
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

      if (optexpr(expr) != Success)
      {
        error() << loc() << "Expected an initialiser expression" << line();
        return Error;
      }

      return Success;
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

      do
      {
        Node<Type> elem;

        if (typeexpr(elem) != Success)
          return Error;

        tup->types.push_back(elem);
      } while (has(TokenKind::Comma));

      if (!has(TokenKind::RParen))
      {
        error() << loc() << "Expected )" << line();
        return Error;
      }

      return Success;
    }

    Result opttyperef(Node<Type>& type)
    {
      // typeref <- ident typeargs? ('::' ident typeargs?)*
      if (!peek(TokenKind::Ident))
        return Skip;

      rewind();
      auto typeref = std::make_shared<TypeRef>();
      typeref->location = previous.location;
      type = typeref;

      do
      {
        bool ok = peek(TokenKind::Ident);
        rewind();

        if (!ok)
          return Success;

        has(TokenKind::DoubleColon);
        has(TokenKind::Ident);

        auto name = std::make_shared<TypeName>();
        name->location = previous.location;
        name->id = previous.location;
        typeref->typenames.push_back(name);

        if (opttypeargs(name->typeargs) == Error)
          return Error;
      } while (peek(TokenKind::DoubleColon));

      return Success;
    }

    Result optviewtype(Node<Type>& type)
    {
      // viewtype <- (typeref ('~>' / '<~'))* (typeref / tupletype)
      // Left associative.
      Result r;

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
          return Success;
        }

        if ((r = opttupletype(next)) != Skip)
          return r;

        if (opttyperef(next) != Success)
          return Error;
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
      Result r;

      if ((r = optfunctiontype(type)) != Success)
        return r;

      if (!has(TokenKind::Symbol, "&"))
        return Success;

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
          return Error;
        }

        isect->types.push_back(elem);
      } while (has(TokenKind::Symbol, "&"));

      return Success;
    }

    Result optuniontype(Node<Type>& type)
    {
      // uniontype <- isecttype ('|' isecttype)*
      Result r;

      if ((r = optisecttype(type)) != Success)
        return r;

      if (!has(TokenKind::Symbol, "|"))
        return Success;

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
          return Error;
        }

        un->types.push_back(elem);
      } while (has(TokenKind::Symbol, "|"));

      return Success;
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
      if (optident(param.id) != Success)
      {
        error() << loc() << "Expected a parameter name" << line();
        return Error;
      }

      param.location = param.id;

      if (oftype(param.type) == Error)
        return Error;

      if (initexpr(param.init) == Error)
        return Error;

      return Success;
    }

    Result signature(Node<Signature>& sig)
    {
      // sig <- typeparams params oftype ('throws' type)? constraints
      sig = std::make_shared<Signature>();

      if (typeparams(sig->typeparams) == Error)
        return Error;

      if (!has(TokenKind::LParen))
      {
        error() << loc() << "Expected (" << line();
        return Error;
      }

      sig->location = previous.location;

      if (has(TokenKind::RParen))
        return Success;

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

      if (oftype(sig->result) == Error)
        return Error;

      if (optthrows(sig->throws) == Error)
        return Error;

      if (optconstraints(sig->constraints) == Error)
        return Error;

      return Success;
    }

    Result optfield(List<Member>& members)
    {
      // field <- ident oftype initexpr ';'
      if (!has(TokenKind::Ident))
        return Skip;

      auto field = std::make_shared<Field>();
      field->location = previous.location;
      field->id = previous.location;

      if (oftype(field->type) == Error)
        return Error;

      if (initexpr(field->init) == Error)
        return Error;

      if (!has(TokenKind::Semicolon))
      {
        error() << loc() << "Expected ;" << line();
        return Error;
      }

      members.push_back(field);
      set_sym(field->id, field);
      return Success;
    }

    Result function(Function& func)
    {
      // function <- (ident / symbol)? signature (block / ';')
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
        return Error;

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

      if (optblock(func.body) == Success)
        return Success;

      if (has(TokenKind::Semicolon))
        return Success;

      error() << loc() << "Expected a block or ;" << line();
      return Error;
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

      if (function(*method) != Success)
        return Error;

      members.push_back(method);
      set_sym_parent(method->name.location, method);
      return Success;
    }

    Result optstaticfunction(List<Member>& members)
    {
      // optstaticfunction <- 'static' function
      if (!has(TokenKind::Static))
        return Skip;

      auto func = std::make_shared<Function>();
      auto st = push(func);

      if (function(*func) != Success)
        return Error;

      members.push_back(func);
      set_sym_parent(func->name.location, func);
      return Success;
    }

    Result optthrows(Node<Type>& type)
    {
      if (!has(TokenKind::Throws))
        return Skip;

      return typeexpr(type);
    }

    Result optconstraints(List<Constraint>& constraints)
    {
      // constraints <- ('where' ident oftype inittype)*
      while (has(TokenKind::Where))
      {
        auto constraint = std::make_shared<Constraint>();
        constraint->location = previous.location;

        if (optident(constraint->id) != Success)
        {
          error() << loc() << "Expected a constraint name" << line();
          return Error;
        }

        if (oftype(constraint->type) == Error)
          return Error;

        if (inittype(constraint->init) == Error)
          return Error;

        constraints.push_back(constraint);
      }

      return Success;
    }

    Result typeparams(std::vector<ID>& typeparams)
    {
      // typeparams <- ('[' ident (',' ident)* ']')?
      if (!has(TokenKind::LSquare))
        return Skip;

      do
      {
        ID id;

        if (optident(id) != Success)
        {
          error() << loc() << "Expected a type parameter name" << line();
          return Error;
        }

        typeparams.push_back(id);
      } while (has(TokenKind::Comma));

      if (!has(TokenKind::RSquare))
      {
        error() << loc() << "Expected , or ]" << line();
        return Error;
      }

      return Success;
    }

    Result entity(Entity& ent)
    {
      // entity <- typeparams oftype constraints
      if (typeparams(ent.typeparams) == Error)
        return Error;

      if (oftype(ent.inherits) == Error)
        return Error;

      if (optconstraints(ent.constraints) == Error)
        return Error;

      return Success;
    }

    Result namedentity(NamedEntity& ent)
    {
      // namedentity <- ident entity
      if (optident(ent.id) != Success)
      {
        error() << loc() << "Expected an entity name" << line();
        return Error;
      }

      if (entity(ent) == Error)
        return Error;

      return Success;
    }

    Result typealias(List<Member>& members)
    {
      // typealias <- 'type' namedentity '=' type ';'
      if (!has(TokenKind::Type))
        return Skip;

      auto alias = std::make_shared<TypeAlias>();
      alias->location = previous.location;

      if (namedentity(*alias) == Error)
        return Error;

      if (!has(TokenKind::Equals))
      {
        error() << loc() << "Expected =" << line();
        return Error;
      }

      if (typeexpr(alias->type) == Error)
        return Error;

      if (!has(TokenKind::Semicolon))
      {
        error() << loc() << "Expected ;" << line();
        return Error;
      }

      members.push_back(alias);
      set_sym(alias->id, alias);
      return Success;
    }

    Result interface(List<Member>& members)
    {
      // interface <- 'interface' namedentity typebody
      if (!has(TokenKind::Interface))
        return Skip;

      auto iface = std::make_shared<Interface>();
      auto st = push(iface);
      iface->location = previous.location;

      if (namedentity(*iface) == Error)
        return Error;

      if (typebody(iface->members) == Error)
        return Error;

      members.push_back(iface);
      set_sym_parent(iface->id, iface);
      return Success;
    }

    Result classdef(List<Member>& members)
    {
      // classdef <- 'class' namedentity typebody
      if (!has(TokenKind::Class))
        return Skip;

      auto cls = std::make_shared<Class>();
      auto st = push(cls);
      cls->location = previous.location;

      if (namedentity(*cls) == Error)
        return Error;

      if (typebody(cls->members) == Error)
        return Error;

      members.push_back(cls);
      set_sym_parent(cls->id, cls);
      return Success;
    }

    Result optmoduledef(Node<Module>& module)
    {
      // moduledef <- 'module' entity ';'
      if (!has(TokenKind::Module))
        return Skip;

      if (module)
      {
        error() << previous.location << "The module has already been defined"
                << text(previous.location) << module->location
                << "The previous definition is here" << text(module->location);

        // Restart after ;
        while (!has(TokenKind::Semicolon))
          ;
        return Skip;
      }

      module = std::make_shared<Module>();
      module->location = previous.location;

      if (entity(*module) == Error)
        return Error;

      if (!has(TokenKind::Semicolon))
      {
        error() << loc() << "Expected ;" << line();
        return Error;
      }

      return Success;
    }

    Result optmember(List<Member>& members)
    {
      // member <- classdef / interface / typealias / field / method / function
      Result r;

      if ((r = classdef(members)) != Skip)
        return r;

      if ((r = interface(members)) != Skip)
        return r;

      if ((r = typealias(members)) != Skip)
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
      if (!has(TokenKind::LBrace))
      {
        error() << loc() << "Expected {" << line();
        return Error;
      }

      if (has(TokenKind::RBrace))
        return Success;

      while (!has(TokenKind::RBrace))
      {
        if (has(TokenKind::End))
        {
          error() << loc() << "Expected }" << line();
          return Error;
        }

        if (optmember(members) != Skip)
          continue;

        error() << loc()
                << "Expected a class, interface, type alias, field, "
                   "method, or function"
                << line();

        // Restart.
        while (!peek(TokenKind::RBrace) && !peek(TokenKind::Class) &&
               !peek(TokenKind::Interface) && !peek(TokenKind::Type) &&
               !peek(TokenKind::Ident) && !peek(TokenKind::Symbol) &&
               !peek(TokenKind::Static))
        {
          take();
        }
      }

      return Success;
    }

    void module(Node<Module>& module, List<Member>& members)
    {
      // module <- (moduledef / member)*
      bool ok = true;

      while (!has(TokenKind::End))
      {
        if ((optmoduledef(module) != Skip) || (optmember(members) != Skip))
        {
          ok = true;
          continue;
        }

        if (ok)
        {
          ok = false;
          error() << loc()
                  << "Expected a module, class, interface, type alias, field, "
                     "method, or function"
                  << line();
        }

        // Restart.
        take();
      }
    }
  };

  Result
  parse_file(const std::string& file, Node<Class>& top, Node<Module>& module)
  {
    auto source = load_source(file);

    if (!source)
      return Error;

    Parse parse(source);
    auto st = parse.push(top);
    parse.module(module, top->members);
    return parse.final_result;
  }

  Result parse_directory(const std::string& path, Node<Class>& top)
  {
    Node<Module> module;
    auto result = Success;

    if (!path::is_directory(path))
    {
      result = parse_file(path, top, module);
    }
    else
    {
      constexpr auto ext = "verona";
      auto files = path::files(path);

      if (files.empty())
      {
        std::cerr << "No " << ext << " files found in " << path << std::endl;
        return Error;
      }

      for (auto& file : files)
      {
        if (ext != path::extension(file))
          continue;

        auto filename = path::join(path, file);

        if (parse_file(filename, top, module) == Error)
          result = Error;
      }
    }

    // TODO: put module stuff in top
    return result;
  }

  std::pair<bool, Node<NodeDef>> parse(const std::string& path)
  {
    auto top = std::make_shared<Class>();
    Result r = parse_directory(path, top);
    return {r == Success, top};
  }
}
