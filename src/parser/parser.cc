#include "parser.h"

#include "../ast/path.h"

#include <deque>

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
    Token consumed;
    std::deque<Token> lookahead;
    size_t la;
    err::Errors& err;

    Parse(Source& source, err::Errors& err)
    : source(source), pos(0), la(0), err(err)
    {}

    bool peek(TokenKind kind)
    {
      if (la >= lookahead.size())
        lookahead.push_back(lex(source, pos));

      assert(la < lookahead.size());

      if (lookahead[la].kind == kind)
      {
        la++;
        return true;
      }

      return false;
    }

    Token take()
    {
      assert(la == 0);

      if (lookahead.size() == 0)
        return lex(source, pos);

      consumed = lookahead.front();
      lookahead.pop_front();
      return consumed;
    }

    void rewind()
    {
      la = 0;
    }

    bool has(TokenKind kind)
    {
      assert(la == 0);

      if (peek(kind))
      {
        rewind();
        take();
        return true;
      }

      return false;
    }

    Token previous()
    {
      return consumed;
    }

    Location location()
    {
      return consumed.location;
    }

    bool is_blockexpr(Node<Expr>& expr)
    {
      switch (expr->kind())
      {
        case Kind::Block:
        case Kind::When:
        case Kind::While:
        case Kind::For:
        case Kind::Match:
        case Kind::Conditional:
        case Kind::Preblock:
          return true;

        default:
          return false;
      }
    }

    Result optident(ID& id)
    {
      if (!has(TokenKind::Ident))
        return Skip;

      id = location();
      return Success;
    }

    Result optwhen(Node<Expr>& expr)
    {
      // when <- `when` tuple block
      if (!has(TokenKind::When))
        return Skip;

      auto when = std::make_shared<When>();
      when->location = location();
      expr = when;

      if (tuple(when->waitfor) != Success)
      {
        err << "Expected a tuple" << err::end;
        return Error;
      }

      if (block(when->behaviour) != Success)
      {
        err << "Expected a block" << err::end;
        return Error;
      }

      return Success;
    }

    Result optforloop(Node<Expr>& expr)
    {
      // for <- `for` `(` expr `in` expr `)` body
      if (!has(TokenKind::For))
        return Skip;

      auto fr = std::make_shared<For>();
      fr->location = location();
      expr = fr;

      if (!has(TokenKind::LParen))
      {
        err << "Expected (" << err::end;
        return Error;
      }

      if (optexpr(fr->left) != Success)
      {
        err << "Expected for-loop state" << err::end;
        return Error;
      }

      if (!has(TokenKind::In))
      {
        err << "Expected 'in'" << err::end;
        return Error;
      }

      if (optexpr(fr->right) != Success)
      {
        err << "Expected for-loop iterator" << err::end;
        return Error;
      }

      if (!has(TokenKind::RParen))
      {
        err << "Expected )" << err::end;
        return Error;
      }

      if (block(fr->body) != Success)
      {
        err << "Expected for-loop body" << err::end;
        return Error;
      }

      return Success;
    }

    Result optwhileloop(Node<Expr>& expr)
    {
      // while <- `while` tuple block
      if (!has(TokenKind::While))
        return Skip;

      auto wh = std::make_shared<While>();
      wh->location = location();
      expr = wh;

      if (tuple(wh->cond) != Success)
      {
        err << "Expected while-loop condition" << err::end;
        return Error;
      }

      if (block(wh->body) != Success)
      {
        err << "Expected while-loop body" << err::end;
        return Error;
      }

      return Success;
    }

    Result matchcase(Node<Case>& expr)
    {
      // case <- ???
      // TODO: case
      return Error;
    }

    Result optmatch(Node<Expr>& expr)
    {
      // match <- `match` tuple `{` case* `}`
      if (!has(TokenKind::Match))
        return Skip;

      auto match = std::make_shared<Match>();
      match->location = location();
      expr = match;

      if (tuple(match->cond) != Success)
      {
        err << "Expected match condition" << err::end;
        return Error;
      }

      if (!has(TokenKind::LBrace))
      {
        err << "Expected {" << err::end;
        return Error;
      }

      while (true)
      {
        match->cases.push_back({});
        Result r = matchcase(match->cases.back());

        if (r == Error)
          return Error;
        else if (r == Skip)
          break;
      }

      if (!has(TokenKind::RBrace))
      {
        err << "Expected a case or }" << err::end;
        return Error;
      }

      return Success;
    }

    Result optconditional(Node<Expr>& expr)
    {
      // if <- `if` tuple block (`else` block)?
      if (!has(TokenKind::If))
        return Skip;

      auto cond = std::make_shared<Conditional>();
      cond->location = location();
      expr = cond;

      if (tuple(cond->cond) != Success)
      {
        err << "Expected a condition" << err::end;
        return Error;
      }

      if (block(cond->on_true) != Success)
      {
        err << "Expected a block" << err::end;
        return Error;
      }

      if (!has(TokenKind::Else))
        return Success;

      if (block(cond->on_false) != Success)
      {
        err << "Expected a block" << err::end;
        return Error;
      }

      return Success;
    }

    Result tuple(Node<Tuple>& tup)
    {
      // tuple <- `(` expr* `)`
      if (!has(TokenKind::LParen))
      {
        err << "Expected (" << err::end;
        return Error;
      }

      tup = std::make_shared<Tuple>();
      tup->location = location();

      if (has(TokenKind::RParen))
        return Success;

      do
      {
        tup->seq.push_back({});

        if (optexpr(tup->seq.back()) != Success)
        {
          err << "Expected an expression" << err::end;
          return Error;
        }
      } while (has(TokenKind::Comma));

      if (!has(TokenKind::RParen))
      {
        err << "Expected , or )" << err::end;
        return Error;
      }

      return Success;
    }

    Result opttuple(Node<Expr>& expr)
    {
      if (!peek(TokenKind::LParen))
        return Skip;

      rewind();
      Node<Tuple> tup;
      Result r = tuple(tup);

      if (tup->seq.size() == 1)
        expr = tup->seq.front();
      else
        expr = tup;

      return r;
    }

    Result block(Node<Block>& blk)
    {
      // block <- `{` (`}` / (preblock / expr ';')* controlflow? `}`)
      if (!has(TokenKind::LBrace))
      {
        err << "Expected {" << err::end;
        return Error;
      }

      blk = std::make_shared<Block>();
      blk->location = location();

      if (has(TokenKind::RBrace))
        return Success;

      Node<Expr> expr;
      Result r;
      bool check_controlflow = false;

      do
      {
        r = optexpr(expr);

        if (r == Skip)
        {
          check_controlflow = true;
          break;
        }

        if (r == Error)
          return Error;

        blk->seq.push_back(expr);
      } while (is_blockexpr(expr) || has(TokenKind::Semicolon));

      if (check_controlflow)
      {
        r = optcontrolflow(expr);

        if (r == Error)
          return Error;

        if (r == Success)
          blk->seq.push_back(expr);
      }

      if (!has(TokenKind::RBrace))
      {
        err << "Expected an expression or }" << err::end;
        return Error;
      }

      return Success;
    }

    Result optblock(Node<Expr>& expr)
    {
      if (!peek(TokenKind::LBrace))
        return Skip;

      rewind();
      Node<Block> blk;
      Result r = block(blk);
      expr = blk;
      return r;
    }

    Result optblockexpr(Node<Expr>& expr)
    {
      // blockexpr <-
      //  block / when / conditional / match / whileloop / forloop
      Result r;

      if ((r = optblock(expr)) != Skip)
        return r;

      if ((r = optwhen(expr)) != Skip)
        return r;

      if ((r = optconditional(expr)) != Skip)
        return r;

      if ((r = optmatch(expr)) != Skip)
        return r;

      if ((r = optwhileloop(expr)) != Skip)
        return r;

      if ((r = optforloop(expr)) != Skip)
        return r;

      return Skip;
    }

    Result optbreak(Node<Expr>& expr)
    {
      // break <- `break`
      if (!has(TokenKind::Break))
        return Skip;

      auto brk = std::make_shared<Break>();
      brk->location = location();
      expr = brk;
      return Success;
    }

    Result optcontinue(Node<Expr>& expr)
    {
      // continue <- `continue`
      if (!has(TokenKind::Continue))
        return Skip;

      auto cont = std::make_shared<Continue>();
      cont->location = location();
      expr = cont;
      return Success;
    }

    Result optreturn(Node<Expr>& expr)
    {
      // return <- `return` expr?
      if (!has(TokenKind::Return))
        return Skip;

      auto ret = std::make_shared<Return>();
      ret->location = location();
      expr = ret;

      if (optexpr(ret->expr) == Error)
        return Error;

      return Success;
    }

    Result optyield(Node<Expr>& expr)
    {
      // yield <- `yield` expr
      if (!has(TokenKind::Yield))
        return Skip;

      auto yield = std::make_shared<Yield>();
      yield->location = location();
      expr = yield;

      if (optexpr(yield->expr) == Error)
        return Error;

      return Success;
    }

    Result optref(Node<Expr>& expr)
    {
      if (!has(TokenKind::Ident))
        return Skip;

      auto ref = std::make_shared<Ref>();
      ref->location = location();
      expr = ref;
      return Success;
    }

    Result optsymref(Node<Expr>& expr)
    {
      if (!has(TokenKind::Symbol))
        return Skip;

      auto ref = std::make_shared<SymRef>();
      ref->location = location();
      expr = ref;
      return Success;
    }

    Result optconstant(Node<Expr>& expr)
    {
      if (
        !has(TokenKind::String) && !has(TokenKind::Int) &&
        !has(TokenKind::Hex) && !has(TokenKind::Binary) &&
        !has(TokenKind::True) && !has(TokenKind::False))
      {
        return Skip;
      }

      auto con = std::make_shared<Constant>();
      con->location = location();
      expr = con;
      return Success;
    }

    Result optstaticref(Node<Expr>& expr)
    {
      // staticref <- id (`::` id)* `::` (id / sym)
      bool ok = peek(TokenKind::Ident) && peek(TokenKind::DoubleColon);
      rewind();

      if (!ok)
        return Skip;

      auto stat = std::make_shared<StaticRef>();
      stat->ref.push_back(take());

      while (has(TokenKind::DoubleColon))
      {
        if (has(TokenKind::Ident))
        {
          stat->ref.push_back(previous());
        }
        else if (has(TokenKind::Symbol))
        {
          stat->ref.push_back(previous());
          return Success;
        }
      }

      return Success;
    }

    Result optatom(Node<Expr>& expr)
    {
      // atom <- staticref / ref / symref / constant / lambda / new / tuple
      Result r;

      if ((r = optstaticref(expr)) != Skip)
        return r;

      if ((r = optref(expr)) != Skip)
        return r;

      if ((r = optsymref(expr)) != Skip)
        return r;

      if ((r = optconstant(expr)) != Skip)
        return r;

      // TODO: lambda, new

      if ((r = opttuple(expr)) != Skip)
        return r;

      return Skip;
    }

    Result optselect(Node<Expr>& expr)
    {
      // select <- expr `.` (id / sym)
      if (!has(TokenKind::Dot))
        return Skip;

      if (!has(TokenKind::Ident) && !has(TokenKind::Symbol))
      {
        err << "Expected an identifier or a symbol" << err::end;
        return Error;
      }

      auto sel = std::make_shared<Select>();
      sel->location = location();
      sel->expr = expr;
      sel->member = previous();
      expr = sel;
      return Success;
    }

    Result optspecialise(Node<Expr>& expr)
    {
      // specialise <- expr `[` (expr (`,` expr)*)?) `]`
      if (!has(TokenKind::LSquare))
        return Skip;

      auto spec = std::make_shared<Specialise>();
      spec->location = location();
      spec->expr = expr;
      expr = spec;

      if (has(TokenKind::RSquare))
        return Success;

      do
      {
        spec->args.push_back({});

        if (optexpr(spec->args.back()) != Success)
        {
          err << "Expected an expression" << err::end;
          return Error;
        }
      } while (has(TokenKind::Comma));

      if (!has(TokenKind::RSquare))
      {
        err << "Expected , or ]" << err::end;
        return Error;
      }

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

      if (tuple(app->args) != Success)
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

    Result optpostfix(Node<Expr>& expr)
    {
      // postfix <- atom (`.` (id / sym) / typeargs / tuple)*
      Result r;

      if ((r = optatom(expr)) != Success)
        return r;

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
    }

    Result optpostorblock(Node<Expr>& expr)
    {
      Result r;

      if ((r = optpostfix(expr)) != Skip)
        return r;

      if ((r = optblockexpr(expr)) != Skip)
        return r;

      return Skip;
    }

    template <typename T>
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
      // prefix <- (ref / symref) prefix / postfix
      // preblock <- (ref / symref) preblock / blockexpr
      List<Expr> list;
      Node<Expr> last;
      Result r;

      while (true)
      {
        Node<Expr> next;

        if ((r = optpostorblock(next)) != Success)
          break;

        if (last)
          list.push_back(last);

        last = next;
      } while ((last->kind() == Kind::Ref) || (last->kind() == Kind::SymRef));

      if (r == Error)
        return Error;

      if (is_blockexpr(last))
        buildpre<Preblock>(list, last);
      else
        buildpre<Prefix>(list, last);

      expr = last;
      return Success;
    }

    Result optinfix(Node<Expr>& expr)
    {
      // infix <- prefix ((ref / symref) infix)? / preblock
      Result r;

      if ((r = optprefix(expr)) != Success)
        return r;

      if (!peek(TokenKind::Ident) && !peek(TokenKind::Symbol))
        return Success;

      rewind();
      auto inf = std::make_shared<Infix>();

      if (optatom(inf->op) != Success)
        return Error;

      inf->location = inf->op->location;
      inf->left = expr;
      expr = inf;

      if (optinfix(inf->right) != Success)
      {
        err << "Expected an expression after an infix operator" << err::end;
        return Error;
      }

      return Success;
    }

    Result optexpr(Node<Expr>& expr)
    {
      // expr <- preblock / infix (`=` assign)?
      Result r;

      if ((r = optinfix(expr)) != Success)
        return r;

      if (is_blockexpr(expr))
        return Success;

      if (!has(TokenKind::Equals))
        return Success;

      auto asgn = std::make_shared<Assign>();
      asgn->location = location();
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
      if (!has(TokenKind::Equals))
        return Skip;

      if (optexpr(expr) != Success)
      {
        err << "Expected an initialiser expression" << err::end;
        return Error;
      }

      return Success;
    }

    Result typeexpr(Node<Type>& type)
    {
      // TODO: type expression
      return Success;
    }

    Result inittype(Node<Type>& type)
    {
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
        err << "Expected a parameter name" << err::end;
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
      // sig <- typeparams params oftype (`throws` type)? constraints
      sig = std::make_shared<Signature>();

      if (typeparams(sig->typeparams) == Error)
        return Error;

      if (!has(TokenKind::LParen))
      {
        err << "Expected (" << err::end;
        return Error;
      }

      sig->location = location();

      if (has(TokenKind::RParen))
        return Success;

      do
      {
        auto param = std::make_shared<Param>();
        sig->params.push_back(param);

        if (parameter(*param) != Success)
        {
          err << "Expected a parameter" << err::end;
          return Error;
        }
      } while (has(TokenKind::Comma));

      if (!has(TokenKind::RParen))
      {
        err << "Expected , or )" << err::end;
        return Error;
      }

      if (oftype(sig->result) == Error)
        return Error;

      if (has(TokenKind::Throws))
      {
        if (typeexpr(sig->throws) == Error)
          return Error;
      }

      if (constraints(sig->constraints) == Error)
        return Error;

      return Success;
    }

    Result field(List<Member>& members)
    {
      // field <- ident oftype initexpr `;`
      if (!has(TokenKind::Ident))
        return Skip;

      auto field = std::make_shared<Field>();
      field->location = location();
      field->id = location();

      if (oftype(field->type) == Error)
        return Error;

      if (initexpr(field->init) == Error)
        return Error;

      if (!has(TokenKind::Semicolon))
      {
        err << "Expected ;" << err::end;
        return Error;
      }

      return Success;
    }

    Result function(Function& func)
    {
      // function <- (ident / symbol)? signature (block / `;`)
      if (has(TokenKind::Ident) || has(TokenKind::Symbol))
      {
        func.location = location();
        func.name = previous();
      }

      if (signature(func.signature) == Error)
        return Error;

      if (!func.location.source)
        func.location = func.signature->location;

      if (block(func.body) == Success)
        return Success;

      if (has(TokenKind::Semicolon))
        return Success;

      err << "Expected a block or ;" << err::end;
      return Error;
    }

    Result method(List<Member>& members)
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
      members.push_back(method);
      return function(*method);
    }

    Result static_function(List<Member>& members)
    {
      // static_function <- `static` function
      if (!has(TokenKind::Static))
        return Skip;

      auto func = std::make_shared<Function>();
      members.push_back(func);
      return function(*func);
    }

    Result constraints(List<Constraint>& constraints)
    {
      // constraints <- (`where` ident oftype inittype)*
      while (has(TokenKind::Where))
      {
        auto constraint = std::make_shared<Constraint>();
        constraint->location = location();

        if (optident(constraint->id) != Success)
        {
          err << "Expected a constraint name" << err::end;
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
      // typeparams <- (`[` ident (`,` ident)* `]`)?
      if (!has(TokenKind::LSquare))
        return Skip;

      do
      {
        ID id;

        if (optident(id) != Success)
        {
          err << "Expected a type parameter name" << err::end;
          return Error;
        }

        typeparams.push_back(id);
      } while (has(TokenKind::Comma));

      if (!has(TokenKind::RSquare))
      {
        err << "Expected , or ]" << err::end;
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

      if (constraints(ent.constraints) == Error)
        return Error;

      return Success;
    }

    Result namedentity(NamedEntity& ent)
    {
      // namedentity <- ident entity
      if (optident(ent.id) != Success)
      {
        err << "Expected an entity name" << err::end;
        return Error;
      }

      if (entity(ent) == Error)
        return Error;

      return Success;
    }

    Result typealias(List<Member>& members)
    {
      // typealias <- `type` namedentity `=` type `;`
      if (!has(TokenKind::Type))
        return Skip;

      auto alias = std::make_shared<TypeAlias>();
      alias->location = location();

      if (namedentity(*alias) == Error)
        return Error;

      if (!has(TokenKind::Equals))
      {
        err << "Expected =" << err::end;
        return Error;
      }

      if (typeexpr(alias->type) == Error)
        return Error;

      if (!has(TokenKind::Semicolon))
      {
        err << "Expected ;" << err::end;
        return Error;
      }

      members.push_back(alias);
      return Success;
    }

    Result interface(List<Member>& members)
    {
      // interface <- `interface` namedentity typebody
      if (!has(TokenKind::Interface))
        return Skip;

      auto iface = std::make_shared<Interface>();
      iface->location = location();

      if (namedentity(*iface) == Error)
        return Error;

      if (typebody(iface->members) == Error)
        return Error;

      members.push_back(iface);
      return Success;
    }

    Result classdef(List<Member>& members)
    {
      // classdef <- `class` namedentity typebody
      if (!has(TokenKind::Class))
        return Skip;

      auto cls = std::make_shared<Class>();
      cls->location = location();

      if (namedentity(*cls) == Error)
        return Error;

      if (typebody(cls->members) == Error)
        return Error;

      members.push_back(cls);
      return Success;
    }

    Result moduledef(List<Member>& members)
    {
      // moduledef <- `module` entity `;`
      if (!has(TokenKind::Module))
        return Skip;

      auto module = std::make_shared<Module>();
      module->location = location();

      if (entity(*module) == Error)
        return Error;

      if (!has(TokenKind::Semicolon))
      {
        err << "Expected ;" << err::end;
        return Error;
      }

      members.push_back(module);
      return Success;
    }

    Result member(List<Member>& members, bool printerr)
    {
      // member <-
      //  moduledef / classdef / interface / typealias /
      //  field / method / function
      Result r;

      if ((r = moduledef(members)) != Skip)
        return r;

      if ((r = classdef(members)) != Skip)
        return r;

      if ((r = interface(members)) != Skip)
        return r;

      if ((r = typealias(members)) != Skip)
        return r;

      if ((r = static_function(members)) != Skip)
        return r;

      if ((r = method(members)) != Skip)
        return r;

      if ((r = field(members)) != Skip)
        return r;

      if (printerr)
      {
        err << "Expected a module, class, interface, type alias, field, "
               "method, or "
               "function"
            << err::end;
      }

      return Error;
    }

    Result typebody(List<Member>& members)
    {
      // typebody <- `{` member* `}`
      if (!has(TokenKind::LBrace))
      {
        err << "Expected {" << err::end;
        return Error;
      }

      if (has(TokenKind::RBrace))
        return Success;

      auto result = Success;
      auto printerr = true;

      while (!has(TokenKind::RBrace))
      {
        if (has(TokenKind::End))
        {
          err << "Expected }" << err::end;
          return Error;
        }

        if (member(members, printerr) != Success)
        {
          printerr = false;
          result = Error;
          take();
        }
        else
        {
          printerr = true;
        }
      }

      return Success;
    }

    Result module(List<Member>& members)
    {
      // module <- member*
      auto result = Success;
      auto printerr = true;

      while (!has(TokenKind::End))
      {
        if (member(members, printerr) != Success)
        {
          printerr = false;
          result = Error;
          take();
        }
        else
        {
          printerr = true;
        }
      }

      return result;
    }
  };

  Result
  parse_file(const std::string& file, Node<Class>& module, err::Errors& err)
  {
    auto source = load_source(file, err);

    if (!source)
      return Error;

    Parse parse(source, err);
    return parse.module(module->members);
  }

  Result parse_directory(
    const std::string& path, Node<Class>& module, err::Errors& err)
  {
    constexpr auto ext = "verona";
    auto files = path::files(path);
    auto result = Success;

    if (files.empty())
    {
      err << "No " << ext << " files found in " << path << err::end;
      return Error;
    }

    for (auto& file : files)
    {
      if (ext != path::extension(file))
        continue;

      auto filename = path::join(path, file);

      if (parse_file(filename, module, err) == Error)
        result = Error;
    }

    return result;
  }

  Node<NodeDef> parse(const std::string& path, err::Errors& err)
  {
    auto module = std::make_shared<Class>();

    if (path::is_directory(path))
      parse_directory(path, module, err);
    else
      parse_file(path, module, err);

    return module;
  }
}
