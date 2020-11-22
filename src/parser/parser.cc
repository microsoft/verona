#include "parser.h"

#include "../ast/path.h"
#include "lexer.h"
#include "source.h"

#include <fstream>

namespace verona::parser
{
  struct Parse;
  using parse_fn = bool (*)(Parse&);

  struct Parse
  {
    Source source;
    size_t pos;
    err::Errors& err;
    List<NodeDef> stack;
    std::vector<Token> lookahead;
    std::vector<size_t> rewind;
    size_t la;

    Parse(Source& source, err::Errors& err, Node<Class>& module)
    : source(source), pos(0), err(err), la(0)
    {
      stack.push_back(module);
    }

    void start()
    {
      rewind.push_back(la);
    }

    void success()
    {
      la = rewind.back();
      lookahead.resize(la);
      rewind.pop_back();
    }

    void fail()
    {
      la = rewind.back();
      rewind.pop_back();
    }

    Token& token()
    {
      if (la >= lookahead.size())
        lookahead.push_back(lex(source, pos));

      assert(la < lookahead.size());
      auto& tok = lookahead[la];
      la++;
      return tok;
    }

    void backup(size_t n = 1)
    {
      assert(n <= la);
      la -= n;
    }

    bool skip(TokenKind kind)
    {
      if (token().kind == kind)
        return true;

      backup();
      return false;
    }

    void push(Node<NodeDef> node)
    {
      stack.push_back(node);
    }

    void pop()
    {
      stack.pop_back();
    }
  };

  struct Rule
  {
    Parse& parse;
    bool success;

    Rule(Parse& parse) : parse(parse), success(false)
    {
      parse.start();
    }

    ~Rule()
    {
      if (success)
        parse.success();
      else
        parse.fail();
    }

    bool operator()()
    {
      success = true;
      return true;
    }
  };

  bool parse_id(Parse& parse, ID& id)
  {
    auto& tok = parse.token();

    if (tok.kind != TokenKind::Ident)
      return false;

    id = tok.location;
    return false;
  }

  bool parse_type(Parse& parse, Node<Type>& type)
  {
    // TODO:
    return false;
  }

  bool parse_inittype(Parse& parse, Node<Type>& type)
  {
    if (!parse.skip(TokenKind::Equal))
      return true;

    return parse_type(parse, type);
  }

  bool parse_oftype(Parse& parse, Node<Type>& type)
  {
    if (!parse.skip(TokenKind::Colon))
      return true;

    return parse_type(parse, type);
  }

  bool parse_constraints(Parse& parse, List<Constraint>& constraints)
  {
    while (true)
    {
      if (!parse.skip(TokenKind::Where))
        return true;

      auto constraint = std::make_shared<Constraint>();

      if (!parse_id(parse, constraint->id))
        return false;

      if (!parse_oftype(parse, constraint->type))
        return false;

      if (!parse_inittype(parse, constraint->init))
        return false;

      constraints.push_back(constraint);
    }
  }

  bool parse_typeparams(Parse& parse, std::vector<ID>& typeparams)
  {
    if (!parse.skip(TokenKind::LSquare))
      return true;

    while (true)
    {
      auto& tok = parse.token();

      switch (tok.kind)
      {
        case TokenKind::LSquare:
          return true;

        case TokenKind::Ident:
        {
          typeparams.push_back(tok.location);
          break;
        }

        default:
        {
          parse.err << "Expected identifier or ]" << err::end;
          return false;
        }
      }

      tok = parse.token();

      switch (tok.kind)
      {
        case TokenKind::LSquare:
          return true;

        case TokenKind::Comma:
          break;

        default:
        {
          parse.err << "Expected , or ]" << err::end;
          return false;
        }
      }
    }
  }

  Node<Class> parse_class(Parse& parse);

  bool parse_members(Parse& parse, List<Member>& members)
  {
    if (!parse.skip(TokenKind::LBrace))
      return false;

    while (true)
    {
      if (parse.skip(TokenKind::RBrace))
        return true;

      Node<Member> member;

      // TODO: choice semantics
      if ((member = parse_class(parse)))
        members.push_back(member);
      // TODO: else other kinds of members
      else
      {
        // TODO: error, restart behaviour
      }
    }

    return true;
  }

  bool parse_entity(Parse& parse, Entity& entity)
  {
    if (!parse_id(parse, entity.id))
      return false;

    if (!parse_typeparams(parse, entity.typeparams))
      return false;

    if (!parse_oftype(parse, entity.inherits))
      return false;

    if (!parse_constraints(parse, entity.constraints))
      return false;

    return true;
  }

  Node<Class> parse_class(Parse& parse)
  {
    // TODO: separate out the ok stuff?
    // only needed for a choice rule.
    // return something besides a Node<Class> ?
    Rule ok(parse);

    if (!parse.skip(TokenKind::Class))
      return {};

    auto cls = std::make_shared<Class>();

    if (!parse_entity(parse, *cls))
      return {};

    if (!parse_members(parse, cls->members))
      return {};

    ok();
    return cls;
  }

  bool
  parse_file(const std::string& file, Node<Class>& module, err::Errors& err)
  {
    auto source = load_source(file, err);

    if (!source)
      return;

    Parse parse(source, err, module);
    return parse_members(parse, module->members);
  }

  void parse_directory(
    const std::string& path, Node<Class>& module, err::Errors& err)
  {
    constexpr auto ext = "verona";
    auto files = path::files(path);

    if (files.empty())
      err << "No " << ext << " files found in " << path << err::end;

    for (auto& file : files)
    {
      if (ext != path::extension(file))
        continue;

      auto filename = path::join(path, file);
      parse_file(filename, module, err);
    }
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
