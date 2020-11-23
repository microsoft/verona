#include "parser.h"

#include "../ast/path.h"
#include "lexer.h"
#include "source.h"

#include <fstream>

namespace verona::parser
{
  struct Parse;
  using parse_fn = bool (*)(Parse&);

  // TODO: restart mechanism

  enum Result
  {
    Skip,
    Success,
    Fail,
  };

  struct Parse
  {
    Source source;
    size_t pos;
    err::Errors& err;
    std::vector<Token> lookahead;
    std::vector<size_t> rewind;
    size_t la;

    Parse(Source& source, err::Errors& err)
    : source(source), pos(0), err(err), la(0)
    {}

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

    Result skip(TokenKind kind)
    {
      if (token().kind == kind)
        return Success;

      backup();
      return Skip;
    }
  };

  Result parse_typebody(Parse& parse, List<Member>& members);

  Result parse_id(Parse& parse, ID& id)
  {
    auto& tok = parse.token();

    if (tok.kind != TokenKind::Ident)
    {
      parse.err << "Expected identifier" << err::end;
      return Fail;
    }

    id = tok.location;
    return Success;
  }

  Result parse_type(Parse& parse, Node<Type>& type)
  {
    // TODO:
    return Success;
  }

  Result parse_inittype(Parse& parse, Node<Type>& type)
  {
    if (parse.skip(TokenKind::Equals) != Success)
      return Skip;

    return parse_type(parse, type);
  }

  Result parse_oftype(Parse& parse, Node<Type>& type)
  {
    if (parse.skip(TokenKind::Colon) != Success)
      return Skip;

    return parse_type(parse, type);
  }

  Result parse_constraints(Parse& parse, List<Constraint>& constraints)
  {
    while (true)
    {
      if (parse.skip(TokenKind::Where) != Success)
        return Success;

      auto constraint = std::make_shared<Constraint>();

      if (parse_id(parse, constraint->id) == Fail)
        return Fail;

      if (parse_oftype(parse, constraint->type) == Fail)
        return Fail;

      if (parse_inittype(parse, constraint->init) == Fail)
        return Fail;

      constraints.push_back(constraint);
    }
  }

  Result parse_typeparams(Parse& parse, std::vector<ID>& typeparams)
  {
    if (parse.skip(TokenKind::LSquare) != Success)
      return Skip;

    while (true)
    {
      ID id;

      if (parse_id(parse, id) == Fail)
        return Fail;

      typeparams.push_back(id);

      if (parse.skip(TokenKind::RSquare) == Success)
        return Success;

      if (parse.skip(TokenKind::Comma) != Success)
      {
        parse.err << "Expected , or ]" << err::end;
        return Fail;
      }
    }
  }

  Result parse_entity(Parse& parse, Entity& entity)
  {
    if (parse_id(parse, entity.id) == Fail)
      return Fail;

    if (parse_typeparams(parse, entity.typeparams) == Fail)
      return Fail;

    if (parse_oftype(parse, entity.inherits) == Fail)
      return Fail;

    if (parse_constraints(parse, entity.constraints) == Fail)
      return Fail;

    return Success;
  }

  Result parse_typealias(Parse& parse, List<Member>& members)
  {
    if (parse.skip(TokenKind::Type) != Success)
      return Skip;

    auto alias = std::make_shared<TypeAlias>();

    if (parse_entity(parse, *alias) == Fail)
      return Fail;

    if (parse.skip(TokenKind::Equals) != Success)
    {
      parse.err << "Expected =" << err::end;
      return Fail;
    }

    if (parse_type(parse, alias->type) == Fail)
      return Fail;

    if (parse.skip(TokenKind::Semicolon) != Success)
    {
      parse.err << "Expected ;" << err::end;
      return Fail;
    }

    members.push_back(alias);
    return Success;
  }

  Result parse_interface(Parse& parse, List<Member>& members)
  {
    if (parse.skip(TokenKind::Interface) != Success)
      return Skip;

    auto iface = std::make_shared<Interface>();

    if (parse_entity(parse, *iface) == Fail)
      return Fail;

    if (parse_typebody(parse, iface->members) == Fail)
      return Fail;

    members.push_back(iface);
    return Success;
  }

  Result parse_class(Parse& parse, List<Member>& members)
  {
    if (parse.skip(TokenKind::Class) != Success)
      return Skip;

    auto cls = std::make_shared<Class>();

    if (parse_entity(parse, *cls) == Fail)
      return Fail;

    if (parse_typebody(parse, cls->members) == Fail)
      return Fail;

    members.push_back(cls);
    return Success;
  }

  Result parse_members(Parse& parse, List<Member>& members)
  {
    if (parse_class(parse, members) == Fail)
      return Fail;

    if (parse_interface(parse, members) == Fail)
      return Fail;

    if (parse_typealias(parse, members) == Fail)
      return Fail;

    // TODO: moduledef, field, function

    parse.err
      << "Expected a module, class, interface, type alias, field, or function"
      << err::end;
    return Fail;
  }

  Result parse_typebody(Parse& parse, List<Member>& members)
  {
    if (parse.skip(TokenKind::LBrace) != Success)
    {
      parse.err << "Expected {" << err::end;
      return Fail;
    }

    while (parse.skip(TokenKind::RBrace) != Success)
    {
      if (parse_members(parse, members) == Fail)
        return Fail;

      // TODO: skip ahead and restart unless we're at the end of the file
    }

    return Success;
  }

  Result parse_module(Parse& parse, List<Member>& members)
  {
    while (parse.skip(TokenKind::End) != Success)
    {
      if (parse_members(parse, members) == Fail)
        return Fail;

      // TODO: discard until we recognise something
    }

    return Success;
  }

  Result
  parse_file(const std::string& file, Node<Class>& module, err::Errors& err)
  {
    auto source = load_source(file, err);

    if (!source)
      return Fail;

    Parse parse(source, err);
    return parse_module(parse, module->members);
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
      return Fail;
    }

    for (auto& file : files)
    {
      if (ext != path::extension(file))
        continue;

      auto filename = path::join(path, file);

      if (parse_file(filename, module, err) == Fail)
        result = Fail;
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
