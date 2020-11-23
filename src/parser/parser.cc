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

    bool skip(TokenKind kind)
    {
      if (token().kind == kind)
        return true;

      backup();
      return false;
    }
  };

  bool parse_typebody(Parse& parse, List<Member>& members);

  bool parse_id(Parse& parse, ID& id)
  {
    auto& tok = parse.token();

    if (tok.kind != TokenKind::Ident)
    {
      parse.err << "Expected identifier" << err::end;
      return false;
    }

    id = tok.location;
    return true;
  }

  bool parse_type(Parse& parse, Node<Type>& type)
  {
    // TODO:
    return false;
  }

  bool parse_inittype(Parse& parse, Node<Type>& type)
  {
    if (!parse.skip(TokenKind::Equals))
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
      ID id;

      if (!parse_id(parse, id))
        return false;

      typeparams.push_back(id);

      if (parse.skip(TokenKind::RSquare))
        return true;

      if (!parse.skip(TokenKind::Comma))
      {
        parse.err << "Expected , or ]" << err::end;
        return false;
      }
    }
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

  bool parse_typealias(Parse& parse, List<Member>& members)
  {
    if (!parse.skip(TokenKind::Type))
      return true;

    auto alias = std::make_shared<TypeAlias>();

    if (!parse_entity(parse, *alias))
      return false;

    if (!parse.skip(TokenKind::Equals))
    {
      parse.err << "Expected =" << err::end;
      return false;
    }

    if (!parse_type(parse, alias->type))
      return false;

    if (!parse.skip(TokenKind::Semicolon))
    {
      parse.err << "Expected ;" << err::end;
      return false;
    }

    members.push_back(alias);
    return true;
  }

  bool parse_interface(Parse& parse, List<Member>& members)
  {
    if (!parse.skip(TokenKind::Interface))
      return true;

    auto iface = std::make_shared<Interface>();

    if (!parse_entity(parse, *iface))
      return false;

    if (!parse_typebody(parse, iface->members))
      return false;

    members.push_back(iface);
    return true;
  }

  bool parse_class(Parse& parse, List<Member>& members)
  {
    if (!parse.skip(TokenKind::Class))
      return true;

    auto cls = std::make_shared<Class>();

    if (!parse_entity(parse, *cls))
      return false;

    if (!parse_typebody(parse, cls->members))
      return false;

    members.push_back(cls);
    return true;
  }

  bool parse_members(Parse& parse, List<Member>& members)
  {
    if (!parse_class(parse, members))
      return false;

    if (!parse_interface(parse, members))
      return false;

    if (!parse_typealias(parse, members))
      return false;

    // TODO: moduledef, field, function

    parse.err
      << "Expected a module, class, interface, type alias, field, or function"
      << err::end;
    return false;
  }

  bool parse_typebody(Parse& parse, List<Member>& members)
  {
    if (!parse.skip(TokenKind::LBrace))
    {
      parse.err << "Expected {" << err::end;
      return false;
    }

    while (!parse.skip(TokenKind::RBrace))
    {
      if (!parse_members(parse, members))
        return false;

      // TODO: skip ahead and restart unless we're at the end of the file
    }

    return true;
  }

  bool parse_module(Parse& parse, List<Member>& members)
  {
    while (!parse.skip(TokenKind::End))
    {
      if (!parse_members(parse, members))
        return false;

      // TODO: discard until we recognise something
    }

    return true;
  }

  bool
  parse_file(const std::string& file, Node<Class>& module, err::Errors& err)
  {
    auto source = load_source(file, err);

    if (!source)
      return false;

    Parse parse(source, err);
    return parse_module(parse, module->members);
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
