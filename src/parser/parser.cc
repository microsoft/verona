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

    Result ident(ID& id)
    {
      auto& tok = token();

      if (tok.kind != TokenKind::Ident)
      {
        err << "Expected identifier" << err::end;
        return Fail;
      }

      id = tok.location;
      return Success;
    }

    Result typeexpr(Node<Type>& type)
    {
      // TODO:
      return Success;
    }

    Result inittype(Node<Type>& type)
    {
      if (skip(TokenKind::Equals) != Success)
        return Skip;

      return typeexpr(type);
    }

    Result oftype(Node<Type>& type)
    {
      if (skip(TokenKind::Colon) != Success)
        return Skip;

      return typeexpr(type);
    }

    Result constraints(List<Constraint>& constraints)
    {
      while (true)
      {
        if (skip(TokenKind::Where) != Success)
          return Success;

        auto constraint = std::make_shared<Constraint>();

        if (ident(constraint->id) == Fail)
          return Fail;

        if (oftype(constraint->type) == Fail)
          return Fail;

        if (inittype(constraint->init) == Fail)
          return Fail;

        constraints.push_back(constraint);
      }
    }

    Result typeparams(std::vector<ID>& typeparams)
    {
      if (skip(TokenKind::LSquare) != Success)
        return Skip;

      while (true)
      {
        ID id;

        if (ident(id) == Fail)
          return Fail;

        typeparams.push_back(id);

        if (skip(TokenKind::RSquare) == Success)
          return Success;

        if (skip(TokenKind::Comma) != Success)
        {
          err << "Expected , or ]" << err::end;
          return Fail;
        }
      }
    }

    Result entity(Entity& entity)
    {
      if (ident(entity.id) == Fail)
        return Fail;

      if (typeparams(entity.typeparams) == Fail)
        return Fail;

      if (oftype(entity.inherits) == Fail)
        return Fail;

      if (constraints(entity.constraints) == Fail)
        return Fail;

      return Success;
    }

    Result typealias(List<Member>& members)
    {
      if (skip(TokenKind::Type) != Success)
        return Skip;

      auto alias = std::make_shared<TypeAlias>();

      if (entity(*alias) == Fail)
        return Fail;

      if (skip(TokenKind::Equals) != Success)
      {
        err << "Expected =" << err::end;
        return Fail;
      }

      if (typeexpr(alias->type) == Fail)
        return Fail;

      if (skip(TokenKind::Semicolon) != Success)
      {
        err << "Expected ;" << err::end;
        return Fail;
      }

      members.push_back(alias);
      return Success;
    }

    Result interface(List<Member>& members)
    {
      if (skip(TokenKind::Interface) != Success)
        return Skip;

      auto iface = std::make_shared<Interface>();

      if (entity(*iface) == Fail)
        return Fail;

      if (typebody(iface->members) == Fail)
        return Fail;

      members.push_back(iface);
      return Success;
    }

    Result classdef(List<Member>& members)
    {
      if (skip(TokenKind::Class) != Success)
        return Skip;

      auto cls = std::make_shared<Class>();

      if (entity(*cls) == Fail)
        return Fail;

      if (typebody(cls->members) == Fail)
        return Fail;

      members.push_back(cls);
      return Success;
    }

    Result memberlist(List<Member>& members)
    {
      if (classdef(members) == Fail)
        return Fail;

      if (interface(members) == Fail)
        return Fail;

      if (typealias(members) == Fail)
        return Fail;

      // TODO: moduledef, field, function

      err
        << "Expected a module, class, interface, type alias, field, or function"
        << err::end;
      return Fail;
    }

    Result typebody(List<Member>& members)
    {
      if (skip(TokenKind::LBrace) != Success)
      {
        err << "Expected {" << err::end;
        return Fail;
      }

      while (skip(TokenKind::RBrace) != Success)
      {
        if (memberlist(members) == Fail)
          return Fail;

        // TODO: skip ahead and restart unless we're at the end of the file
      }

      return Success;
    }

    Result module(List<Member>& members)
    {
      while (skip(TokenKind::End) != Success)
      {
        if (memberlist(members) == Fail)
          return Fail;

        // TODO: discard until we recognise something
      }

      return Success;
    }
  };

  Result
  parse_file(const std::string& file, Node<Class>& module, err::Errors& err)
  {
    auto source = load_source(file, err);

    if (!source)
      return Fail;

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
