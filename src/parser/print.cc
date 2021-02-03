// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "print.h"

#include "escaping.h"
#include "pretty.h"

#include <deque>
#include <sstream>
#include <variant>

namespace verona::parser
{
  // Forward reference to break cycles.
  PrettyStream& operator<<(PrettyStream& out, const Node<NodeDef>& node);

  template<typename T>
  PrettyStream& operator<<(PrettyStream& out, Node<T>& node)
  {
    return out << static_cast<Node<NodeDef>>(node);
  }

  template<typename T>
  PrettyStream& operator<<(PrettyStream& out, std::vector<T>& vec)
  {
    if (vec.size() > 0)
    {
      out << sep << start("", '[');
      out << vec[0];

      for (size_t i = 1; i < vec.size(); i++)
        out << sep << vec[i];

      out << sep << endtoken(']');
    }
    else
    {
      out << sep << "[]";
    }

    return out;
  }

  PrettyStream& operator<<(PrettyStream& out, Location& loc)
  {
    if (!loc.source)
      return out << sep << "()";

    return out << sep << loc.view();
  }

  PrettyStream& operator<<(PrettyStream& out, Token& token)
  {
    switch (token.kind)
    {
      case TokenKind::EscapedString:
        return out << start("string") << sep << q
                   << escape(escapedstring(token.location.view())) << q << end;

      case TokenKind::UnescapedString:
        return out << start("string") << sep << q
                   << escape(unescapedstring(token.location.view())) << q
                   << end;

      case TokenKind::Character:
        return out << start("char") << sep << q
                   << escape(escapedstring(token.location.view())) << q << end;

      case TokenKind::Int:
        return out << start("int") << token.location << end;

      case TokenKind::Float:
        return out << start("float") << token.location << end;

      case TokenKind::Hex:
        return out << start("hex") << token.location << end;

      case TokenKind::Binary:
        return out << start("binary") << token.location << end;

      default:
        break;
    }

    return out << token.location;
  }

  PrettyStream& operator<<(PrettyStream& out, Param& param)
  {
    return out << start("param") << param.id << param.type << param.init << end;
  }

  PrettyStream& operator<<(PrettyStream& out, UnionType& un)
  {
    return out << start("uniontype") << un.types << end;
  }

  PrettyStream& operator<<(PrettyStream& out, IsectType& isect)
  {
    return out << start("isecttype") << isect.types << end;
  }

  PrettyStream& operator<<(PrettyStream& out, TupleType& tuple)
  {
    return out << start("tupletype") << tuple.types << end;
  }

  PrettyStream& operator<<(PrettyStream& out, FunctionType& func)
  {
    return out << start("functiontype") << func.left << func.right << end;
  }

  PrettyStream& operator<<(PrettyStream& out, ViewType& view)
  {
    return out << start("viewtype") << view.left << view.right << end;
  }

  PrettyStream& operator<<(PrettyStream& out, ExtractType& extract)
  {
    return out << start("extracttype") << extract.left << extract.right << end;
  }

  PrettyStream& operator<<(PrettyStream& out, TypeName& name)
  {
    return out << start("typename") << name.id << name.typeargs << end;
  }

  PrettyStream& operator<<(PrettyStream& out, TypeRef& typeref)
  {
    return out << start("typeref") << typeref.typenames << end;
  }

  PrettyStream& operator<<(PrettyStream& out, Signature& sig)
  {
    return out << start("signature") << sig.typeparams << sig.params
               << sig.result << sig.throws << sig.constraints << end;
  }

  PrettyStream& operator<<(PrettyStream& out, Function& func)
  {
    return out << start("function") << func.name.location << func.signature
               << func.body << end;
  }

  PrettyStream& operator<<(PrettyStream& out, Method& meth)
  {
    return out << start("method") << meth.name.location << meth.signature
               << meth.body << end;
  }

  PrettyStream& operator<<(PrettyStream& out, Field& field)
  {
    return out << start("field") << field.id << field.type << field.init << end;
  }

  PrettyStream& operator<<(PrettyStream& out, Constraint& con)
  {
    return out << start("constraint") << con.id << con.type << con.init << end;
  }

  PrettyStream& operator<<(PrettyStream& out, Open& open)
  {
    return out << start("open") << open.type << end;
  }

  PrettyStream& operator<<(PrettyStream& out, TypeAlias& alias)
  {
    return out << start("typealias") << alias.id << alias.typeparams
               << alias.inherits << alias.constraints << alias.type << end;
  }

  PrettyStream& operator<<(PrettyStream& out, Interface& iface)
  {
    return out << start("interface") << iface.id << iface.typeparams
               << iface.inherits << iface.constraints << iface.members << end;
  }

  PrettyStream& operator<<(PrettyStream& out, Class& cls)
  {
    return out << start("class") << cls.id << cls.typeparams << cls.inherits
               << cls.constraints << cls.members << end;
  }

  PrettyStream& operator<<(PrettyStream& out, Module& module)
  {
    return out << start("module") << module.typeparams << module.inherits
               << module.constraints << end;
  }

  PrettyStream& operator<<(PrettyStream& out, Tuple& tuple)
  {
    return out << start("tuple") << tuple.seq << tuple.type << end;
  }

  PrettyStream& operator<<(PrettyStream& out, Block& block)
  {
    return out << start("block") << block.seq << end;
  }

  PrettyStream& operator<<(PrettyStream& out, When& when)
  {
    return out << start("when") << when.waitfor << when.behaviour << end;
  }

  PrettyStream& operator<<(PrettyStream& out, While& wh)
  {
    return out << start("while") << wh.cond << wh.body << end;
  }

  PrettyStream& operator<<(PrettyStream& out, Case& c)
  {
    return out << start("case") << c.pattern << c.guard << c.body << end;
  }

  PrettyStream& operator<<(PrettyStream& out, Match& match)
  {
    return out << start("match") << match.cond << match.cases << end;
  }

  PrettyStream& operator<<(PrettyStream& out, If& cond)
  {
    return out << start("if") << cond.cond << cond.on_true << cond.on_false
               << end;
  }

  PrettyStream& operator<<(PrettyStream& out, Lambda& lambda)
  {
    return out << start("lambda") << lambda.signature << lambda.body << end;
  }

  PrettyStream& operator<<(PrettyStream& out, Break& br)
  {
    return out << start("break") << end;
  }

  PrettyStream& operator<<(PrettyStream& out, Continue& cont)
  {
    return out << start("continue") << end;
  }

  PrettyStream& operator<<(PrettyStream& out, Return& ret)
  {
    return out << start("return") << ret.expr << end;
  }

  PrettyStream& operator<<(PrettyStream& out, Yield& yield)
  {
    return out << start("yield") << yield.expr << end;
  }

  PrettyStream& operator<<(PrettyStream& out, Assign& assign)
  {
    return out << start("assign") << assign.left << assign.right << end;
  }

  PrettyStream& operator<<(PrettyStream& out, Infix& infix)
  {
    return out << start("infix") << infix.op << infix.left << infix.right
               << end;
  }

  PrettyStream& operator<<(PrettyStream& out, Prefix& prefix)
  {
    return out << start("prefix") << prefix.op << prefix.expr << end;
  }

  PrettyStream& operator<<(PrettyStream& out, Select& select)
  {
    return out << start("select") << select.expr << select.member << end;
  }

  PrettyStream& operator<<(PrettyStream& out, Specialise& spec)
  {
    return out << start("specialise") << spec.expr << spec.typeargs << end;
  }

  PrettyStream& operator<<(PrettyStream& out, Apply& apply)
  {
    return out << start("apply") << apply.expr << apply.args << end;
  }

  PrettyStream& operator<<(PrettyStream& out, Ref& ref)
  {
    return out << start("ref") << ref.location << ref.type << end;
  }

  PrettyStream& operator<<(PrettyStream& out, SymRef& symref)
  {
    return out << start("symref") << symref.location << end;
  }

  PrettyStream& operator<<(PrettyStream& out, StaticRef& staticref)
  {
    return out << start("staticref") << staticref.path << staticref.ref << end;
  }

  PrettyStream& operator<<(PrettyStream& out, Let& let)
  {
    return out << start("let") << let.decl << end;
  }

  PrettyStream& operator<<(PrettyStream& out, Var& var)
  {
    return out << start("var") << var.decl << end;
  }

  PrettyStream& operator<<(PrettyStream& out, Constant& con)
  {
    return out << con.value;
  }

  PrettyStream& operator<<(PrettyStream& out, New& n)
  {
    return out << start("new") << n.args << n.in << end;
  }

  PrettyStream& operator<<(PrettyStream& out, ObjectLiteral& obj)
  {
    return out << start("object") << obj.inherits << obj.members << obj.in
               << end;
  }

  PrettyStream& operator<<(PrettyStream& out, Concat& con)
  {
    return out << start("concat") << con.list << end;
  }

  PrettyStream& operator<<(PrettyStream& out, const Node<NodeDef>& node)
  {
    if (!node)
      return out << sep << "()";

    switch (node->kind())
    {
      // Definitions
      case Kind::Constraint:
        return out << node->as<Constraint>();

      case Kind::Open:
        return out << node->as<Open>();

      case Kind::TypeAlias:
        return out << node->as<TypeAlias>();

      case Kind::Interface:
        return out << node->as<Interface>();

      case Kind::Class:
        return out << node->as<Class>();

      case Kind::Module:
        return out << node->as<Module>();

      case Kind::Field:
        return out << node->as<Field>();

      case Kind::Param:
        return out << node->as<Param>();

      case Kind::Signature:
        return out << node->as<Signature>();

      case Kind::Function:
        return out << node->as<Function>();

      case Kind::Method:
        return out << node->as<Method>();

      // Types
      case Kind::UnionType:
        return out << node->as<UnionType>();

      case Kind::IsectType:
        return out << node->as<IsectType>();

      case Kind::TupleType:
        return out << node->as<TupleType>();

      case Kind::FunctionType:
        return out << node->as<FunctionType>();

      case Kind::ViewType:
        return out << node->as<ViewType>();

      case Kind::ExtractType:
        return out << node->as<ExtractType>();

      case Kind::TypeName:
        return out << node->as<TypeName>();

      case Kind::TypeRef:
        return out << node->as<TypeRef>();

      // Expressions
      case Kind::Tuple:
        return out << node->as<Tuple>();

      case Kind::Block:
        return out << node->as<Block>();

      case Kind::When:
        return out << node->as<When>();

      case Kind::While:
        return out << node->as<While>();

      case Kind::Case:
        return out << node->as<Case>();

      case Kind::Match:
        return out << node->as<Match>();

      case Kind::If:
        return out << node->as<If>();

      case Kind::Lambda:
        return out << node->as<Lambda>();

      case Kind::Break:
        return out << node->as<Break>();

      case Kind::Continue:
        return out << node->as<Continue>();

      case Kind::Return:
        return out << node->as<Return>();

      case Kind::Yield:
        return out << node->as<Yield>();

      case Kind::Assign:
        return out << node->as<Assign>();

      case Kind::Infix:
        return out << node->as<Infix>();

      case Kind::Prefix:
        return out << node->as<Prefix>();

      case Kind::Inblock:
        return out << node->as<Inblock>();

      case Kind::Preblock:
        return out << node->as<Preblock>();

      case Kind::Select:
        return out << node->as<Select>();

      case Kind::Specialise:
        return out << node->as<Specialise>();

      case Kind::Apply:
        return out << node->as<Apply>();

      case Kind::Ref:
        return out << node->as<Ref>();

      case Kind::SymRef:
        return out << node->as<SymRef>();

      case Kind::StaticRef:
        return out << node->as<StaticRef>();

      case Kind::Let:
        return out << node->as<Let>();

      case Kind::Var:
        return out << node->as<Var>();

      case Kind::Constant:
        return out << node->as<Constant>();

      case Kind::New:
        return out << node->as<New>();

      case Kind::ObjectLiteral:
        return out << node->as<ObjectLiteral>();

      case Kind::Concat:
        return out << node->as<Concat>();
    }

    return out;
  }

  std::ostream& operator<<(std::ostream& out, const pretty& pret)
  {
    PrettyStream ss(out);
    ss << pret.node;
    ss.flush();
    return out;
  }
}
