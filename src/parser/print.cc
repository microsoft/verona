// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "print.h"

#include <sstream>
#include <string_view>

namespace verona::parser
{
  constexpr size_t indent_count = 2;

  struct separator
  {};
  constexpr separator sep;

  std::ostream& operator<<(std::ostream& out, const separator& sep)
  {
    return out << ' ';
  }

  struct endtoken
  {};
  constexpr endtoken end;

  std::ostream& operator<<(std::ostream& out, const endtoken& end)
  {
    return out << ')';
  }

  struct start
  {
    std::string_view view;
    start(const char* text) : view(text) {}
  };

  std::ostream& operator<<(std::ostream& out, const start& st)
  {
    return out << sep << '(' << st.view;
  }

  template<typename T>
  std::ostream& operator<<(std::ostream& out, Node<T>& node)
  {
    return out << static_cast<Node<NodeDef>>(node);
  }

  template<typename T>
  std::ostream& operator<<(std::ostream& out, std::vector<Node<T>>& vec)
  {
    out << sep << '[';

    if (vec.size() > 0)
    {
      out << static_cast<Node<NodeDef>>(vec[0]);

      for (size_t i = 1; i < vec.size(); i++)
        out << sep << static_cast<Node<NodeDef>>(vec[i]);

      out << sep;
    }

    return out << ']';
  }

  template<typename T>
  std::ostream& operator<<(std::ostream& out, std::vector<T>& vec)
  {
    out << sep << '[';

    if (vec.size() > 0)
    {
      out << vec[0];

      for (size_t i = 1; i < vec.size(); i++)
        out << sep << vec[i];

      out << sep;
    }

    return out << ']';
  }

  std::ostream& operator<<(std::ostream& out, Location& loc)
  {
    if (!loc.source)
      return out << sep << "()";

    std::string_view v{loc.source->contents};
    return out << sep << v.substr(loc.start, loc.end - loc.start + 1);
  }

  std::ostream& operator<<(std::ostream& out, Token& token)
  {
    switch (token.kind)
    {
      case TokenKind::String:
        return out << start("string") << token.location << end;

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

  std::ostream& operator<<(std::ostream& out, Param& param)
  {
    return out << start("param") << param.id << param.type << param.init << end;
  }

  std::ostream& operator<<(std::ostream& out, UnionType& un)
  {
    return out << start("uniontype") << un.types << end;
  }

  std::ostream& operator<<(std::ostream& out, IsectType& isect)
  {
    return out << start("isecttype") << isect.types << end;
  }

  std::ostream& operator<<(std::ostream& out, TupleType& tuple)
  {
    return out << start("tupletype") << tuple.types << end;
  }

  std::ostream& operator<<(std::ostream& out, FunctionType& func)
  {
    return out << start("functiontype") << func.left << func.right << end;
  }

  std::ostream& operator<<(std::ostream& out, ViewType& view)
  {
    return out << start("viewtype") << view.left << view.right << end;
  }

  std::ostream& operator<<(std::ostream& out, ExtractType& extract)
  {
    return out << start("extracttype") << extract.left << extract.right << end;
  }

  std::ostream& operator<<(std::ostream& out, TypeName& name)
  {
    return out << start("typename") << name.id << name.typeargs << end;
  }

  std::ostream& operator<<(std::ostream& out, TypeRef& typeref)
  {
    return out << start("typeref") << typeref.typenames << end;
  }

  std::ostream& operator<<(std::ostream& out, Signature& sig)
  {
    return out << start("signature") << sig.typeparams << sig.params
               << sig.result << sig.throws << sig.constraints << end;
  }

  std::ostream& operator<<(std::ostream& out, Function& func)
  {
    return out << start("function") << func.name.location << func.signature
               << func.body << end;
  }

  std::ostream& operator<<(std::ostream& out, Method& meth)
  {
    return out << start("method") << meth.name.location << meth.signature
               << meth.body << end;
  }

  std::ostream& operator<<(std::ostream& out, Field& field)
  {
    return out << start("field") << field.id << field.type << field.init << end;
  }

  std::ostream& operator<<(std::ostream& out, Constraint& con)
  {
    return out << start("constraint") << con.id << con.type << con.init << end;
  }

  std::ostream& operator<<(std::ostream& out, Open& open)
  {
    return out << start("open") << open.type << end;
  }

  std::ostream& operator<<(std::ostream& out, TypeAlias& alias)
  {
    return out << start("typealias") << alias.id << alias.typeparams
               << alias.inherits << alias.constraints << alias.type << end;
  }

  std::ostream& operator<<(std::ostream& out, Interface& iface)
  {
    return out << start("interface") << iface.id << iface.typeparams
               << iface.inherits << iface.constraints << iface.members << end;
  }

  std::ostream& operator<<(std::ostream& out, Class& cls)
  {
    return out << start("class") << cls.id << cls.typeparams << cls.inherits
               << cls.constraints << cls.members << end;
  }

  std::ostream& operator<<(std::ostream& out, Module& module)
  {
    return out << start("module") << module.typeparams << module.inherits
               << module.constraints << end;
  }

  std::ostream& operator<<(std::ostream& out, Tuple& tuple)
  {
    return out << start("tuple") << tuple.seq << tuple.type << end;
  }

  std::ostream& operator<<(std::ostream& out, Block& block)
  {
    return out << start("block") << block.seq << end;
  }

  std::ostream& operator<<(std::ostream& out, When& when)
  {
    return out << start("when") << when.waitfor << when.behaviour << end;
  }

  std::ostream& operator<<(std::ostream& out, While& wh)
  {
    return out << start("while") << wh.cond << wh.body << end;
  }

  std::ostream& operator<<(std::ostream& out, Case& c)
  {
    return out << start("case") << c.pattern << c.guard << c.body << end;
  }

  std::ostream& operator<<(std::ostream& out, Match& match)
  {
    return out << start("match") << match.cond << match.cases << end;
  }

  std::ostream& operator<<(std::ostream& out, If& cond)
  {
    return out << start("if") << cond.cond << cond.on_true << cond.on_false
               << end;
  }

  std::ostream& operator<<(std::ostream& out, Lambda& lambda)
  {
    return out << start("lambda") << lambda.signature << lambda.body << end;
  }

  std::ostream& operator<<(std::ostream& out, Break& br)
  {
    return out << start("break") << end;
  }

  std::ostream& operator<<(std::ostream& out, Continue& cont)
  {
    return out << start("continue") << end;
  }

  std::ostream& operator<<(std::ostream& out, Return& ret)
  {
    return out << start("return") << ret.expr << end;
  }

  std::ostream& operator<<(std::ostream& out, Yield& yield)
  {
    return out << start("yield") << yield.expr << end;
  }

  std::ostream& operator<<(std::ostream& out, Assign& assign)
  {
    return out << start("assign") << assign.left << assign.right << end;
  }

  std::ostream& operator<<(std::ostream& out, Infix& infix)
  {
    return out << start("infix") << infix.op << infix.left << infix.right
               << end;
  }

  std::ostream& operator<<(std::ostream& out, Prefix& prefix)
  {
    return out << start("prefix") << prefix.op << prefix.expr << end;
  }

  std::ostream& operator<<(std::ostream& out, Select& select)
  {
    return out << start("select") << select.expr << select.member << end;
  }

  std::ostream& operator<<(std::ostream& out, Specialise& spec)
  {
    return out << start("specialise") << spec.expr << spec.typeargs << end;
  }

  std::ostream& operator<<(std::ostream& out, Apply& apply)
  {
    return out << start("apply") << apply.expr << apply.args << end;
  }

  std::ostream& operator<<(std::ostream& out, Ref& ref)
  {
    return out << start("ref") << ref.location << ref.type << end;
  }

  std::ostream& operator<<(std::ostream& out, SymRef& symref)
  {
    return out << start("symref") << symref.location << end;
  }

  std::ostream& operator<<(std::ostream& out, StaticRef& staticref)
  {
    return out << start("staticref") << staticref.ref << end;
  }

  std::ostream& operator<<(std::ostream& out, Let& let)
  {
    return out << start("let") << let.decl << end;
  }

  std::ostream& operator<<(std::ostream& out, Var& var)
  {
    return out << start("var") << var.decl << end;
  }

  std::ostream& operator<<(std::ostream& out, Constant& con)
  {
    return out << con.value;
  }

  std::ostream& operator<<(std::ostream& out, New& n)
  {
    return out << start("new") << n.args << n.in << end;
  }

  std::ostream& operator<<(std::ostream& out, ObjectLiteral& obj)
  {
    return out << start("object") << obj.inherits << obj.members << obj.in
               << end;
  }

  std::ostream& operator<<(std::ostream& out, const Node<NodeDef>& node)
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
    }

    return out;
  }

  size_t length(const std::string_view& view, size_t indent, size_t width)
  {
    size_t depth = 1;
    size_t len = 1;
    const char* delim;

    if (view[0] == '(')
    {
      delim = "()";
    }
    else if (view[0] == '[')
    {
      delim = "[]";
    }
    else
    {
      auto pos = view.find_first_of(" )]");

      if (pos == std::string::npos)
        return view.size();

      return pos;
    }

    auto current = view.substr(1);

    while (depth > 0)
    {
      auto pos = current.find_first_of(delim);

      // We've reached the end of the input.
      if (pos == std::string::npos)
        return len + current.size();

      // Handle nesting.
      if (current[pos] == delim[0])
        depth++;
      else
        depth--;

      current = current.substr(pos + 1);
      len += pos + 1;

      // Return early if we've exceeded the available space.
      if ((indent + len) > width)
        return -1;
    }

    return len;
  }

  bool print_node(
    std::ostream& out, std::string_view& view, size_t indent, size_t width)
  {
    char end;

    auto pos = view.find_first_not_of(' ');
    view = view.substr(pos);

    if (view[0] == '(')
      end = ')';
    else if (view[0] == '[')
      end = ']';
    else if ((view[0] == ')') || (view[0] == ']'))
      return false;
    else
      end = '\0';

    out << std::string(indent, ' ');
    auto len = length(view, indent, width);

    if ((len != -1) && (!end || ((indent + len) <= width)))
    {
      // Print on a single line if it's a leaf or it's short enough.
      out << view.substr(0, len) << std::endl;
      view = view.substr(len);
      return true;
    }

    // Print the header.
    pos = view.find(' ');
    out << view.substr(0, pos) << std::endl;
    view = view.substr(pos + 1);

    // Print the nodes.
    while (print_node(out, view, indent + indent_count, width))
      ;

    // Print the terminator.
    out << std::string(indent, ' ') << end << std::endl;
    view = view.substr(1);
    return true;
  }

  std::ostream& operator<<(std::ostream& out, const pretty& pret)
  {
    std::ostringstream ss;
    ss << pret.node;

    auto sexpr = ss.str();
    std::string_view view{sexpr};
    print_node(out, view, 0, pret.width);

    return out;
  }
}
