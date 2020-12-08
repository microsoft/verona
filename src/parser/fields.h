// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "ast.h"

namespace verona::parser
{
  template<typename T>
  void fieldsof(T& target, NodeDef& node)
  {
    // Used for nodes with no fields.
  }

  template<typename T>
  void fieldsof(T& target, Open& open)
  {
    target << open.type;
  }

  template<typename T>
  void fieldsof(T& target, TypeAlias& alias)
  {
    target << alias.id << alias.typeparams << alias.inherits << alias.type;
  }

  template<typename T>
  void fieldsof(T& target, Interface& iface)
  {
    target << iface.id << iface.typeparams << iface.inherits << iface.members;
  }

  template<typename T>
  void fieldsof(T& target, Class& cls)
  {
    target << cls.id << cls.typeparams << cls.inherits << cls.members;
  }

  template<typename T>
  void fieldsof(T& target, Module& module)
  {
    target << module.typeparams << module.inherits;
  }

  template<typename T>
  void fieldsof(T& target, Field& field)
  {
    target << field.id << field.type << field.init;
  }

  template<typename T>
  void fieldsof(T& target, Param& param)
  {
    target << param.id << param.type << param.init;
  }

  template<typename T>
  void fieldsof(T& target, TypeParam& tp)
  {
    target << tp.id << tp.type << tp.init;
  }

  template<typename T>
  void fieldsof(T& target, Signature& sig)
  {
    target << sig.typeparams << sig.params << sig.result << sig.throws;
  }

  template<typename T>
  void fieldsof(T& target, Function& func)
  {
    target << func.name << func.signature << func.body;
  }

  template<typename T>
  void fieldsof(T& target, Method& meth)
  {
    target << meth.name << meth.signature << meth.body;
  }

  template<typename T>
  void fieldsof(T& target, TypeOp& to)
  {
    target << to.types;
  }

  template<typename T>
  void fieldsof(T& target, TypePair& tp)
  {
    target << tp.left << tp.right;
  }

  template<typename T>
  void fieldsof(T& target, TypeName& tn)
  {
    target << tn.value << tn.typeargs;
  }

  template<typename T>
  void fieldsof(T& target, TypeRef& tr)
  {
    target << tr.typenames;
  }

  template<typename T>
  void fieldsof(T& target, Tuple& tuple)
  {
    target << tuple.seq << tuple.type;
  }

  template<typename T>
  void fieldsof(T& target, Block& block)
  {
    target << block.seq;
  }

  template<typename T>
  void fieldsof(T& target, When& when)
  {
    target << when.waitfor << when.behaviour;
  }

  template<typename T>
  void fieldsof(T& target, While& wh)
  {
    target << wh.cond << wh.body;
  }

  template<typename T>
  void fieldsof(T& target, Case& c)
  {
    target << c.pattern << c.guard << c.body;
  }

  template<typename T>
  void fieldsof(T& target, Match& match)
  {
    target << match.cond << match.cases;
  }

  template<typename T>
  void fieldsof(T& target, If& cond)
  {
    target << cond.cond << cond.on_true << cond.on_false;
  }

  template<typename T>
  void fieldsof(T& target, Lambda& lambda)
  {
    target << lambda.signature << lambda.body;
  }

  template<typename T>
  void fieldsof(T& target, Return& ret)
  {
    target << ret.expr;
  }

  template<typename T>
  void fieldsof(T& target, Assign& assign)
  {
    target << assign.left << assign.right;
  }

  template<typename T>
  void fieldsof(T& target, Infix& infix)
  {
    target << infix.op << infix.left << infix.right;
  }

  template<typename T>
  void fieldsof(T& target, Prefix& prefix)
  {
    target << prefix.op << prefix.expr;
  }

  template<typename T>
  void fieldsof(T& target, Select& select)
  {
    target << select.expr << select.member;
  }

  template<typename T>
  void fieldsof(T& target, Specialise& spec)
  {
    target << spec.expr << spec.typeargs;
  }

  template<typename T>
  void fieldsof(T& target, Apply& apply)
  {
    target << apply.expr << apply.args;
  }

  template<typename T>
  void fieldsof(T& target, Ref& ref)
  {
    target << ref.location << ref.type;
  }

  template<typename T>
  void fieldsof(T& target, SymRef& sym)
  {
    target << sym.location;
  }

  template<typename T>
  void fieldsof(T& target, StaticRef& sr)
  {
    target << sr.path << sr.ref;
  }

  template<typename T>
  void fieldsof(T& target, Let& let)
  {
    target << let.decl;
  }

  template<typename T>
  void fieldsof(T& target, Constant& constant)
  {
    target << constant.value;
  }

  template<typename T>
  void fieldsof(T& target, New& n)
  {
    target << n.args << n.in;
  }

  template<typename T>
  void fieldsof(T& target, ObjectLiteral& obj)
  {
    target << obj.inherits << obj.members << obj.in;
  }

  template<typename T>
  struct fields
  {
    T& arg;
    fields(T& arg) : arg(arg) {}
  };

  template<typename Out, typename T>
  Out& operator<<(Out& out, const fields<T>& f)
  {
    fieldsof(out, f.arg);
    return out;
  }
}
