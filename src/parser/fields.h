// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "ast.h"

namespace verona::parser
{
  // This header allows the << operator to be used over AST types, with the
  // result being that the fields of the AST node are forwarded to the <<
  // operator over the target. This is not for Verona fields, but for AST
  // fields.

  template<typename T>
  void fieldsof(T& target, NodeDef& node)
  {
    // Used for nodes with no fields.
  }

  template<typename T>
  void fieldsof(T& target, Using& use)
  {
    target << use.type;
  }

  template<typename T>
  void fieldsof(T& target, TypeAlias& alias)
  {
    target << alias.id << alias.typeparams << alias.type;
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
    target << field.location << field.type << field.init;
  }

  template<typename T>
  void fieldsof(T& target, Param& param)
  {
    target << param.location << param.type << param.init;
  }

  template<typename T>
  void fieldsof(T& target, TypeParam& tp)
  {
    target << tp.location << tp.type << tp.init;
  }

  template<typename T>
  void fieldsof(T& target, Function& func)
  {
    target << func.name << func.typeparams << func.params << func.result
           << func.body;
  }

  template<typename T>
  void fieldsof(T& target, ThrowType& tt)
  {
    target << tt.type;
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
    target << tn.location << tn.typeargs;
  }

  template<typename T>
  void fieldsof(T& target, TypeRef& tr)
  {
    target << tr.typenames;
  }

  template<typename T>
  void fieldsof(T& target, TypeList& tl)
  {
    target << tl.location;
  }

  template<typename T>
  void fieldsof(T& target, Oftype& oftype)
  {
    target << oftype.expr << oftype.type;
  }

  template<typename T>
  void fieldsof(T& target, Tuple& tuple)
  {
    target << tuple.seq;
  }

  template<typename T>
  void fieldsof(T& target, When& when)
  {
    target << when.waitfor << when.behaviour;
  }

  template<typename T>
  void fieldsof(T& target, Try& tr)
  {
    target << tr.body << tr.catches;
  }

  template<typename T>
  void fieldsof(T& target, Match& match)
  {
    target << match.test << match.cases;
  }

  template<typename T>
  void fieldsof(T& target, Lambda& lambda)
  {
    target << lambda.typeparams << lambda.params << lambda.body;
  }

  template<typename T>
  void fieldsof(T& target, Assign& assign)
  {
    target << assign.left << assign.right;
  }

  template<typename T>
  void fieldsof(T& target, Select& select)
  {
    target << select.expr << select.typenames << select.args;
  }

  template<typename T>
  void fieldsof(T& target, Ref& ref)
  {
    target << ref.location;
  }

  template<typename T>
  void fieldsof(T& target, Let& let)
  {
    target << let.location;
  }

  template<typename T>
  void fieldsof(T& target, Throw& thr)
  {
    target << thr.expr;
  }

  template<typename T>
  void fieldsof(T& target, New& n)
  {
    target << n.in << n.args;
  }

  template<typename T>
  void fieldsof(T& target, ObjectLiteral& obj)
  {
    target << obj.in << obj.inherits << obj.members;
  }

  template<typename T>
  void fieldsof(T& target, Constant& constant)
  {
    target << constant.location;
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
