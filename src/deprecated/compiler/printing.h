// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "compiler/ast.h"
#include "compiler/type.h"

#include <ostream>

namespace verona::compiler
{
  struct Constraint;

  std::ostream& operator<<(std::ostream& out, const Program& p);
  std::ostream& operator<<(std::ostream& out, const Entity& entity);
  std::ostream& operator<<(std::ostream& out, const Member& member);

  std::ostream& operator<<(std::ostream& out, const AssertionKind& kind);
  std::ostream& operator<<(std::ostream& out, const StaticAssertion& assertion);

  std::ostream& operator<<(std::ostream& out, const FnSignature& sig);
  std::ostream& operator<<(std::ostream& out, const FnBody& body);
  std::ostream& operator<<(std::ostream& out, const Receiver& r);
  std::ostream& operator<<(std::ostream& out, const FnParameter& param);
  std::ostream& operator<<(std::ostream& out, const Generics& g);
  std::ostream& operator<<(std::ostream& out, const TypeParameterDef& param);
  std::ostream& operator<<(std::ostream& out, const Entity::Kind& k);

  std::ostream& operator<<(std::ostream& out, const WhereClause& clause);
  std::ostream& operator<<(std::ostream& out, const WhereClause::Kind& k);
  std::ostream& operator<<(std::ostream& out, const WhereClauseTerm& t);

  std::ostream& operator<<(std::ostream& out, const Type& ty);
  std::ostream& operator<<(std::ostream& out, const CapabilityKind& c);
  std::ostream& operator<<(std::ostream& out, const ApplyRegionType::Mode& m);
  std::ostream& operator<<(std::ostream& out, const Region& r);
  std::ostream& operator<<(std::ostream& out, const Polarity& p);
  std::ostream& operator<<(std::ostream& out, const NewParent& p);
  std::ostream& operator<<(std::ostream& out, const InferableTypeSequence& seq);
  std::ostream& operator<<(std::ostream& out, const TypeSignature& signature);

  std::ostream& operator<<(std::ostream& out, const Expression& e);
  std::ostream& operator<<(std::ostream& out, const Argument& p);
  std::ostream& operator<<(std::ostream& out, const MatchArm& a);
  std::ostream& operator<<(std::ostream& out, const BinaryOperator& op);
  std::ostream& operator<<(std::ostream& out, const Constraint& p);

  std::ostream& operator<<(std::ostream& out, const TypeExpression& te);

  std::ostream& operator<<(std::ostream& out, const LocalID& l);
}
