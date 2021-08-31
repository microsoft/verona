// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "compiler/typecheck/permission_check.h"

#include "compiler/printing.h"
#include "compiler/typecheck/capability_predicate.h"

#include <fmt/ostream.h>

namespace verona::compiler
{
  class CheckPermissions
  {
  public:
    CheckPermissions(Context& context, const TypecheckResults& typecheck)
    : context_(context), typecheck_(typecheck)
    {}

    void process(const MethodIR& mir)
    {
      for (auto& ir : mir.function_irs)
      {
        IRTraversal traversal(*ir);
        while (BasicBlock* bb = traversal.next())
        {
          const TypeAssignment& assignment = typecheck_.types.at(bb);
          for (const auto& stmt : bb->statements)
          {
            std::visit([&](const auto& s) { visit_stmt(assignment, s); }, stmt);
          }
        }
      }
    }

  private:
    void visit_stmt(const TypeAssignment& assignment, const CallStmt& stmt) {}

    void
    visit_stmt(const TypeAssignment& assignment, const StaticTypeStmt& stmt)
    {}

    void visit_stmt(const TypeAssignment& assignment, const NewStmt& stmt) {}

    void visit_stmt(const TypeAssignment& assignment, const MatchBindStmt& stmt)
    {}

    void visit_stmt(const TypeAssignment& assignment, const ReadFieldStmt& stmt)
    {
      require_readable(assignment, stmt, stmt.base);
    }

    void
    visit_stmt(const TypeAssignment& assignment, const WriteFieldStmt& stmt)
    {
      require_writable(assignment, stmt, stmt.base);
    }

    void visit_stmt(const TypeAssignment& assignment, const ViewStmt& stmt)
    {
      require_writable(assignment, stmt, stmt.input);
    }

    void visit_stmt(const TypeAssignment& assignment, const WhenStmt& stmt)
    {
      for (auto& v : stmt.captures)
      {
        const TypePtr& base_type = assignment.at(v);
        PredicateSet predicates = predicates_for_type(base_type);
        if (!predicates.contains(CapabilityPredicate::Sendable))
        {
          report(
            context_,
            stmt.source_range,
            DiagnosticKind::Error,
            Diagnostic::TypeNotSendableForWhen,
            *base_type,
            v.variable);
        }
      }
    }

    void visit_stmt(const TypeAssignment& assignment, const CopyStmt& stmt) {}

    void
    visit_stmt(const TypeAssignment& assignment, const IntegerLiteralStmt& stmt)
    {}

    void
    visit_stmt(const TypeAssignment& assignment, const StringLiteralStmt& stmt)
    {}

    void visit_stmt(const TypeAssignment& assignment, const UnitStmt& stmt) {}

    void visit_stmt(const TypeAssignment& assignment, const EndScopeStmt& stmt)
    {}

    void visit_stmt(const TypeAssignment& assignment, const OverwriteStmt& stmt)
    {}

    void require_readable(
      const TypeAssignment& assignment,
      const BaseStatement& stmt,
      Variable variable)
    {
      const TypePtr& base_type = assignment.at(variable);
      PredicateSet predicates = predicates_for_type(base_type);

      if (!predicates.contains(CapabilityPredicate::Readable))
      {
        report(
          context_,
          stmt.source_range,
          DiagnosticKind::Error,
          Diagnostic::TypeNotReadable,
          *base_type);
      }
    }

    void require_writable(
      const TypeAssignment& assignment,
      const BaseStatement& stmt,
      Variable variable)
    {
      const TypePtr& base_type = assignment.at(variable);
      PredicateSet predicates = predicates_for_type(base_type);

      if (!predicates.contains(CapabilityPredicate::Writable))
      {
        report(
          context_,
          stmt.source_range,
          DiagnosticKind::Error,
          Diagnostic::TypeNotWritable,
          *base_type);
      }
    }

  private:
    Context& context_;
    const TypecheckResults& typecheck_;
  };

  bool check_permissions(
    Context& context, const MethodIR& mir, const TypecheckResults& typecheck)
  {
    CheckPermissions(context, typecheck).process(mir);
    return !context.have_errors_occurred();
  }
}
