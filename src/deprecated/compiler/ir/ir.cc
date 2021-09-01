// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "compiler/ir/ir.h"

#include "compiler/format.h"
#include "compiler/printing.h"
#include "compiler/typecheck/typecheck.h"
#include "compiler/zip.h"

#include <fmt/ostream.h>

namespace verona::compiler
{
  namespace
  {
    void visit_terminator_successors(
      const IfTerminator& terminator,
      const std::function<void(BasicBlock*)> visitor)
    {
      visitor(terminator.true_target);
      visitor(terminator.false_target);
    }

    void visit_terminator_successors(
      const MatchTerminator& terminator,
      const std::function<void(BasicBlock*)> visitor)
    {
      for (const auto& arm : terminator.arms)
      {
        visitor(arm.target);
      }
    }

    void visit_terminator_successors(
      const BranchTerminator& terminator,
      const std::function<void(BasicBlock*)> visitor)
    {
      visitor(terminator.target);
    }

    void visit_terminator_successors(
      const ReturnTerminator& terminator,
      const std::function<void(BasicBlock*)> visitor)
    {}
  }

  void BasicBlock::visit_successors(
    const std::function<void(BasicBlock*)> visitor) const
  {
    std::visit(
      [&](const auto& term) { visit_terminator_successors(term, visitor); },
      *terminator);
  }

  IRTraversal::IRTraversal(const FunctionIR& ir)
  {
    enqueue_block(ir.entry);
  }

  BasicBlock* IRTraversal::next()
  {
    while (!queue_.empty())
    {
      BasicBlock* bb = queue_.front();
      queue_.pop();

      if (!visited_.insert(bb).second)
        continue;

      if (bb->terminator.has_value())
      {
        bb->visit_successors(
          [&](BasicBlock* successor) { enqueue_block(successor); });
      }
      return bb;
    }

    return nullptr;
  }

  void IRTraversal::enqueue_block(BasicBlock* bb)
  {
    if (visited_.find(bb) == visited_.end())
    {
      queue_.push(bb);
    }
  }

  std::ostream& operator<<(std::ostream& s, const BasicBlock& bb)
  {
    return s << "BB" << bb.index;
  }

  std::ostream& operator<<(std::ostream& s, const Variable& v)
  {
    if (v.lid.has_value())
      return s << v.lid.value();
    return s << v.index;
  }
}
