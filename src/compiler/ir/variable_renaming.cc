// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "compiler/ir/variable_renaming.h"

#include "compiler/format.h"
#include "compiler/ir/ir.h"
#include "compiler/zip.h"
#include "ds/helpers.h"

#include <cassert>

namespace verona::compiler
{
  VariableRenaming VariableRenaming::identity()
  {
    return VariableRenaming({}, nullptr, nullptr);
  }

  /* static */
  VariableRenaming VariableRenaming::compute(
    const BasicBlock* from, const BasicBlock* to, Direction direction)
  {
    std::map<Variable, Variable> mapping;
    if (!to->phi_nodes.empty())
    {
      // The IR guarantees that if `to` has phi_nodes, `from` ends with a
      // branch terminator.
      const BranchTerminator& terminator = from->branch_terminator();
      assert(terminator.target == to);

      for (auto [input, output] :
           safe_zip(terminator.phi_arguments, to->phi_nodes))
      {
        bool inserted;
        switch (direction)
        {
          case Direction::Forwards:
            inserted = mapping.insert({input, output}).second;
            break;
          case Direction::Backwards:
            inserted = mapping.insert({output, input}).second;
            break;

            EXHAUSTIVE_SWITCH;
        }

        if (!inserted)
          throw std::logic_error("SSA variable mapped multiple times");
      }
    }

    switch (direction)
    {
      case Direction::Forwards:
        return VariableRenaming(mapping, from, to);

      case Direction::Backwards:
        return VariableRenaming(mapping, to, from);

        EXHAUSTIVE_SWITCH;
    }
  }

  /* static */
  VariableRenaming
  VariableRenaming::forwards(const BasicBlock* from, const BasicBlock* to)
  {
    return compute(from, to, Direction::Forwards);
  }

  /* static */
  VariableRenaming
  VariableRenaming::backwards(const BasicBlock* from, const BasicBlock* to)
  {
    return compute(from, to, Direction::Backwards);
  }

  Variable VariableRenaming::apply(Variable variable) const
  {
    if (auto it = mapping_.find(variable); it != mapping_.end())
      return it->second;
    else
      return variable;
  }

  VariableRenaming VariableRenaming::invert() const
  {
    std::map<Variable, Variable> inverse;
    for (const auto& [from, to] : mapping_)
    {
      if (!inverse.insert({to, from}).second)
      {
        fmt::print(std::cerr, "invert([{}])\n", *this);
        throw std::logic_error(
          "SSA variable mapped multiple times when inverting");
      }
    }

    return VariableRenaming(inverse, range_, domain_);
  }

  VariableRenaming
  VariableRenaming::compose(const VariableRenaming& other) const
  {
    if (domain_ == nullptr && range_ == nullptr)
      return other;
    else if (other.domain_ == nullptr && other.range_ == nullptr)
      return *this;

    if (domain_ != other.range_)
    {
      fmt::print(
        std::cerr,
        "Mismatch in range and domain of VariableRenaming composition:"
        " ({} -> {}) ∘ ({} -> {})\n",
        *domain_,
        *range_,
        *other.domain_,
        *other.range_);
      abort();
    }

    VariableRenaming other_inverse = other.invert();

    std::map<Variable, Variable> result;
    for (const auto& [from, to] : other.mapping_)
    {
      Variable output = this->apply(to);
      if (from != output)
        result.insert({from, output});
    }

    for (const auto& [from, to] : this->mapping_)
    {
      if (other_inverse.mapping_.find(from) == other_inverse.mapping_.end())
      {
        if (!result.insert({from, to}).second)
          throw std::logic_error(
            "SSA variable mapped multiple times when composing");
      }
    }

    return VariableRenaming(result, other.domain_, range_);
  }

  VariableRenaming VariableRenaming::filter(
    std::function<bool(Variable, Variable)> predicate) const
  {
    std::map<Variable, Variable> compressed_mapping;
    for (auto [from, to] : mapping_)
    {
      if (predicate(from, to))
      {
        compressed_mapping.insert({from, to});
      }
    }
    return VariableRenaming(compressed_mapping, domain_, range_);
  }

  bool VariableRenaming::operator<(const VariableRenaming& other) const
  {
    return std::tie(mapping_, domain_, range_) <
      std::tie(other.mapping_, other.domain_, other.range_);
  }

  std::ostream& operator<<(std::ostream& out, const VariableRenaming& renaming)
  {
    if (renaming.domain_ == nullptr && renaming.range_ == nullptr)
    {
      fmt::print(out, "(identity-renaming)");
    }
    else
    {
      auto entry_formatter = [&](const auto& entry) {
        return fmt::format("{} -> {}", entry.first, entry.second);
      };
      fmt::print(
        out,
        "({} -> {}: [{}])",
        *renaming.domain_,
        *renaming.range_,
        format::comma_sep(renaming.mapping_, entry_formatter));
    }
    return out;
  }
}
