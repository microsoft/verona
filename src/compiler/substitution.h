// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "compiler/ast.h"
#include "compiler/freevars.h"
#include "compiler/mapper.h"
#include "compiler/printing.h"

#include <map>

namespace verona::compiler
{
  class Substitution
  {
  public:
    Substitution() {}
    Substitution(InferTypePtr infer, TypePtr ty)
    {
      insert(infer, ty);
    }
    Substitution(UnboundedTypeSequence unbounded, BoundedTypeSequence bounded)
    {
      insert(unbounded, bounded);
    }

    bool is_trivial() const
    {
      return types_.empty() && sequences_.empty();
    }

    void insert(InferTypePtr infer, TypePtr ty)
    {
      bool inserted = types_.insert({infer, ty}).second;
      if (!inserted)
      {
        std::cerr << "infer variable already in substitution: " << *infer
                  << std::endl;
        abort();
      }
    }

    void insert(UnboundedTypeSequence unbounded, BoundedTypeSequence bounded)
    {
      bool inserted = sequences_.insert({unbounded, bounded}).second;
      if (!inserted)
      {
        std::cerr << "unbounded sequence already in substitution" << std::endl;
        abort();
      }
    }

    const std::map<InferTypePtr, TypePtr>& types() const
    {
      return types_;
    }

    const std::map<UnboundedTypeSequence, InferableTypeSequence>&
    sequences() const
    {
      return sequences_;
    }

    bool operator<(const Substitution& other) const
    {
      return std::tie(types_, sequences_) <
        std::tie(other.types_, other.sequences_);
    }

    template<typename T>
    auto apply(Context& context, const T& value) const
    {
      Applier v(context, *this);
      return v.apply(value);
    }

    void apply_to(Context& context, Substitution* rhs) const
    {
      Applier v(context, *this);
      v.apply_to(rhs);
    }

    void print(std::ostream& s) const
    {
      for (auto it : types())
      {
        s << " " << *it.first << " --> " << *it.second << std::endl;
      }
    }

  private:
    std::map<InferTypePtr, TypePtr> types_;
    std::map<UnboundedTypeSequence, InferableTypeSequence> sequences_;

    class Applier : public RecursiveTypeMapper
    {
    public:
      explicit Applier(Context& context, const Substitution& substitution)
      : RecursiveTypeMapper(context), substitution_(substitution)
      {}

      void apply_to(Substitution* rhs)
      {
        for (auto& right : rhs->types_)
        {
          right.second = apply(right.second);
        }
        for (const auto& left : substitution_.types_)
        {
          rhs->types_.insert(left);
        }
        for (auto& right : rhs->sequences_)
        {
          right.second = apply(right.second);
        }
        for (const auto& left : substitution_.sequences_)
        {
          rhs->sequences_.insert(left);
        }
      }

      TypePtr visit_infer(const InferTypePtr& ty) final
      {
        auto it = substitution_.types_.find(ty);
        if (it != substitution_.types_.end())
        {
          // Substitution doesn't support replacing inference variables with
          // open terms. If we did want to support that, we would have to shift
          // up the substitution as we enter binders.
          assert(context().free_variables(it->second).is_fixpoint_closed());
          return it->second;
        }
        else
        {
          return ty;
        }
      }

      InferableTypeSequence
      visit_sequence(const UnboundedTypeSequence& sequence) final
      {
        auto it = substitution_.sequences_.find(sequence);
        if (it != substitution_.sequences_.end())
          return it->second;
        else
          return sequence;
      }

    private:
      /**
       * Check whether the Substitution modifies the type at all.
       *
       *  This is done by intersecting the free-variables of the type with the
       *  set of variables modified by the substitution.
       *
       *  Free-variables are cached, meaning we can do this without recursing
       *  into the type structure.
       */
      bool modifies_type(const TypePtr& ty) const final
      {
        const std::map<InferTypePtr, TypePtr>& mapping = substitution_.types_;
        const FreeVariables& freevars = context().free_variables(ty);

        return overlaps(freevars.inference, substitution_.types_) ||
          overlaps(freevars.sequences, substitution_.sequences_);
      }

      /**
       * Check whether a set overlaps with the domain of a map.
       *
       * The implementation takes advantage of the fact that values are ordered,
       * making this linear in time.
       */
      template<typename T, typename U>
      static bool overlaps(const std::set<T>& left, const std::map<T, U>& right)
      {
        auto left_it = left.begin();
        auto right_it = right.begin();

        while (left_it != left.end() && right_it != right.end())
        {
          if (*left_it < right_it->first)
          {
            left_it++;
          }
          else if (right_it->first < *left_it)
          {
            right_it++;
          }
          else
          {
            return true;
          }
        }

        return false;
      }

      const Substitution& substitution_;
    };
  };
}
