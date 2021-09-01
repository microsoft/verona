// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "interpreter/vm.h"

namespace verona::interpreter
{
  using bytecode::Register;
  using bytecode::RegisterSpan;

  /**
   * Adaptor around a RegisterSpan, providing iterators that yield a Value for
   * each register.
   *
   * It comes in two variants, ConstValueList and ValueList, depending on
   * whether it yields const references or not.
   */
  template<bool IsConst>
  class BaseValueList
  {
    using vm_pointer = std::conditional_t<IsConst, const VM*, VM*>;
    using value_type = std::conditional_t<IsConst, const Value, Value>;

    template<bool IsReverse>
    struct base_iterator
    {
      value_type& operator*()
      {
        if constexpr (IsReverse)
          return vm_->read(*(register_ - 1));
        else
          return vm_->read(*register_);
      }

      bool operator!=(const base_iterator& other) const
      {
        assert(vm_ == other.vm_);
        return register_ != other.register_;
      }

      base_iterator& operator++()
      {
        if constexpr (IsReverse)
          register_--;
        else
          register_++;
        return *this;
      }

    private:
      explicit base_iterator(vm_pointer vm, const Register* reg)
      : vm_(vm), register_(reg)
      {}
      vm_pointer vm_;
      const Register* register_;

      friend class BaseValueList;
    };

  public:
    BaseValueList(vm_pointer vm, RegisterSpan registers)
    : vm_(vm), registers_(registers)
    {}

    using iterator = base_iterator<false>;
    using reverse_iterator = base_iterator<true>;

    iterator begin() const
    {
      return iterator(vm_, registers_.begin());
    }

    iterator end() const
    {
      return iterator(vm_, registers_.end());
    }

    reverse_iterator rbegin() const
    {
      return reverse_iterator(vm_, registers_.end());
    }

    reverse_iterator rend() const
    {
      return reverse_iterator(vm_, registers_.begin());
    }

  private:
    vm_pointer vm_;
    RegisterSpan registers_;
  };

  using ValueList = BaseValueList<false>;
  using ConstValueList = BaseValueList<true>;
}
