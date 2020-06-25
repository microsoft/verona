// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "compiler/dataflow/work_set.h"

/**
 * This file implements a generic framework for backwards dataflow analysis.
 */
namespace verona::compiler
{
  template<typename State>
  struct DataflowResults
  {
    template<typename T, typename S, typename R>
    friend class DataflowAnalysis;

    const State& state_in(const BasicBlock* bb) const
    {
      return states_in_.at(bb);
    }

  private:
    std::unordered_map<const BasicBlock*, State> states_in_;
  };

  template<typename T, typename State, typename Result = DataflowResults<State>>
  class DataflowAnalysis
  {
  public:
    DataflowAnalysis() : result_(std::make_unique<Result>()) {}

    template<typename... Args>
    static std::unique_ptr<Result> run(Args&&... args, const MethodIR& mir)
    {
      T analysis(std::forward<Args>(args)...);
      for (const auto& ir : mir.function_irs)
        analysis.process(*ir);
      return std::move(analysis.result_);
    }

  protected:
    /**
     * Propagate state from (successor, state_in) backwards to
     * (current, state_out).
     */
    void propagate(
      const BasicBlock* successor,
      const State& state_in,
      const BasicBlock* current,
      State& state_out)
    {
      throw std::logic_error("Did not override `propagate`");
    }

    /**
     * Compute the state for the given basic block.
     */
    void compute(const BasicBlock* bb, State& state)
    {
      throw std::logic_error("Did not override `compute`");
    }

    /**
     * Determine whether the state in of a basic block has changed across two
     * iterations.
     */
    bool has_changed(const State& old_state, const State& new_state)
    {
      throw std::logic_error("Did not override `has_changed`");
    }

  private:
    void process(const FunctionIR& ir)
    {
      for (const BasicBlock* bb : ir.exits)
      {
        work_set_.insert(bb);
      }

      while (!work_set_.empty())
      {
        const BasicBlock* bb = work_set_.remove();
        visit_basic_block(bb);
      }
    }

    State out_state(const BasicBlock* bb)
    {
      State state;
      bb->visit_successors([&](const BasicBlock* successor) {
        auto it = result_->states_in_.find(successor);
        if (it != result_->states_in_.end())
        {
          self()->propagate(successor, it->second, bb, state);
        }
      });
      return std::move(state);
    }

    void visit_basic_block(const BasicBlock* bb)
    {
      State state = out_state(bb);

      self()->compute(bb, state);

      auto [it, inserted] =
        result_->states_in_.try_emplace(bb, std::move(state));
      if (!inserted)
      {
        if (self()->has_changed(it->second, state))
        {
          it->second = std::move(state);
          inserted = true;
        }
      }

      if (inserted)
      {
        for (const BasicBlock* predecessor : bb->predecessors)
        {
          work_set_.insert(predecessor);
        }
      }
    }

    T* self()
    {
      return static_cast<T*>(this);
    }

  protected:
    Result& result()
    {
      return *result_;
    }

  private:
    WorkSet<const BasicBlock*> work_set_;
    std::unique_ptr<Result> result_;
  };
};
