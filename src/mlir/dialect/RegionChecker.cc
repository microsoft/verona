// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "dialect/RegionChecker.h"

#include "Query.h"
#include "dialect/TopologicalFacts.h"
#include "dialect/Typechecker.h"
#include "dialect/VeronaOps.h"
#include "dialect/VeronaTypes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/SetVector.h"

/// The set of facts F(b) available in a block b is defined as follows:
///
///   F(b) = local-facts(b)
///        ∪ derived-facts(F(b))
///        ∪ (intersect { rename(p, b, F(p)) | p ∈ predecessors(b) })
///
/// - Local facts are those introduced by operations within the block. For
///   instance, given `%x = verona.copy %y`, the local facts would contain
///   `alias(%x, %y)`. This is easily computed, by walking over the list of
///   operations and invoking custom per-operation logic.
///
/// - Derived facts are inferred from one or more existing facts. For instance,
///   if `alias(%x, %y)` and `alias(%y, %z)` are known (they are local facts for
///   example), then `alias(%x, %z)` is true. These facts are defined by a set
///   of datalog-like rules over other facts. The result is obtained by applying
///   the rules until a (least) fixed point is reached.
///
/// - If a fact is available in all predecessors of b, then it is available in b
///   as well. There are however a few twists:
///
///   + If a fact refers to variables that are used as block arguments in the
///     transition, the appropriate renaming is applied to the fact's contents.
///
///   + If a fact refers to variables that are not defined in the target block
///     (ie. the variable does not dominate the target block), the fact is not
///     propagated.
///
///   + Because of loops, we may arrive in a situation where
///     `F(b) = F(a) ∩ F(b)`, that is a fact is available in `b` only if it is
///     available in `b`. In these circumstances, we consider the fact to be
///     available (assuming it is also available in block a), resulting in a
///     greatest fixed point.
///
///   Propagated facts are computed by iterating over the control-flow graph,
///   using a worklist of blocks to consider. Initially we assume all facts to
///   be true in all blocks. We apply the flow equation for F(b) in a forward
///   analysis style. On every iteration, the set of available facts can only
///   shrink, until the fixed-point is reached.
///
/// Both iterative processes, the datalog evaluation of derived facts and the
/// propagation throughthe CFG, contribute to one another. Any time the CFG
/// propagation takes a step, it fully re-evaluates all derived facts.
/// TODO: look into whether this can be done incrementally instead. Non-trivial
/// since one iterative process grows the size of the relation whereas the
/// other one shrinks it.
///
namespace mlir::verona
{
  struct FactEvaluator;

  struct StableFacts
  {
    SmallVector<Alias, 0> aliases;

    StableFacts() {}
    explicit StableFacts(SmallVector<Alias, 0>&& aliases) : aliases(aliases) {}

    /// Compute the set of facts that are propagated from one block to the
    /// other. This takes into account block parameter induced renamings, and
    /// filters out variables that are undefined in the target block.
    ///
    /// The function F is invoked on every fact that is propagated to the
    /// target.
    template<typename F>
    void getOutgoingFacts(
      const DominanceInfo& dominance,
      Block* origin,
      unsigned successorIndex,
      Block* target,
      F&& f) const;

    /// Returns true is `this` is "smaller" than `other`. This method is used to
    /// decide when a fixed-point is reached.
    bool refines(const StableFacts& other) const
    {
      assert(aliases.size() <= other.aliases.size());
      return aliases.size() < other.aliases.size();
    }
  };

  struct FactEvaluator
  {
    FactEvaluator() : defined(engine), aliases(engine) {}

    void add(Alias fact)
    {
      aliases.add(fact);
    }

    // TODO: we don't currently support evaluating these facts.
    void add(In fact) {}
    void add(From fact) {}

    /// Add facts derived from a basic block's operations.
    ///
    /// For example, if `block` contains a `%x = copy %y` operation, the fact
    /// `alias(%x, %y)` is added to the evaluator. The RegionCheckInterface
    /// implementation of each operation is used to determine what facts to add.
    ///
    void addLocalFacts(Block* block)
    {
      for (Value v : block->getArguments())
      {
        if (isaVeronaType(v.getType()))
          defined.add(Defined(v));
      }
      for (Operation& op : *block)
      {
        for (Value v : op.getResults())
        {
          if (isaVeronaType(v.getType()))
            defined.add(Defined(v));
        }
        if (auto iface = llvm::dyn_cast<RegionCheckInterface>(op))
        {
          iface.add_facts(*this);
        }
      }
    }

    /// Add facts coming from the predecessors of `block`. Only facts available
    /// in all predecessors are added. The `factsMap` is used to look up the
    /// facts of each block.
    void addIncomingFacts(
      const DominanceInfo& dominance,
      const DenseMap<Block*, StableFacts>& factsMap,
      Block* block);

    /// Compute derived facts by iterating over the set of rules until a fixed
    /// point is reached.
    void process()
    {
      while (engine.iterate())
      {
        iterate();
      }
    }

    /// Extract facts from this evaluator into an equivalent StableFacts.
    StableFacts finish() &&
    {
      return StableFacts(std::move(aliases).finish());
    }

  private:
    /// Apply all the fact derivation rules once.
    void iterate()
    {
      // alias(x, x) :-
      //   defined(x).
      aliases.from_map(defined, [](const auto& r) -> Alias {
        return Alias(r.value, r.value);
      });

      // alias(x, y) :-
      //   alias(y, x).
      aliases.from_map(
        aliases, [](const auto& r) -> Alias { return Alias(r.right, r.left); });

      // alias(y, z) :-
      //   alias(x, y),
      //   alias(x, z).
      //
      // This isn't quite the usual formulation of transitivity; you would
      // generally expect `alias(x, z) :- alias(x, y), alias(y, z).`
      // However, because `alias` is symmetric the two are equvalent and the one
      // used here doesn't need joining on the second variable (and a separate
      // index).
      aliases.from_join(
        aliases, aliases, [](const auto& r1, const auto& r2) -> Alias {
          assert(r1.left == r2.left);
          return Alias(r1.right, r2.right);
        });
    }

    /// Compator type made to operate on Value and Type.
    ///
    /// The QueryEngine needs facts to be ordered, but Value and Type do not
    /// provide operator< implementations. We provide our own ordering by using
    /// their opaque pointer representiation, which should be totally ordered.
    struct ValueCmp
    {
      bool operator()(Value left, Value right) const
      {
        return std::less()(
          left.getAsOpaquePointer(), right.getAsOpaquePointer());
      }

      bool operator()(Type left, Type right) const
      {
        return std::less()(
          left.getAsOpaquePointer(), right.getAsOpaquePointer());
      }

      bool operator()(ArrayRef<Type> left, ArrayRef<Type> right) const
      {
        return std::lexicographical_compare(
          left.begin(), left.end(), right.begin(), right.end(), *this);
      }
    };

    QueryEngine engine;
    Relation<Defined, Index<ValueCmp, 0>> defined;
    Relation<Alias, Index<ValueCmp, 0, 1>> aliases;
  };

  /// A BranchMap transforms SSA values along an edge of the control flow graph,
  /// taking into account renamings and undefined values.
  ///
  /// Once constructed, its `withValue` method can be used to determine all
  /// names a value has after the edge is followed. In the following example,
  /// along the edge between ^bb0 and ^bb1, assuming ^bb0 dominates ^bb1, then
  /// `withValue(%x)` would yield values %x, %y and %z. If ^bb0 does not
  /// dominates ^bb1, then `withValue(%x)` would only yield values %y and %z.
  ///
  ///  ^bb0:
  ///   %x = ....
  ///   br ^bb1(%x, %x)
  ///
  ///  ^bb1(%y, %z):
  ///   ...
  ///
  /// This class operates as a "best-effort"; not all terminators provide enough
  /// information about what values are passed on, hence not all target values
  /// may be yielded. This is sound in its application to propagate facts. If a
  /// target variable is missed, the set of available facts will only be smaller
  /// than what it could have been.
  struct BranchMap
  {
    BranchMap(
      const DominanceInfo& dominance,
      Block* origin,
      unsigned successorIndex,
      Block* target)
    : dominance(dominance), target(target)
    {
      assert(origin->getSuccessor(successorIndex) == target);

      Optional<OperandRange> branchOperands;
      if (auto branch = dyn_cast<BranchOpInterface>(origin->getTerminator()))
        branchOperands = branch.getSuccessorOperands(successorIndex);

      if (branchOperands.hasValue())
      {
        ArrayRef<BlockArgument> blockArguments = target->getArguments();

        assert(branchOperands->size() == blockArguments.size());
        for (const auto& [operand, argument] :
             llvm::zip(*branchOperands, blockArguments))
        {
          valueMap[operand].push_back(argument);
        }
      }
    }

    template<typename F>
    void withValue(Value value, F&& f)
    {
      // We propagate the variable un-renamed only if it is defined in the
      // target block, ie. its definition dominates the block.
      //
      // We're interested in knowing whether the variable is defined at the very
      // beginning of the block, before block arguments are even defined, hence
      // the use of proper dominance.
      if (dominance.properlyDominates(value.getParentBlock(), target))
        f(value);

      auto it = valueMap.find(value);
      if (it != valueMap.end())
      {
        for (Value argument : it->second)
        {
          f(argument);
        }
      }
    }

  private:
    /// Map from one value to the list of values that receive it.
    ///
    /// For example given `br ^bb(%x, %x)` where `^bb` is defined as
    /// `^bb(%y, %z)`, then `valueMap(%x) = [%y, %z]`.
    DenseMap<Value, SmallVector<Value, 1>> valueMap;

    const DominanceInfo& dominance;
    Block* target;
  };

  /// Utility class used to take the intersection of facts from many incoming
  /// edges.
  ///
  /// Each fact produced by a predecessor block is added to the intersector. At
  /// the end, any fact which was added as many times to the intersector as
  /// there are predecessors is kept.
  class FactIntersector
  {
  public:
    FactIntersector() {}

    void operator()(const Alias& fact)
    {
      auto [it, inserted] =
        aliases.insert({fact, std::make_pair(0, predecessors)});
      assert(it->second.second <= predecessors);

      if (inserted || it->second.second < predecessors)
      {
        it->second.first += 1;
      }
    }

    void next()
    {
      predecessors++;
    }

    size_t count() const
    {
      return predecessors;
    }

    template<typename F>
    void finish(F f) const
    {
      for (const auto& [fact, entry] : aliases)
      {
        assert(entry.first <= predecessors);
        if (entry.first == predecessors)
          f(fact);
      }
    }

    FactIntersector(const FactIntersector&) = delete;
    FactIntersector& operator=(const FactIntersector&) = delete;

  private:
    size_t predecessors = 0;

    // For each fact, we keep track of 1) the number of predecessors that have
    // added it and 2) the index of the last predecessor that has. The latter
    // allows us to avoid spurious increments if a precessor has multiple copies
    // of a fact.
    DenseMap<Alias, std::pair<size_t, size_t>> aliases;
  };

  template<typename F>
  void StableFacts::getOutgoingFacts(
    const DominanceInfo& dominance,
    Block* origin,
    unsigned successorIndex,
    Block* target,
    F&& f) const
  {
    BranchMap branchMap(dominance, origin, successorIndex, target);

    for (const Alias& r : aliases)
    {
      branchMap.withValue(r.left, [&](Value left) {
        branchMap.withValue(
          r.right, [&](Value right) { f(Alias(left, right)); });
      });
    }
  }

  void FactEvaluator::addIncomingFacts(
    const DominanceInfo& dominance,
    const DenseMap<Block*, StableFacts>& factsMap,
    Block* block)
  {
    // We add all incoming facts by taking the intersection of all the
    // predecessors' outgoing facts.
    FactIntersector intersector;
    for (auto it = block->pred_begin(); it != block->pred_end(); it++)
    {
      // If a predecessor is missing from the factsMap, its set of facts has not
      // been evaluated yet. We assume "all facts" are available in the block,
      // thus we do not need to include it in the intersection.
      auto facts_it = factsMap.find(*it);
      if (facts_it != factsMap.end())
      {
        facts_it->second.getOutgoingFacts(
          dominance, *it, it.getSuccessorIndex(), block, intersector);
        intersector.next();
      }
    }

    // If the block has any predecessors, we must have found at least one for
    // which we have evaluated facts already, to the fact that we process basic
    // blocks top-down.
    //
    // If this weren't true - if the block has predecessors but none of them had
    // been evaluated yet - we would need "all facts" to be available in this
    // block (since they are in all predecessors), which we can't represent.
    assert(block->hasNoPredecessors() || intersector.count() > 0);

    intersector.finish([&](auto fact) { add(fact); });
  }

  void RegionCheckerPass::runOnFunction()
  {
    llvm::SetVector<Block*> worklist;
    DenseMap<Block*, StableFacts> facts;

    DominanceInfo dominance(getOperation());

    worklist.insert(&getOperation().getCallableRegion()->front());
    while (!worklist.empty())
    {
      Block* current = worklist.pop_back_val();
      SmallVector<StableFacts, 2> incoming_facts;

      FactEvaluator evaluator;
      evaluator.addLocalFacts(current);
      evaluator.addIncomingFacts(dominance, facts, current);
      evaluator.process();
      StableFacts result = std::move(evaluator).finish();

      auto [it, inserted] = facts.insert({current, StableFacts()});
      if (inserted || result.refines(it->second))
      {
        it->second = result;
        worklist.insert(current->succ_begin(), current->succ_end());
      }
    }

    // TODO: find a more appropriate way to print the results.
    AsmState state(getOperation());
    for (Block& block : getOperation())
    {
      const StableFacts& data = facts[&block];
      block.printAsOperand(llvm::errs(), state);
      llvm::errs() << "\n";
      for (const auto& it : data.aliases)
      {
        llvm::errs() << "  ";
        it.print(llvm::errs(), state);
        llvm::errs() << "\n";
      }
    }
  }

  void CopyOp::add_facts(FactEvaluator& facts)
  {
    facts.add(Alias(output(), input()));
  }

  void ViewOp::add_facts(FactEvaluator& facts)
  {
    facts.add(Alias(output(), input()));
  }

  void FieldReadOp::add_facts(FactEvaluator& facts)
  {
    facts.add(In(output(), origin(), getFieldType()));
  }

  void FieldWriteOp::add_facts(FactEvaluator& facts)
  {
    facts.add(From(output(), origin(), {}));
  }

#include "dialect/RegionChecker.cpp.inc"
}
