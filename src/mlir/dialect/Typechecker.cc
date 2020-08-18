// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "Typechecker.h"

#include "TypecheckInterface.h"
#include "VeronaTypes.h"

namespace mlir::verona
{
  LogicalResult typecheck(Operation* op)
  {
    auto callback = [](TypecheckInterface innerOp) -> WalkResult {
      // If typecheck fails, WalkResult::interrupt is returned.
      return innerOp.typecheck();
    };

    if (op->walk(callback).wasInterrupted())
      return failure();
    else
      return success();
  }

  void TypecheckerPass::runOnOperation()
  {
    if (failed(typecheck(getOperation())))
    {
      signalPassFailure();
    }

    // Typechecking does not modify the IR, so all analysis are preserved.
    markAllAnalysesPreserved();
  }

  namespace detail
  {
    /// A Rule wraps a callable object with arguments of type `Left` and
    /// `Right`.
    ///
    /// Whenever the Rule is applied to arbitrary Type values, the left-hand
    /// side and right-hand side Type are respectively dyncasted to `Left` and
    /// `Right`.
    ///
    /// If either dyncast fails, the rule fails. Otherwise, the decision is
    /// delegated to the embedded callable object, passing it the dyncasted
    /// values.
    template<typename Left, typename Right, typename F>
    struct Rule
    {
      constexpr Rule(F f) : inner(std::forward<F>(f)) {}
      bool operator()(Type left, Type right)
      {
        auto derivedLeft = left.dyn_cast<Left>();
        auto derivedRight = right.dyn_cast<Right>();
        if (!derivedLeft || !derivedRight)
          return false;

        return inner(derivedLeft, derivedRight);
      }

    private:
      F inner;
    };

    // We use a deduction guide to make `Rule([](T1, T2) { ... })` work, without
    // needing to repeat T1 and T2 (ie. the types the Rule will dyncast to) as
    // template parameters of Rule.
    //
    // However, extracting the types T1 and T2 from the type of the lambda F is
    // not trivial. We achieve it by matching against the type of the operator()
    // method. The matching part works by specializing a rule_traits struct for
    // member methods. The trick was borrowed from MLIR's walk methods in
    // llvm/include/mlir/IR/Visitors.h
    //
    // If the lambda is too generic, eg. `[](auto x, auto y) { ... }` taking the
    // address of the operator() method will fail and so will compilation. This
    // is intentional as we wouldn't know in that case what the target of the
    // dyncast should be.
    template<typename F>
    struct rule_traits;
    template<typename F>
    Rule(F f)
      ->Rule<
        typename rule_traits<decltype(&F::operator())>::Left,
        typename rule_traits<decltype(&F::operator())>::Right,
        F>;

    template<typename C, typename A1, typename A2>
    struct rule_traits<bool (C::*)(A1, A2) const>
    {
      using Left = A1;
      using Right = A2;
    };

    template<typename C, typename A1, typename A2>
    struct rule_traits<bool (C::*)(A1, A2)>
    {
      using Left = A1;
      using Right = A2;
    };

    /// From a set of callable objects, create a single closure which calls each
    /// of them in succession, until one of them works.
    ///
    /// The callables `Fs` should have precise (ie. non-auto) argument types,
    /// such that the deduction guide described earlier works.
    template<typename... Fs>
    static constexpr auto combineRules(Fs... rules)
    {
      return [=](Type lhs, Type rhs) { return (Rule(rules)(lhs, rhs) || ...); };
    }
  }

  /// Set of rules used for subtype-checking. Each rule is checked (in the
  /// order they are laid out) until one of them works.
  ///
  /// `RULES` is a callable object, which takes two Type values and returns a
  /// boolean.
  static constexpr auto RULES = detail::combineRules(
    [](Type left, Type right) { return left == right; },
    [](JoinType left, Type right) {
      return llvm::all_of(left.getElements(), [&](Type element) {
        return isSubtype(element, right);
      });
    },
    [](Type left, MeetType right) {
      return llvm::all_of(right.getElements(), [&](Type element) {
        return isSubtype(left, element);
      });
    },
    [](MeetType left, Type right) {
      return llvm::any_of(left.getElements(), [&](Type element) {
        return isSubtype(element, right);
      });
    },
    [](Type left, JoinType right) {
      return llvm::any_of(right.getElements(), [&](Type element) {
        return isSubtype(left, element);
      });
    });

  bool isSubtype(Type lhs, Type rhs)
  {
    assert(isaVeronaType(lhs));
    assert(isaVeronaType(rhs));

    return RULES(lhs, rhs);
  }

  LogicalResult checkSubtype(Operation* op, Type lhs, Type rhs)
  {
    assert(isaVeronaType(lhs));
    assert(isaVeronaType(rhs));

    Type normalizedLeft = normalizeType(lhs);
    Type normalizedRight = normalizeType(rhs);
    if (!isSubtype(normalizedLeft, normalizedRight))
    {
      InFlightDiagnostic diag = op->emitError()
        << lhs << " is not a subtype of " << rhs;

      // If we did some normalization, attach a note with the normalized type to
      // make debugging easier.
      if (lhs != normalizedLeft || rhs != normalizedRight)
        diag.attachNote() << "using normalized types " << normalizedLeft
                          << " and " << normalizedRight;

      return failure();
    }

    return success();
  }
}
