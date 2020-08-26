// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#pragma once

/**
 * Language ABI and Lowering rules.
 *
 * This is a temporary handler for holding on to ABI specific ideas like how
 * to lower an iterator or a lambda or constants. Due to the current nature
 * of the AST, we need those ideas here. If we improve the AST, or create an
 * improved veneer of ABI related lowering, we should move this logic there
 * and remove these handlers.
 *
 * FIXME: For now, we're using strings to determine the types, but we'll need
 * to plug in a builder/context pair to create them correctly. If we keep the
 * names here in sync with the generator (unstable requirement), it can work for
 * now.
 *
 * The aim is to have objects that can generate structures, functions and other
 * objects in the right place (within the right lexical context, etc.), and
 * teach the generator to create those patterns on demand or to find the right
 * implementation (for ex. user-defined containers and functors).
 */
namespace mlir::verona::ABI
{
  /**
   * Loop iterator for producers expected to conform to the following pattern:
   * $iter = container;
   * while ($iter.has_value())
   * {
   *   item = $iter.apply();
   *   // use `item`
   *   $iter.next();
   * }
   *
   * This is the expected implementation of the following `for` loop:
   * for (item in container)
   * {
   *   // use `item`
   * }
   */
  struct LoopIterator
  {
    /// Iteration handler, using `$` as a compiler-generated symbol
    constexpr static const char* const handler = "$iter";

    /// Iteration value check, to be performed before taking a value.
    struct check
    {
      constexpr static const char* const name = "has_value";
      constexpr static const char* const args[]{handler};
      constexpr static const char* const types[]{"unk"};
      constexpr static const char* const retTy{"bool"};
    };

    /// Iteration value copy, should only be called if `has_value` is true.
    struct apply
    {
      constexpr static const char* const name = "apply";
      constexpr static const char* const args[]{handler};
      constexpr static const char* const types[]{"unk"};
      constexpr static const char* const retTy{"unk"};
    };

    /// Iteration pointer increment, moves on to the next element in the list.
    struct next
    {
      constexpr static const char* const name = "next";
      constexpr static const char* const args[]{handler};
      constexpr static const char* const types[]{"unk"};
      constexpr static const char* const retTy{};
    };
  };
}
