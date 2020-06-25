// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "compiler/mapper.h"

/**
 * Recursive types, of the form (μX. T), are represented using De Bruijn
 * indices.
 *
 * The FixpointType node represents μ binders, whereas the FixpointVariableType
 * node represents uses of fixpoint variables.
 *
 * We practically never manipulate open terms, making this similar to locally
 * nameless binders [0]. In particular, unfolding fixpoints is defined in terms
 * of the open operation, described in Figure 1, and closing is described in
 * Figure 3.
 *
 * [0]: Aydemir, Brian, Arthur Charguéraud, Benjamin C. Pierce, Randy Pollack,
 *      Stephanie Weirich, and Stephanie Weirich. "Engineering formal
 *      metatheory."
 *      Proceedings of the 35th Annual ACM SIGPLAN-SIGACT Symposium on
 *      Principles of Programming Languages
 *      https://www.chargueraud.org/research/2007/binders/binders_popl_08.pdf
 */
namespace verona::compiler
{
  /**
   * Expand a fixpoint of the form (μX. T) into T [(μX. T) / X]
   *
   * Given the open operation from [0]:
   *
   *   unfold(μX. T) = open(T, μX. T)
   */
  TypePtr unfold_fixpoint(Context& context, const FixpointTypePtr& fixpoint);

  /**
   * Replace `infer` in `type` by a fixpoint variable. The fixpoint variable
   * will refer to the innermost binder wrapping the result.
   */
  TypePtr close_fixpoint(
    Context& context, const InferTypePtr& infer, const TypePtr& type);

  TypePtr shift_fixpoint(
    Context& context,
    const TypePtr& type,
    ptrdiff_t distance,
    size_t cutoff = 0);
}
