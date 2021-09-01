// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once
#include "compiler/ir/ir.h"

#include <queue>
#include <tuple>

/**
 * Build the SSA IR from a function's AST, as a single pass.
 *
 * We use the algorithm from [0] to construct the SSA and the CFG
 * simultaneously.
 *
 * Each basic block maintains a mapping from symbols to SSA variable.
 * This map is updated when processing local variable assignements.
 *
 * When a symbol is referenced, the current BB's map is used to find a
 * definition within the same BB. If the symbol is not found, i.e. it is not
 * defined in this BB, we search for it in all predecessors and create a phi
 * node to merge the definitions.
 *
 * When processing loops, not all predecessors of the loop header are known,
 * preventing the recursion described above. When the loop header block is
 * created, it is marked as "pending". Searching for a symbol in an "pending"
 * block, first checks if the loop defines the variable somewhere.  If it does
 * not then, we can simply skip to find the definition from before the loop.
 * If it does define the variable, then we create a fresh variable, and then
 * record it as a "pending" phi node.
 *
 * Once the loop is 'finish'ed, all predecessors of the header block are known,
 * enabling us to the unset the pending status. For each pending phi node in the
 * header, we search in the predecessors to find definition, and set up a phi
 * node accordingly.
 *
 * [0] describes optimizations to avoid trivial phi nodes, and claims that the
 * resulting SSA is minimal for programs with reducible control flow.
 * So far, we've only implemented the "Marker Algorithm", from
 * Section 3.3 of [0]. This will avoid trivial nodes in acyclic sections of
 * the program, but not when loops are present.
 *
 * We have adapted the marker algorithm, due to the highly structured nature
 * of input language.  By calculating the assignments in an expression, we
 * can generate minimal & pruned SSA.
 *
 * [0]: "Simple and Efficient Construction of Static Single Assignment Form"
 *      https://pp.info.uni-karlsruhe.de/uploads/publikationen/braun13cc.pdf
 *
 */
namespace verona::compiler
{
  struct FnSignature;
  struct FnBody;

  std::unique_ptr<MethodIR>
  build_ir(const FnSignature& sig, const FnBody& body);
}
