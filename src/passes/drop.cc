// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "lang.h"

namespace verona
{
  struct track
  {
    Nodes blocks;
    Nodes stack;
    std::map<Location, bool> params;
    NodeMap<std::map<Location, bool>> lets;
    std::map<Location, std::vector<std::pair<Node, Node>>> refs;
    NodeMap<Node> parents;
    NodeMap<Nodes> children;
    NodeMap<Nodes> successors;
    bool llvm;

    track(bool llvm) : llvm(llvm) {}

    void gen(const Location& loc)
    {
      if (!llvm)
      {
        if (stack.empty())
          params[loc] = true;
        else
          lets[stack.back()][loc] = true;
      }
    }

    void ref(const Location& loc, Node node)
    {
      if (llvm)
        refs[loc].push_back({{}, node});
      else
        refs[loc].push_back({stack.back(), node});
    }

    bool is_successor(Node of, Node block)
    {
      // A successor block is a block that could execute after this one. This
      // is one of the following:
      // * A parent (any distance) block, or
      // * A child (any distance) block, or
      // * A child (any distance) block of a conditional in a parent block,
      // where the conditional follows this block.

      // Check if it's the same block or a child.
      if ((of == block) || is_child(of, block))
        return true;

      // Only check parents and successors if this isn't an early return.
      return (block->back()->type() != Return) &&
        (is_parent(of, block) || is_successor_or_child(of, block));
    }

    bool is_parent(Node of, Node block)
    {
      if (of->parent()->type() == Function)
        return false;

      auto& parent = parents.at(of);
      return (parent == block) || is_successor_or_child(parent, block) ||
        is_parent(parent, block);
    }

    bool is_child(Node of, Node block)
    {
      return std::any_of(
        children[of].begin(), children[of].end(), [&](auto& c) {
          return (c == block) || is_child(c, block);
        });
    }

    bool is_successor_or_child(Node of, Node block)
    {
      return std::any_of(
        successors[of].begin(), successors[of].end(), [&](auto& c) {
          return (c == block) || is_child(c, block);
        });
    }

    void pre_block(Node block)
    {
      if (llvm)
        return;

      if (stack.empty())
      {
        lets[block] = params;
      }
      else
      {
        auto parent = stack.back();

        for (auto& child : children[parent])
        {
          // The new child is a successor of the old children unless it's a
          // sibling block in a conditional.
          if (child->parent() != block->parent())
            successors[child].push_back(block);
        }

        children[parent].push_back(block);
        parents[block] = parent;
        lets[block] = lets[parent];
      }

      stack.push_back(block);
      blocks.push_back(block);
    }

    void post_block()
    {
      if (!llvm)
        stack.pop_back();
    }

    size_t post_function()
    {
      size_t changes = 0;

      if (llvm)
      {
        for (auto& [loc, list] : refs)
        {
          for (auto it = list.begin(); it != list.end(); ++it)
          {
            auto ref = it->second;
            auto parent = ref->parent();
            bool immediate = parent->type() == Block;

            if (immediate && (parent->back() != ref))
              parent->replace(ref);
            else
              parent->replace(ref, Move << (ref / Ident));

            changes++;
          }
        }

        return changes;
      }

      for (auto& [loc, list] : refs)
      {
        for (auto it = list.begin(); it != list.end(); ++it)
        {
          auto refblock = it->first;
          auto ref = it->second;
          auto id = ref / Ident;
          auto parent = ref->parent()->shared_from_this();
          bool immediate = parent->type() == Block;
          bool discharging = true;

          // We're the last use if there is no following use in this or any
          // successor block.
          for (auto next = it + 1; next != list.end(); ++next)
          {
            if (is_successor(refblock, next->first))
            {
              discharging = false;
              break;
            }
          }

          if (discharging && immediate && (parent->back() == ref))
            parent->replace(ref, Move << id);
          else if (discharging && immediate)
            parent->replace(ref, Drop << id);
          else if (discharging)
            parent->replace(ref, Move << id);
          else if (immediate)
            parent->replace(ref);
          else
            parent->replace(ref, Copy << id);

          // If this is a discharging use, mark the variable as discharged in
          // all predecessor and successor blocks.
          if (discharging)
          {
            bool forward = true;

            for (auto& block : blocks)
            {
              if (block == refblock)
                forward = false;

              if (
                forward ? is_successor(block, refblock) :
                          is_successor(refblock, block))
              {
                lets[block][id->location()] = false;
              }
            }
          }

          changes++;
        }
      }

      for (auto& block : blocks)
      {
        auto& let = lets[block];

        for (auto& it : let)
        {
          if (it.second)
          {
            block->insert(block->begin(), Drop << (Ident ^ it.first));
            changes++;
          }
        }
      }

      return changes;
    }
  };

  PassDef drop()
  {
    auto drop_map = std::make_shared<std::vector<track>>();

    PassDef drop = {
      dir::topdown | dir::once,
      {
        (T(Param) / T(Bind)) << T(Ident)[Id] >> ([drop_map](Match& _) -> Node {
          drop_map->back().gen(_(Id)->location());
          return NoChange;
        }),

        T(RefLet)[RefLet] << T(Ident)[Id] >> ([drop_map](Match& _) -> Node {
          drop_map->back().ref(_(Id)->location(), _(RefLet));
          return NoChange;
        }),

        T(LLVM) >> ([drop_map](Match&) -> Node {
          drop_map->back().llvm = true;
          return NoChange;
        }),

        T(Call) << (T(FunctionName)[Op] * T(Args)[Args]) >>
          ([drop_map](Match& _) -> Node {
            if (is_llvm_call(_(Op), _(Args)->size()))
              drop_map->back().llvm = true;

            return NoChange;
          }),
      }};

    drop.pre(Block, [drop_map](Node node) {
      drop_map->back().pre_block(node);
      return 0;
    });

    drop.post(Block, [drop_map](Node) {
      drop_map->back().post_block();
      return 0;
    });

    drop.pre(Function, [drop_map](Node f) {
      auto llvm = (f / LLVMFuncType)->type() == LLVMFuncType;
      drop_map->push_back(track(llvm));
      return 0;
    });

    drop.post(Function, [drop_map](Node) {
      auto changes = drop_map->back().post_function();
      drop_map->pop_back();
      return changes;
    });

    return drop;
  }
}
