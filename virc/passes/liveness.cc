#include "../analysis/bitset.h"
#include "../lang.h"

#include <queue>
#include <unordered_map>

namespace virc
{
  static void eliminate_moves(Node top)
  {
    for (auto& func : *top)
    {
      if (func != Func)
        continue;

      std::set<std::string> var_names;
      std::unordered_map<std::string, std::string> aliases;
      std::vector<Node> moves;

      for (auto& var : *(func / Vars))
        var_names.insert(std::string((var / LocalId)->location().view()));

      for (auto& label : *(func / Labels))
      {
        for (auto& stmt : *(label / Body))
        {
          if (stmt != Move)
            continue;

          auto dst_name = std::string((stmt / LocalId)->location().view());
          auto src_name = std::string((stmt / Rhs)->location().view());

          if (var_names.count(dst_name) || var_names.count(src_name))
            continue;

          aliases.emplace(std::move(dst_name), std::move(src_name));
          moves.push_back(stmt);
        }
      }

      auto resolve = [&](std::string name) {
        std::vector<std::string> path;
        auto it = aliases.find(name);

        while (it != aliases.end())
        {
          path.push_back(name);
          assert(path.size() <= aliases.size());
          name = it->second;
          it = aliases.find(name);
        }

        for (auto& alias : path)
          aliases.at(alias) = name;

        return name;
      };

      for (auto& [dst, src] : aliases)
        src = resolve(src);

      for (auto& move : moves)
        move->parent()->replace(move);

      std::vector<std::pair<Node, std::string>> uses;

      func->traverse([&](Node& n) {
        if (n == LocalId)
        {
          auto it = aliases.find(std::string(n->location().view()));

          if (it != aliases.end())
            uses.emplace_back(n, it->second);
        }

        return true;
      });

      for (auto& [use, name] : uses)
        use->parent()->replace(use, LocalId ^ name);
    }
  }

  PassDef liveness(std::shared_ptr<Compilation> state)
  {
    PassDef p{"liveness", wfIR, dir::topdown | dir::once, {}};

    p.post([state](auto top) {
      FuncState* func = nullptr;
      LabelState* label = nullptr;
      Bitset vars;

      auto def = [&](Node& id) {
        auto r = *func->get_register_id(id);
        std::tuple<bool, std::string> def_ret = label->def(r, id, vars.test(r));

        if (!std::get<0>(def_ret))
        {
          state->error = true;
          id->parent()->replace(id, err(clone(id), std::get<1>(def_ret)));
        }
      };

      auto use = [&](Node& id) {
        if (!label->use(*func->get_register_id(id), id))
        {
          state->error = true;
          id->parent()->replace(id, err(clone(id), "use of dead register"));
        }
      };

      auto kill = [&](Node& id) {
        if (!label->kill(*func->get_register_id(id)))
        {
          state->error = true;
          id->parent()->replace(id, err(clone(id), "use of dead register"));
        }
      };

      top->traverse(
        [&](auto node) {
          if (node == Func)
          {
            func = &state->get_func(node / FunctionId);
            label = nullptr;
            vars.resize(func->register_names.size());

            for (auto var : *(node / Vars))
              vars.set(*func->get_register_id(var / LocalId));
          }
          else if (node == Label)
          {
            label = &func->get_label(node / LabelId);
          }
          else if (node == Move)
          {
            kill(node / Rhs);
            def(node / LocalId);
          }
          else if (node->in({Drop, AtTeardown}))
          {
            kill(node / LocalId);
          }
          else if (node == Freeze)
          {
            kill(node / Rhs);
            def(node / LocalId);
          }
          else if (node->in({HeapArray, Add, Sub, Mul, Div,     Mod,  Pow, And,
                             Or,        Xor, Shl, Shr, Eq,      Ne,   Lt,  Le,
                             Gt,        Ge,  Min, Max, LogBase, Atan2}))
          {
            use(node / Lhs);
            use(node / Rhs);
            def(node / LocalId);
          }
          else if (node->in(
                     {Convert,
                      Copy,
                      ArrayRef,
                      Load,
                      Store,
                      Lookup,
                      Neg,
                      Not,
                      Abs,
                      Ceil,
                      Floor,
                      Exp,
                      Log,
                      Sqrt,
                      Cbrt,
                      IsInf,
                      IsNaN,
                      Sin,
                      Cos,
                      Tan,
                      Asin,
                      Acos,
                      Atan,
                      Sinh,
                      Cosh,
                      Tanh,
                      Asinh,
                      Acosh,
                      Atanh,
                      Bits,
                      Len,
                      MakePtr,
                      Read,
                      SetRaise,
                      Typetest,
                      MakeCallback,
                      CodePtrCallback,
                      FreeCallback,
                      Pin,
                      Unpin}))
          {
            use(node / Rhs);
            def(node / LocalId);
          }
          else if (node == HeapArrayConst)
          {
            use(node / Lhs);
            def(node / LocalId);
          }
          else if (node == Merge)
          {
            use(node / Lhs);
            use(node / Rhs);
            def(node / LocalId);
          }
          else if (node == FFILoad)
          {
            use(node / Lhs);
            use(node / Rhs);
            use(node / Kind);
            def(node / LocalId);
          }
          else if (node == FFIStore)
          {
            use(node / Lhs);
            use(node / Rhs);
            use(node / Kind);
            use(node / ValueSrc);
            def(node / LocalId);
          }
          else if (node == RegisterRef)
          {
            // RegisterRef must target a var register, not an SSA register.
            auto src_r = func->get_register_id(node / Rhs);

            if (src_r && !vars.test(*src_r))
            {
              state->error = true;
              node->parent()->replace(
                node, err(node / Rhs, "RegisterRef targets non-var register"));
              return true;
            }

            def(node / LocalId);
          }
          else if (node->in({NewArray, StackArray, RegionArray}))
          {
            use(node / Rhs);
            def(node / LocalId);
          }
          else if (node->in({Const,
                             ConstStr,
                             NewArrayConst,
                             StackArrayConst,
                             RegionArrayConst,
                             FieldRef,
                             ArrayRefConst,
                             FFI,
                             FFIStruct,
                             GetRaise,
                             Const_E,
                             Const_Pi,
                             Const_Inf,
                             Const_NaN,
                             AddExternal,
                             RemoveExternal,
                             MemoSlot,
                             ArrayCopy,
                             ArrayFill,
                             ArrayCompare}))
          {
            def(node / LocalId);
          }
          else if (node->in({Return, Raise, TailcallDyn}))
          {
            kill(node / LocalId);
          }
          else if (node == Arg)
          {
            if (node / Type == ArgCopy)
              use(node / Rhs);
            else
              kill(node / Rhs);
          }
          else if (node == MoveArg)
          {
            kill(node / Rhs);
          }
          else if (node == Jump)
          {
            // pred/succ edges are built by assign_ids.
          }
          else if (node == Cond)
          {
            use(node / LocalId);
          }
          else if (node == Error)
          {
            return false;
          }

          return true;
        },
        [&](auto node) {
          // Handle these in post, because the arguments will be pushed first.
          if (node->in({Heap, CallDyn, TryCallDyn, WhenDyn}))
          {
            use(node / Rhs);
            def(node / LocalId);
          }
          else if (node->in({New, Stack, Region, Call, When}))
          {
            def(node / LocalId);
          }
        });

      top->traverse([&](auto node) {
        if (node == Func)
        {
          auto target = (node / Labels)->front() / LabelId;
          auto& func_state = state->get_func(node / FunctionId);
          auto vars = Bitset(func_state.register_names.size());

          for (auto var : *(node / Vars))
            vars.set(*func_state.get_register_id(var / LocalId));

          // Backward data-flow.
          std::queue<size_t> wl;
          for (size_t i = 0; i < func_state.labels.size(); i++)
            wl.push(i);

          while (!wl.empty())
          {
            auto id = wl.front();
            wl.pop();

            // Calculate a new out-set that is everything our successors need.
            // Calculate a new defd set that is everything our successors
            // define.
            auto& l = func_state.labels.at(id);
            auto new_out = l.out;
            auto new_defd = Bitset(func_state.register_names.size());

            for (auto succ_id : l.succ)
            {
              auto& succ = func_state.labels.at(succ_id);
              new_out |= succ.in;
              new_defd |= succ.defd;

              auto redef = l.defd & succ.defd & ~vars;

              for (size_t r = 0; r < func_state.register_names.size(); r++)
              {
                if (succ.defd.test(r) && !l.first_def.at(r))
                  l.first_def[r] = succ.first_def.at(r);

                if (succ.first_use[r])
                  l.first_use[r] = succ.first_use.at(r);
              }

              if (redef)
              {
                for (size_t r = 0; r < func_state.register_names.size(); r++)
                {
                  if (
                    redef.test(r) &&
                    (l.first_def.at(r) != succ.first_def.at(r)))
                  {
                    state->error = true;
                    node << err(
                      clone(succ.first_def.at(r)),
                      "label redefines non-variable register");
                    return true;
                  }
                }
              }
            }

            if (new_out & l.dead)
            {
              for (auto succ_id : l.succ)
              {
                auto& succ = func_state.labels.at(succ_id);

                for (size_t r = 0; r < func_state.register_names.size(); r++)
                {
                  if (succ.in.test(r) && l.dead.test(r))
                  {
                    state->error = true;
                    node << err(
                      clone(succ.first_use.at(r)),
                      "label requires dead register");
                    return true;
                  }
                }
              }
            }

            // Keep new_defd.
            l.defd |= new_defd;

            // Calculate a new in-set that is our out-set, minus our own
            // out-set, plus our own in-set.
            auto new_in = new_out;
            new_in &= ~l.out;
            new_in |= l.in;

            // If the new in-set is different from the old one, keep both the
            // new in-set and the new out-set, and add all predecessors to the
            // worklist.
            if (new_in != l.in)
            {
              l.in = new_in;
              l.out = new_out;

              for (auto pred_id : l.pred)
                wl.push(pred_id);
            }
          }

          // Check that everything is defined.
          auto& label = func_state.get_label(target);
          auto params = Bitset(func_state.register_names.size());

          for (auto param : *(node / Params))
            params.set(*func_state.get_register_id(param / LocalId));

          // undef_registers is the set of registers that are still required
          // that are not paramters
          auto undef_registers = label.in & ~params;
          if (undef_registers)
          {
            size_t first_undef = 0;
            while (first_undef < func_state.register_names.size() &&
                   !undef_registers.test(first_undef))
            {
              first_undef++;
            }
            state->error = true;
            node << err(
              clone(label.first_use.at(first_undef)),
              "use of undefined register");
            return true;
          }

          // Check for multiple assignment to non-variable params.
          auto bad_assign = params & label.defd & ~vars;

          if (bad_assign)
          {
            for (size_t r = 0; r < func_state.register_names.size(); r++)
            {
              if (bad_assign.test(r))
              {
                state->error = true;
                node << err(
                  clone(label.first_use.at(r)),
                  "label writes to non-variable parameter");
                return true;
              }
            }
          }

          for (auto label_node : *(node / Labels))
          {
            auto& l = func_state.get_label(label_node / LabelId);
            auto unneeded = Bitset(func_state.register_names.size());
            Node body = label_node / Body;
            std::vector<std::pair<Node, Node>> drops;

            for (auto succ_idx : l.succ)
              unneeded |= func_state.labels.at(succ_idx).in;

            unneeded = l.out & ~unneeded;

            for (size_t r = 0; r < func_state.register_names.size(); r++)
            {
              if (unneeded.test(r))
              {
                auto last = l.last_use.at(r);

                if (!last)
                {
                  if (!l.used.test(r) && !vars.test(r))
                  {
                    // Dead def: defined but never used or killed.
                    // Find the defining statement in this label's body
                    // and insert a drop after it.
                    for (auto it = body->begin(); it != body->end(); ++it)
                    {
                      auto& s = *it;

                      if (s->empty() || (s->front() != LocalId))
                        continue;

                      auto dst = func_state.get_register_id(s->front());

                      if (dst && (*dst == r))
                      {
                        body->insert(std::next(it), Drop << clone(s->front()));
                        break;
                      }
                    }
                  }

                  continue;
                }

                auto parent = last->parent();

                if ((parent == Arg) && (parent->front() == ArgCopy))
                {
                  l.automove(r);
                  continue;
                }

                if (parent == Copy)
                {
                  l.automove(r);
                  continue;
                }

                Node stmt = last;

                while (stmt && (stmt->parent() != body))
                  stmt = stmt->parent();

                if (!stmt)
                  continue;

                drops.emplace_back(stmt, Drop << clone(last));
                l.last_use[r] = {};
              }
            }

            for (auto& [stmt, drop] : drops)
            {
              auto it = body->find(stmt);
              assert(it != body->end());
              body->insert(std::next(it), drop);
            }
          }
        }
        else if (node == Error)
        {
          return false;
        }

        return true;
      });

      if (!state->error)
        eliminate_moves(top);

      return 0;
    });

    return p;
  }
}
