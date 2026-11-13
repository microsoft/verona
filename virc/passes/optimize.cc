#include "../lang.h"

namespace virc
{
  PassDef optimize(std::shared_ptr<Compilation> state)
  {
    PassDef p{
      "optimize",
      wfIR,
      dir::once,
      {
        // No rewrite rules — all work done in post().
      }};

    p.post([state](auto top) {
      // Counter for generating unique alpha-renamed local names.
      size_t inline_counter = 0;
      // Helper: find a Class definition by ClassId.
      auto find_class = [&](const Node& class_id) -> Node {
        for (auto& child : *top)
        {
          if (
            child == Class &&
            (child / ClassId)->location() == class_id->location())
            return child;
        }
        return {};
      };

      // Helper: find a Primitive definition by type token.
      auto find_primitive = [&](const Node& type_node) -> Node {
        for (auto& child : *top)
        {
          if (child == Primitive && (child / Type)->type() == type_node->type())
            return child;
        }
        return {};
      };

      // Helper: check if a type token is a simple primitive.
      auto is_primitive = [](const Node& t) -> bool {
        return t->type().in(
          {None,
           Bool,
           I8,
           I16,
           I32,
           I64,
           U8,
           U16,
           U32,
           U64,
           ILong,
           ULong,
           ISize,
           USize,
           F32,
           F64,
           Ptr});
      };

      // Helper: find a Func definition by FunctionId.
      auto find_func = [&](const Node& func_id) -> Node {
        for (auto& child : *top)
        {
          if (
            child == Func &&
            (child / FunctionId)->location() == func_id->location())
            return child;
        }
        return {};
      };

      // Helper: functions that capture a raise target are not safe to inline,
      // because moving their body changes which frame a block lambda captures.
      auto captures_raise_target = [](const Node& func) {
        bool found = false;
        func->traverse([&](Node& n) {
          if (n->type() == GetRaise)
          {
            found = true;
            return false;
          }
          return true;
        });
        return found;
      };

      // Helper: resolve a receiver type + method_id to a FunctionId node.
      // Returns empty Node if unresolvable.
      auto resolve_method =
        [&](const Node& src_type, const Node& method_id) -> Node {
        if (!src_type)
          return {};

        // Cannot devirtualize unions or Dyn.
        if (src_type == Union || src_type == Dyn)
          return {};

        Node cls;
        if (src_type == ClassId)
          cls = find_class(src_type);
        else if (is_primitive(src_type))
          cls = find_primitive(src_type);
        else if (src_type->type().in({Array, Cown, Ref}))
        {
          for (auto& child : *top)
          {
            if (
              child == Primitive && (child / Type)->type() == src_type->type())
            {
              cls = child;
              break;
            }
          }
        }

        if (!cls)
          return {};

        for (auto& method : *(cls / Methods))
        {
          if ((method / MethodId)->location() == method_id->location())
            return method / FunctionId;
        }
        return {};
      };

      // Process each function.
      for (auto& func_node : *top)
      {
        if (func_node != Func)
          continue;

        auto func_key =
          std::string((func_node / FunctionId)->location().view());
        auto lookups_it = state->func_lookups.find(func_key);

        // Empty lookup map for functions with no Lookup statements.
        std::unordered_map<std::string, LookupInfo> empty_lookups;
        auto& lookups = (lookups_it != state->func_lookups.end()) ?
          lookups_it->second :
          empty_lookups;

        // Collect dead lookup register names (Phase C will remove them).
        std::set<std::string> dead_lookups;

        auto& func_state = state->get_func(func_node / FunctionId);
        auto caller_vars = func_node / Vars;

        // Phase A: Devirtualize.
        // Phase B: Inline trivial calls.
        // We do both in a single pass through each label's body, since
        // Phase A produces Call nodes that Phase B can immediately inline.
        for (auto& label : *(func_node / Labels))
        {
          auto body = label / Body;

          // Collect replacements (can't mutate during iteration).
          std::vector<std::pair<Node, Node>> replacements;

          for (auto& stmt : *body)
          {
            // Phase A: Devirtualize CallDyn.
            if (stmt == CallDyn || stmt == TryCallDyn)
            {
              auto fn_ptr_name = std::string((stmt / Rhs)->location().view());
              auto it = lookups.find(fn_ptr_name);

              if (it == lookups.end())
                continue;

              auto& info = it->second;
              auto func_id = resolve_method(info.src_type, info.method_id);

              if (stmt == CallDyn)
              {
                if (!func_id)
                  continue;

                // Replace CallDyn with Call.
                Node call = Call << clone(stmt / LocalId) << clone(func_id)
                                 << clone(stmt / Args);
                replacements.push_back({stmt, call});
                dead_lookups.insert(fn_ptr_name);
              }
              else // TryCallDyn
              {
                if (func_id)
                {
                  // Method exists — replace with Call.
                  Node call = Call << clone(stmt / LocalId) << clone(func_id)
                                   << clone(stmt / Args);
                  replacements.push_back({stmt, call});
                }
                else if (
                  info.src_type && info.src_type != Union &&
                  info.src_type != Dyn)
                {
                  // Known non-union type, method doesn't exist — produce
                  // none.
                  Node none_const = Const << clone(stmt / LocalId) << None
                                          << None;
                  replacements.push_back({stmt, none_const});
                }
                else
                {
                  continue;
                }
                dead_lookups.insert(fn_ptr_name);
              }
            }
            // Phase A: Devirtualize WhenDyn.
            else if (stmt == WhenDyn)
            {
              auto fn_ptr_name = std::string((stmt / Rhs)->location().view());
              auto it = lookups.find(fn_ptr_name);

              if (it == lookups.end())
                continue;

              auto& info = it->second;
              auto func_id = resolve_method(info.src_type, info.method_id);

              if (!func_id)
                continue;

              // Replace WhenDyn with When.
              Node when = When << clone(stmt / LocalId) << clone(func_id)
                               << clone(stmt / Args) << clone(stmt / Cown);
              replacements.push_back({stmt, when});
              dead_lookups.insert(fn_ptr_name);
            }
          }

          // Apply Phase A replacements.
          for (auto& [old_node, new_node] : replacements)
            old_node->parent()->replace(old_node, new_node);

          // Phase B: Inline single-label functions.
          // Fixpoint loop: repeat until no more calls are inlined.
          // Each iteration may expose new Call targets from inlined bodies.
          bool inline_changed = true;

          while (inline_changed)
          {
            inline_changed = false;

            for (size_t si = 0; si < body->size(); si++)
            {
              auto stmt = body->at(si);

              if (stmt != Call)
                continue;

              auto target = find_func(stmt / FunctionId);

              if (!target)
                continue;

              // Don't inline functions that capture a raise target into an
              // ordinary caller. Inlining would move block-lambda creation
              // and change the captured frame. The synthetic @main wrapper is
              // safe because it immediately delegates to main and returns.
              if (
                captures_raise_target(target) &&
                (func_node / FunctionId)->location().view() != "@main")
                continue;

              // Don't inline self-recursive calls.
              if (
                (target / FunctionId)->location() ==
                (func_node / FunctionId)->location())
                continue;

              auto labels_node = target / Labels;

              if (labels_node->size() != 1)
                continue;

              auto target_label = labels_node->front();
              auto target_body = target_label / Body;
              auto target_term = target_label->back();

              if (target_term != Return)
                continue;

              // Build param→arg mapping.
              auto params = target / Params;
              auto args = stmt / Args;

              if (params->size() != args->size())
                continue;

              std::unordered_map<std::string, std::string> param_map;
              auto p_it = params->begin();
              auto a_it = args->begin();

              while (p_it != params->end() && a_it != args->end())
              {
                auto param_name =
                  std::string(((*p_it) / LocalId)->location().view());
                auto arg_name = std::string(((*a_it) / Rhs)->location().view());
                param_map[param_name] = arg_name;
                ++p_it;
                ++a_it;
              }

              auto call_dst_name =
                std::string((stmt / LocalId)->location().view());
              auto ret_name =
                std::string((target_term / LocalId)->location().view());

              // Build alpha-rename map for callee-internal locals.
              // Includes: callee dst registers, callee vars that aren't
              // params. Does NOT include params (mapped to args) or the
              // return value (mapped to call dst).
              std::unordered_map<std::string, std::string> rename_map;

              // Map return value → call dst.
              rename_map[ret_name] = call_dst_name;

              // Collect all callee locals that need renaming.
              for (auto& ts : *target_body)
              {
                auto dst_name = std::string((ts / LocalId)->location().view());

                if (
                  param_map.count(dst_name) == 0 &&
                  rename_map.count(dst_name) == 0)
                {
                  auto new_name = "$opt_" + std::to_string(inline_counter++);
                  rename_map[dst_name] = new_name;
                }
              }

              // Handle identity functions: empty body, return a param.
              if (target_body->size() == 0)
              {
                auto ret_it = param_map.find(ret_name);

                if (ret_it != param_map.end())
                {
                  // Return value is a param — emit Copy(call_dst, arg).
                  Node copy = Copy << (LocalId ^ call_dst_name)
                                   << (LocalId ^ ret_it->second);
                  body->replace(stmt, copy);
                  inline_changed = true;
                }
                continue;
              }

              // Clone and remap all callee body statements.
              // The full rename map: param_map + rename_map.
              // param_map: callee param name → caller arg name
              // rename_map: callee internal name → alpha-renamed name
              auto remap = [&](Node& n) {
                // Walk all LocalId descendants and remap.
                std::function<void(Node&)> walk = [&](Node& node) {
                  for (size_t i = 0; i < node->size(); i++)
                  {
                    auto child = node->at(i);

                    if (child == LocalId)
                    {
                      auto name = std::string(child->location().view());

                      // Check param_map first.
                      auto pit = param_map.find(name);

                      if (pit != param_map.end())
                      {
                        node->replace(child, LocalId ^ pit->second);

                        // ArgMove→ArgCopy for caller-owned registers.
                        if (node == Arg && (node / Type) == ArgMove)
                          node->replace(node / Type, ArgCopy);
                      }
                      else
                      {
                        // Check rename_map.
                        auto rit = rename_map.find(name);

                        if (rit != rename_map.end())
                          node->replace(child, LocalId ^ rit->second);
                      }
                    }
                    else if (child->size() > 0)
                    {
                      walk(child);
                    }
                  }
                };
                walk(n);
              };

              // Register new locals and vars in the caller.
              for (auto& [old_name, new_name] : rename_map)
              {
                // Skip the return→call_dst mapping (already registered).
                if (new_name == call_dst_name)
                  continue;

                // Create a temporary node for registration.
                Node new_local = LocalId ^ new_name;
                func_state.add_register(new_local);
              }

              // Add callee vars to caller vars (alpha-renamed).
              for (auto& v : *(target / Vars))
              {
                auto var_name = std::string((v / LocalId)->location().view());
                auto rit = rename_map.find(var_name);

                if (rit != rename_map.end())
                  caller_vars
                    << (VarDef << (LocalId ^ rit->second) << clone(v / Type));
                else
                {
                  // Var is a param name — shouldn't happen but handle
                  // gracefully by skipping.
                }
              }

              // Build the replacement statement list.
              std::vector<Node> inlined_stmts;

              for (auto& ts : *target_body)
              {
                Node cloned = clone(ts);
                remap(cloned);
                inlined_stmts.push_back(cloned);
              }

              auto ret_it = param_map.find(ret_name);

              if (ret_it != param_map.end())
              {
                Node copy = Copy << (LocalId ^ call_dst_name)
                                 << (LocalId ^ ret_it->second);
                inlined_stmts.push_back(copy);
              }

              // Replace the Call with the inlined statements.
              // Insert all statements at the Call's position, then remove
              // the Call.
              auto it = body->find(stmt);
              assert(it != body->end());

              for (auto& is : inlined_stmts)
                it = body->insert(++it, is);

              body->erase(body->find(stmt), std::next(body->find(stmt)));

              // Restart the scan — inlined body may contain new Calls.
              inline_changed = true;
              break;
            }
          } // while (inline_changed)

          // Phase C: Remove dead Lookups and Drops.
          // Only remove a Lookup if no remaining statement uses it.
          std::set<std::string> still_used;

          for (auto& stmt : *body)
          {
            if (stmt->type().in({CallDyn, TryCallDyn, WhenDyn}))
            {
              auto name = std::string((stmt / Rhs)->location().view());
              still_used.insert(name);
            }
          }

          std::vector<Node> to_remove;

          for (auto& stmt : *body)
          {
            if (stmt == Lookup)
            {
              auto dst_name = std::string((stmt / LocalId)->location().view());

              if (dead_lookups.count(dst_name) && !still_used.count(dst_name))
                to_remove.push_back(stmt);
            }
            else if (stmt == Drop)
            {
              auto drop_name = std::string((stmt / LocalId)->location().view());

              if (dead_lookups.count(drop_name) && !still_used.count(drop_name))
                to_remove.push_back(stmt);
            }
          }

          for (auto& node : to_remove)
            node->parent()->replace(node);

          // Remove Copy(dst, src) when dst is unused. This must run before
          // liveness because generated code may copy an undefined old value
          // into an unused result.
          std::set<std::string> all_used;

          for (auto& lbl : *(func_node / Labels))
          {
            for (auto& s : *(lbl / Body))
            {
              // Most statements define their first LocalId. Statements that
              // only consume registers have no destination to skip.
              size_t first_use = s->in({Drop, AtTeardown}) ? 0 : 1;

              for (size_t i = first_use; i < s->size(); i++)
              {
                s->at(i)->traverse([&](Node& n) {
                  if (n == LocalId)
                    all_used.insert(std::string(n->location().view()));
                  return true;
                });
              }
            }

            // Terminator uses.
            (lbl / Return)->traverse([&](Node& n) {
              if (n == LocalId)
                all_used.insert(std::string(n->location().view()));
              return true;
            });
          }

          std::vector<Node> dead_copies;

          for (auto& stmt : *body)
          {
            if (stmt != Copy)
              continue;

            auto dst_name = std::string((stmt / LocalId)->location().view());

            if (!all_used.count(dst_name))
              dead_copies.push_back(stmt);
          }

          for (auto& dc : dead_copies)
            dc->parent()->replace(dc);
        }

        // Resize LabelState structures to account for new registers
        // added during inlining. This is needed because liveness runs
        // after optimize and uses these structures.
        for (auto& ls : func_state.labels)
          ls.resize(func_state.register_names.size());

        // Phase A for terminators: Devirtualize TailcallDyn.
        for (auto& label : *(func_node / Labels))
        {
          auto term = label->back();

          if (term != TailcallDyn)
            continue;

          auto fn_ptr_name = std::string((term / LocalId)->location().view());
          auto it = lookups.find(fn_ptr_name);

          if (it == lookups.end())
            continue;

          auto& info = it->second;
          auto func_id = resolve_method(info.src_type, info.method_id);

          if (!func_id)
            continue;

          Node tailcall = Tailcall << clone(func_id) << clone(term / MoveArgs);
          label->replace(term, tailcall);
          dead_lookups.insert(fn_ptr_name);
        }
      }

      return 0;
    });

    return p;
  }
}
