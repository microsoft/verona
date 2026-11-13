#include "../lang.h"
#include "../subtype.h"

#include <virc/analysis/ir_subtype.h>

namespace vc
{
  const std::map<std::string_view, Token> wrapper_types = {
    {"array", Array},
    {"cown", Cown},
    {"ref", Ref},
  };

  const std::map<std::string_view, Node> primitive_types = {
    {"none", None},
    {"bool", Bool},
    {"i8", I8},
    {"i16", I16},
    {"i32", I32},
    {"i64", I64},
    {"u8", U8},
    {"u16", U16},
    {"u32", U32},
    {"u64", U64},
    {"ilong", ILong},
    {"ulong", ULong},
    {"isize", ISize},
    {"usize", USize},
    {"f32", F32},
    {"f64", F64},
  };

  // Primitive types nested under _builtin::ffi.
  const std::map<std::string_view, Node> ffi_primitive_types = {
    {"ptr", Ptr},
  };

  struct Reifier
  {
    Reifier() {}

    // Check if a def's name is in the primitive or ffi_primitive maps.
    bool is_any_primitive(const Node& def) const
    {
      auto name = (def / Ident)->location().view();
      return primitive_types.find(name) != primitive_types.end() ||
        ffi_primitive_types.find(name) != ffi_primitive_types.end();
    }

    // Check if a def is transitively under the _builtin scope.
    bool is_under_builtin(const Node& def) const
    {
      auto parent = def->parent(ClassDef);

      while (parent)
      {
        if (parent == builtin)
          return true;
        parent = parent->parent(ClassDef);
      }

      return false;
    }

    bool contains_dyn(const Node& type) const
    {
      if (!type)
        return false;

      if (type == Dyn)
        return true;

      for (auto& child : *type)
      {
        if (contains_dyn(child))
          return true;
      }

      return false;
    }

    bool contains_typeid(const Node& type) const
    {
      if (!type)
        return false;

      if (type == TypeId)
        return true;

      for (auto& child : *type)
      {
        if (contains_typeid(child))
          return true;
      }

      return false;
    }

    bool is_nomatch_ir(const Node& type) const
    {
      return type && (type == ClassId) &&
        (type->location().view() == "_builtin::nomatch::0");
    }

    bool has_unresolved_type(const Node& type, const NodeMap<Node>& subst) const
    {
      std::set<Node> seen;
      return has_unresolved_type(type, subst, seen);
    }

    bool contains_typeparam_ref(const Node& type) const
    {
      if (!type)
        return false;

      if (type == Type)
        return contains_typeparam_ref(type->front());

      if (type == TypeVar)
        return true;

      if (type == TypeName)
      {
        auto def = top;

        for (auto it = type->begin(); it != type->end(); ++it)
        {
          auto& elem = *it;

          for (auto& arg : *(elem / TypeArgs))
          {
            if (contains_typeparam_ref(arg))
              return true;
          }

          auto defs = def->look((elem / Ident)->location());

          if (defs.empty())
            return false;

          if (defs.size() > 1 && (it + 1) != type->end())
          {
            auto next_ident = (*(it + 1)) / Ident;
            auto picked = defs.front();

            for (auto& d : defs)
            {
              if (!d->look(next_ident->location()).empty())
              {
                picked = d;
                break;
              }
            }

            def = picked;
          }
          else
          {
            def = defs.front();
          }

          if (def == TypeParam)
            return true;
        }

        return false;
      }

      for (auto& child : *type)
      {
        if (contains_typeparam_ref(child))
          return true;
      }

      return false;
    }

    bool has_unresolved_type(
      const Node& type, const NodeMap<Node>& subst, std::set<Node>& seen) const
    {
      if (!type)
        return false;

      if (type == Type)
        return has_unresolved_type(type->front(), subst, seen);

      if (type == TypeVar)
        return true;

      if (type == TypeName)
      {
        auto def = top;

        for (auto it = type->begin(); it != type->end(); ++it)
        {
          auto& elem = *it;

          for (auto& arg : *(elem / TypeArgs))
          {
            if (has_unresolved_type(arg, subst, seen))
              return true;
          }

          auto defs = def->look((elem / Ident)->location());

          if (defs.empty())
            return false;

          if (defs.size() > 1 && (it + 1) != type->end())
          {
            auto next_ident = (*(it + 1)) / Ident;
            auto picked = defs.front();

            for (auto& d : defs)
            {
              if (!d->look(next_ident->location()).empty())
              {
                picked = d;
                break;
              }
            }

            def = picked;
          }
          else
          {
            def = defs.front();
          }

          if (def == TypeParam)
          {
            if (!seen.insert(def).second)
              return true;

            auto find = subst.find(def);

            if (find == subst.end())
            {
              seen.erase(def);
              return true;
            }

            auto unresolved = has_unresolved_type(find->second, subst, seen);
            seen.erase(def);
            return unresolved;
          }
        }

        return false;
      }

      for (auto& child : *type)
      {
        if (has_unresolved_type(child, subst, seen))
          return true;
      }

      return false;
    }

    void emit_unresolved_type_error(const Node& blame, std::string_view context)
    {
      errors.push_back(err(
        blame,
        std::format("Could not resolve {} during monomorphization", context)));
    }

    Node reify_emitted_type(
      const Node& source,
      const NodeMap<Node>& subst,
      const Node& blame,
      std::string_view context)
    {
      auto ir_type = reify_type(source, subst);

      if (contains_dyn(ir_type) && has_unresolved_type(source, subst))
        emit_unresolved_type_error(blame, context);

      return ir_type;
    }

    void run(Node& top_)
    {
      top = top_;
      builtin = top->look(Location("_builtin")).front();

      // Create a call to main and reify it.
      auto main_module = top->front();
      assert(main_module == ClassDef);
      assert((main_module / TypeParams)->empty());

      // Check that main is defined under the main module and has no type
      // parameters.
      auto main_defs = main_module->look(Location("main"));
      if (main_defs.empty())
      {
        top << err(
          main_module,
          "No `main` function found under the main module: " +
            std::string((main_module / Ident)->location().view()));
        return;
      }

      auto main_def = main_defs.front();
      if ((main_def / TypeParams)->empty() == false)
      {
        top << err(
          main_def,
          "`main` function cannot have type parameters: " +
            std::string((main_def / Ident)->location().view()));
        return;
      }

      auto id = top->fresh();
      auto main_call = Call
        << (LocalId ^ id) << Rhs
        << (FuncName << (NameElement << clone(main_module / Ident) << TypeArgs)
                     << (NameElement << (Ident ^ "main") << TypeArgs))
        << Args;

      reify_call(main_call, {});

      // Iteratively reify classes/aliases/functions. Method registrations
      // happen inline: reify_class registers all existing MIs on the new
      // class, and reify_lookup registers the new MI on all existing classes.
      std::vector<Reification*> deferred_typevar;
      drain_worklist(deferred_typevar);
      resolve_shapes();
      prune_empty_shape_unions();
      process_pending_callbacks(false);
      drain_worklist(deferred_typevar);
      resolve_shapes();
      prune_empty_shape_unions();
      process_pending_callbacks(true);
      drain_worklist(deferred_typevar);

      std::vector<Reification*> reified_functions;

      for (auto& key : map_order)
      {
        if (key != Function)
          continue;

        for (auto& r : map[key])
        {
          if (r.reification)
            reified_functions.push_back(&r);
        }
      }

      // Second pass: refine unresolved parameter and return types from the
      // already-reified bodies now that all callees and methods are available.
      bool changed;
      do
      {
        changed = false;

        for (auto r : reified_functions)
        {
          auto func = r->reification;
          auto labels = func / Labels;

          // Rebuild local_types by scanning the reified body.
          local_types.clear();
          lookup_info.clear();

          // Track param types.
          for (auto& p : *(func / Params))
            local_types[(p / LocalId)->location()] = clone(p / Type);

          for (auto& lbl : *labels)
          {
            for (auto& stmt : *(lbl / Body))
            {
              if (stmt->in({Const, Convert}))
              {
                local_types[(stmt / LocalId)->location()] = clone(stmt / Type);
              }
              else if (stmt == ConstStr)
              {
                local_types[(stmt / LocalId)->location()] = Array << clone(U8);
              }
              else if (stmt == TypedMove)
              {
                auto dst_loc = (stmt / LocalId)->location();
                local_types[dst_loc] = clone(stmt / Type);
              }
              else if (stmt->in({Copy, Move}))
              {
                auto dst_loc = (stmt / LocalId)->location();

                auto src_it = local_types.find((stmt / Rhs)->location());
                if (src_it != local_types.end())
                {
                  auto dst_it = local_types.find(dst_loc);

                  local_types[dst_loc] = (dst_it == local_types.end()) ?
                    clone(src_it->second) :
                    merge_refined_type(dst_it->second, src_it->second);
                }
              }
              else if (stmt->in({New, Stack}))
              {
                local_types[(stmt / LocalId)->location()] =
                  clone(stmt / ClassId);
              }
              else if (stmt == MemoSlot)
              {
                auto type = empty_class_type(stmt / MemoId);
                if (type)
                  local_types[(stmt / LocalId)->location()] = type;
              }
              else if (stmt == FieldRef)
              {
                auto obj_loc = (stmt / Arg / Rhs)->location();
                auto obj_it = local_types.find(obj_loc);
                if (obj_it != local_types.end() && (obj_it->second == ClassId))
                {
                  auto ft = find_field_type(obj_it->second, stmt / FieldId);
                  if (ft)
                    local_types[(stmt / LocalId)->location()] = Ref << ft;
                }
              }
              else if (stmt == Lookup)
              {
                auto mid = (stmt / MethodId)->location().view();
                lookup_info[(stmt / LocalId)->location()] = {
                  std::string(mid), (stmt / Rhs)->location()};
              }
              else if (stmt == Call)
              {
                if (auto* target = find_function_reification(stmt / FunctionId))
                  changed |=
                    refine_function_params(*target, stmt->at(2), false);

                auto ret = find_func_return_type(stmt / FunctionId);
                if (ret)
                  local_types[(stmt / LocalId)->location()] = ret;
              }
              else if (stmt->in({CallDyn, TryCallDyn}))
              {
                auto src_loc = (stmt / Rhs)->location();
                auto li = lookup_info.find(src_loc);
                if (li != lookup_info.end())
                {
                  auto recv_it = local_types.find(li->second.recv_loc);
                  if (recv_it != local_types.end())
                  {
                    auto targets = find_method_targets(
                      recv_it->second,
                      li->second.method_id,
                      stmt->at(2),
                      false);

                    if (
                      targets.empty() &&
                      receiver_is_param(func, li->second.recv_loc))
                    {
                      auto fallback_targets = find_method_targets(
                        Dyn, li->second.method_id, stmt->at(2), false);

                      if (refine_receiver_type(
                            func,
                            li->second.recv_loc,
                            recv_it->second,
                            fallback_targets))
                      {
                        changed = true;
                        targets = std::move(fallback_targets);
                      }
                    }

                    bool unresolved_receiver = contains_dyn(recv_it->second) ||
                      contains_typeid(recv_it->second);

                    bool skip_param_refinement =
                      unresolved_receiver && (targets.size() > 1);

                    if (!skip_param_refinement)
                    {
                      for (auto* target : targets)
                      {
                        if (target)
                          changed |=
                            refine_function_params(*target, stmt->at(2), false);
                      }
                    }

                    auto ret = find_method_return_type(targets);

                    if (!ret && recv_it->second == Ref)
                    {
                      auto& mid = li->second.method_id;
                      if (mid.starts_with("*::"))
                        ret = clone(recv_it->second->front());
                    }

                    if (ret)
                      local_types[(stmt / LocalId)->location()] = ret;
                  }
                }
              }
              else if (stmt == Load)
              {
                auto src_it = local_types.find((stmt / Rhs)->location());
                if (src_it != local_types.end() && (src_it->second == Ref))
                  local_types[(stmt / LocalId)->location()] =
                    clone(src_it->second->front());
              }
              else if (stmt == WhenDyn)
              {
                auto cown_type = stmt / Cown;
                auto li = lookup_info.find((stmt / Rhs)->location());

                if (li != lookup_info.end())
                {
                  auto recv_it = local_types.find(li->second.recv_loc);

                  if (recv_it != local_types.end())
                  {
                    auto targets = find_method_targets(
                      recv_it->second, li->second.method_id, stmt->at(2), true);

                    if (
                      targets.empty() &&
                      receiver_is_param(func, li->second.recv_loc))
                    {
                      auto fallback_targets = find_method_targets(
                        Dyn, li->second.method_id, stmt->at(2), true);

                      if (refine_receiver_type(
                            func,
                            li->second.recv_loc,
                            recv_it->second,
                            fallback_targets))
                      {
                        changed = true;
                        targets = std::move(fallback_targets);
                      }
                    }

                    bool unresolved_receiver = contains_dyn(recv_it->second) ||
                      contains_typeid(recv_it->second);

                    bool skip_param_refinement =
                      unresolved_receiver && (targets.size() > 1);

                    if (!skip_param_refinement)
                    {
                      for (auto* target : targets)
                      {
                        if (target)
                          changed |=
                            refine_function_params(*target, stmt->at(2), true);
                      }
                    }

                    // TypeVar-returning behaviors may have been reified with a
                    // provisional return (for example just nomatch) before the
                    // deferred return-type pass converges. Refresh the result
                    // cown type when the reified return differs from the
                    // current cown type.
                    {
                      auto ret = find_method_return_type(targets);

                      if (ret && (ret != Dyn))
                      {
                        Node new_cown = Cown << clone(ret);

                        if (!cown_type->equals(new_cown))
                        {
                          stmt->replace(cown_type, new_cown);
                          changed = true;
                        }

                        cown_type = stmt / Cown;
                      }
                    }
                  }
                }

                local_types[(stmt / LocalId)->location()] = clone(cown_type);
              }
              else if (stmt == When)
              {
                auto cown_type = stmt / Cown;

                if (auto* target = find_function_reification(stmt / FunctionId))
                {
                  changed |= refine_function_params(*target, stmt->at(2), true);

                  bool needs_refresh = (cown_type == Cown) &&
                    ((cown_type->front() == Dyn) ||
                     ((target->def / Type)->front() == TypeVar));

                  if (needs_refresh)
                  {
                    Node ret = target->reification / Type;

                    if (ret && (ret != Dyn))
                    {
                      Node new_cown = Cown << clone(ret);

                      if (!cown_type->equals(new_cown))
                      {
                        stmt->replace(cown_type, new_cown);
                        changed = true;
                      }

                      cown_type = stmt / Cown;
                    }
                  }
                }

                local_types[(stmt / LocalId)->location()] = clone(cown_type);
              }
            }
          }

          auto current_ret = func / Type;
          if (
            ((r->def / Type)->front() == TypeVar) || is_nomatch_ir(current_ret))
          {
            // Now try to infer the return type from Return locals.
            // Collect all distinct return types to build a union if needed.
            Nodes ret_types;

            for (auto& lbl : *labels)
            {
              auto term = lbl / Return;
              if (term != Return)
                continue;

              auto ret_loc = (term / LocalId)->location();
              auto it = local_types.find(ret_loc);
              if (it == local_types.end())
                continue;

              bool dup = false;

              for (auto& existing : ret_types)
              {
                if (existing->equals(it->second))
                {
                  dup = true;
                  break;
                }
              }

              if (!dup)
                ret_types.push_back(clone(it->second));
            }

            Node new_ret;

            if (ret_types.size() == 1)
              new_ret = ret_types.front();
            else if (ret_types.size() > 1)
            {
              Node union_node = Union;

              for (auto& rt : ret_types)
                union_node << clone(rt);

              new_ret = union_node;
            }

            if (new_ret && new_ret->type() != Dyn)
            {
              auto old_type = func / Type;
              if (!old_type->equals(new_ret))
              {
                func->replace(old_type, new_ret);
                changed = true;
              }
            }
          }
        }

      } while (changed);

      for (auto r : reified_functions)
      {
        for (auto& lbl : *(r->reification / Labels))
        {
          for (auto& stmt : *(lbl / Body))
          {
            if (stmt == TypedMove)
            {
              auto replacement = Move ^ stmt;
              replacement << clone(stmt / LocalId) << clone(stmt / Rhs);
              stmt->parent()->replace(stmt, replacement);
            }
          }
        }
      }

      for (auto r : deferred_typevar)
      {
        if (r->reification && ((r->reification / Type) == Dyn))
          emit_unresolved_type_error(r->def / Ident, "return type");
      }

      for (auto r : reified_functions)
      {
        auto params = r->reification / Params;
        auto def_params = r->def / Params;

        for (size_t i = 0; i < params->size(); i++)
        {
          if (
            contains_dyn(params->at(i) / Type) &&
            has_unresolved_type((def_params->at(i) / Type), r->subst))
          {
            emit_unresolved_type_error(
              def_params->at(i) / Ident, "parameter type");
          }
        }
      }

      resolve_shapes();
      record_dynamic_call_edges();

      // Remove existing contents.
      top->erase(top->begin(), top->end());

      // Add an entry point for main.
      top
        << (Func << (FunctionId ^ "@main") << Params << None << Vars
                 << (Labels
                     << (Label << (LabelId ^ "start") << (Body << main_call)
                               << (Return << (LocalId ^ id)))));

      // Add reified classes, type aliases, and functions.
      // Emit non-Primitive entries first so that Type entries (which
      // define TypeId resolutions) are in top before we check Primitives.
      for (auto& key : map_order)
        for (auto& r : map[key])
          if (r.reification && r.reification != Primitive)
          {
            if (r.def == Function && (r.def / Lhs) == Once)
            {
              auto orig_id = r.reification / FunctionId;
              auto orig_id_str = std::string(orig_id->location().view());
              auto init_id_str = orig_id_str + "$once";
              auto slot_id_str = orig_id_str + "$slot";
              auto r_type = r.reification / Type;
              auto params = r.reification / Params;
              auto vars = r.reification / Vars;
              auto labels = r.reification / Labels;

              top << (Func << (FunctionId ^ orig_id_str) << Params
                           << clone(r_type)
                           << (Vars << (VarDef << (LocalId ^ "$memo") << Dyn))
                           << (Labels
                               << (Label
                                   << (LabelId ^ "entry")
                                   << (Body
                                       << (MemoSlot << (LocalId ^ "$memo")
                                                    << (MemoId ^ slot_id_str)))
                                   << (Return << (LocalId ^ "$memo")))))
                  << (Func << (FunctionId ^ init_id_str) << params
                           << clone(r_type) << vars << labels);
            }
            else
            {
              top << r.reification;
            }
          }

      for (auto& [_, instance] : empty_class_memos)
        top << instance.memo << instance.initializer;

      // Emit Primitives, deduplicating wrappers whose inner types are
      // invariantly equivalent after alias/shape resolution. Rebuild the
      // wfIR symbol table after populating top so IRSubtype can resolve
      // TypeId through Top::look(), then use it for union set equality and
      // wrapper invariance.
      {
        WFContext wf_ctx(wfIR);
        wfIR.build_st(top);
        std::vector<std::pair<Token, Node>> emitted;

        for (auto& key : map_order)
        {
          for (auto& r : map[key])
          {
            if (!r.reification || r.reification != Primitive)
              continue;

            auto ptype = r.reification / Type;

            if (ptype->in({Array, Ref, Cown}))
            {
              auto wrapper = ptype->type();
              auto inner = ptype->front();
              bool dup = false;

              for (auto& [ew, ei] : emitted)
              {
                if (ew == wrapper && virc::IRSubtype.invariant(top, inner, ei))
                {
                  dup = true;
                  break;
                }
              }

              if (dup)
                continue;

              emitted.emplace_back(wrapper, inner);
            }

            top << r.reification;
          }
        }
      }

      // Add reified libraries.
      for (auto& [_, lib] : libs)
        top << lib;

      // Add any errors collected during reification.
      for (auto& e : errors)
        top << e;
    }

    const CallGraph& calls() const
    {
      return call_graph;
    }

    const OnceFunctions& once_functions() const
    {
      return once_funcs;
    }

  private:
    // Each ClassDef (including primitives), TypeAlias, or Function that we
    // reify gets a Reification struct.
    struct Reification
    {
      Node def;
      NodeMap<Node> subst;
      Node id;
      Node reification;
      Node resolved_name; // Resolved TypeName for shape checking
    };

    static bool same_reification_id(const Node& left, const Node& right)
    {
      if (!left || !right || (left->type() != right->type()))
        return false;

      if (left->in({ClassId, FunctionId, TypeId}))
        return left->location().view() == right->location().view();

      Node left_id = left;
      Node right_id = right;
      return left_id->equals(right_id);
    }

    struct ReceiverSet
    {
      bool all;
      Nodes types;

      bool contains(const Node& receiver) const
      {
        return all ||
          std::any_of(types.begin(), types.end(), [&](auto& candidate) {
                 return same_reification_id(candidate, receiver);
               });
      }
    };

    // A MethodInvocation captures a Lookup site so we can register the
    // appropriate Method entries on every class/primitive that could receive
    // a CallDyn on this MethodId.
    struct MethodInvocation
    {
      Location caller_id;
      Node site;
      Node receiver_type;
      std::string method_id; // the compiled MethodId string
      std::string name; // function name
      size_t arity; // parameter count
      Token hand; // Lhs (ref) or Rhs
      Node typeargs; // cloned TypeArgs from the Lookup
      NodeMap<Node> call_subst; // substitution context at the call site
      ReceiverSet receivers;
    };

    struct DynamicCallTarget
    {
      Node receiver;
      Node callee;
    };

    struct PendingCallback
    {
      Node site;
      Node type;
      bool required;
      Node registrant;
      Node registration_site;
    };

    struct EmptyClassMemo
    {
      Node type;
      Node memo;
      Node initializer;
    };

    Node top;
    Node builtin;
    NodeMap<std::deque<Reification>> map;
    std::vector<Node> map_order;
    std::vector<Reification*> worklist;
    std::map<Location, Node> libs;
    std::map<std::string, EmptyClassMemo> empty_class_memos;
    Nodes errors;
    std::vector<MethodInvocation> method_invocations;
    std::map<std::string, std::vector<DynamicCallTarget>> dynamic_call_targets;
    std::vector<PendingCallback> pending_callbacks;
    std::map<std::string, std::vector<std::vector<Node>>> method_index;
    CallGraph call_graph;
    OnceFunctions once_funcs;
    std::map<std::pair<const NodeDef*, const NodeDef*>, bool>
      shape_subtype_cache;

    // Per-function local type map: LocalId location -> reified type.
    // Populated during reify_function, used by reify_lookup.
    std::map<Location, Node> local_types;

    // Per-function lookup info: maps Lookup dst location to
    // {MethodId string, receiver location}. Used to resolve CallDyn and
    // When return types.
    struct LookupInfo
    {
      std::string method_id;
      Location recv_loc;
    };
    std::map<Location, LookupInfo> lookup_info;

    void
    record_call_edge(const Node& caller, const Node& callee, const Node& site)
    {
      assert(caller == FunctionId);
      assert(callee == FunctionId);

      auto caller_id = caller->location();
      auto callee_id = callee->location();
      call_graph[caller_id].try_emplace(callee_id, site);
    }

    void record_dynamic_call_edges()
    {
      // Shape TypeIds are resolved only after the reification worklist has
      // converged. Filter the targets indexed by register_method using the
      // final receiver sets.
      for (auto& mi : method_invocations)
      {
        auto targets = dynamic_call_targets.find(mi.method_id);

        if (targets == dynamic_call_targets.end())
          continue;

        auto receivers = extract_receivers(mi.receiver_type);
        Node caller = FunctionId ^ mi.caller_id;

        for (auto& target : targets->second)
        {
          if (!receivers.contains(target.receiver))
            continue;

          record_call_edge(caller, target.callee, mi.site);
        }
      }
    }

    Node empty_class_type(const Node& memo_id)
    {
      auto find =
        empty_class_memos.find(std::string(memo_id->location().view()));
      if (find == empty_class_memos.end())
        return {};

      return clone(find->second.type);
    }

    Node ensure_empty_class_memo(const Node& class_id)
    {
      auto class_name = std::string(class_id->location().view());
      auto memo_id = class_name + "$instance";
      auto find = empty_class_memos.find(memo_id);
      if (find != empty_class_memos.end())
        return clone(find->second.memo / MemoId);

      auto initializer_id = class_name + "$instance_init";
      Node instance = LocalId ^ "$instance";

      Node initializer = Func
        << (FunctionId ^ initializer_id) << Params << clone(class_id)
        << (Vars << (VarDef << clone(instance) << clone(class_id)))
        << (Labels
            << (Label
                << (LabelId ^ "entry")
                << (Body << (New << clone(instance) << clone(class_id) << Args))
                << (Return << clone(instance))));

      Node memo = Memo << (MemoId ^ memo_id) << (FunctionId ^ initializer_id);

      empty_class_memos.emplace(
        memo_id,
        EmptyClassMemo{
          clone(class_id), std::move(memo), std::move(initializer)});
      return MemoId ^ memo_id;
    }

    void drain_worklist(std::vector<Reification*>& deferred_typevar)
    {
      while (!worklist.empty())
      {
        auto r = worklist.back();
        worklist.pop_back();

        if (r->def == ClassDef)
          reify_class(*r);
        else if (r->def == TypeAlias)
          reify_typealias(*r);
        else if (r->def == Function)
        {
          reify_function(*r);

          // If the function had a TypeVar return in the def, defer it
          // for a second pass when all callees are reified. The first
          // pass may have produced a partial return type (e.g., nomatch
          // from match arms when the main arm's CallDyn wasn't tracked).
          if (r->reification && (r->def / Type)->front() == TypeVar)
            deferred_typevar.push_back(r);
        }
        else
        {
          assert(false);
        }
      }
    }

    void resolve_shapes()
    {
      SequentCtx ctx{top, {}, {}};

      for (auto& key : map_order)
      {
        for (auto& r : map[key])
        {
          if (r.def != ClassDef || (r.def / Shape) != Shape)
            continue;

          assert(r.resolved_name);
          Node union_node = Union;

          for (auto& ckey : map_order)
          {
            for (auto& cr : map[ckey])
            {
              if (cr.def != ClassDef || (cr.def / Shape) == Shape)
                continue;
              if (!cr.resolved_name)
                continue;

              auto cache_key =
                std::make_pair(cr.resolved_name.get(), r.resolved_name.get());
              auto [cache_it, inserted] =
                shape_subtype_cache.try_emplace(cache_key, false);

              if (
                inserted &&
                check_shape_subtype(ctx, cr.resolved_name, r.resolved_name))
              {
                cache_it->second = true;
              }

              if (cache_it->second)
                union_node << clone(cr.id);
            }
          }

          if (union_node->empty())
          {
            // No concrete classes implement this shape. Emit an empty
            // Type entry so TypeId references are valid. The typetest
            // for this shape will always fail at runtime (no value of
            // this type can exist), making the code path unreachable.
            r.reification = Type << clone(r.id) << Union;
          }
          else if (union_node->size() == 1)
            r.reification = Type << clone(r.id) << union_node->front();
          else
            r.reification = Type << clone(r.id) << union_node;
        }
      }
    }

    // After resolve_shapes, prune TypeId members from unions in class fields
    // where the shape resolved to an empty union (no implementors). These
    // TypeIds were kept as "pending" during reify_type but now have definitive
    // empty-union reifications.
    void prune_empty_shape_unions()
    {
      auto is_empty_shape = [&](const Node& type_id) -> bool {
        for (auto& key : map_order)
        {
          for (auto& cr : map[key])
          {
            if (cr.id && same_reification_id(cr.id, type_id))
            {
              if (cr.reification && (cr.reification == Type))
              {
                auto& body = cr.reification->back();
                return (body == Union) && body->empty();
              }

              return false;
            }
          }
        }

        return true;
      };

      // Prune a union node in place, returning the simplified result.
      auto prune_union = [&](const Node& u) -> Node {
        Node result = Union;

        for (auto& child : *u)
        {
          if ((child == TypeId) && is_empty_shape(child))
            continue;

          result << clone(child);
        }

        if (result->empty())
          return Dyn;

        if (result->size() == 1)
          return result->front();

        return result;
      };

      for (auto& key : map_order)
      {
        for (auto& r : map[key])
        {
          if (!r.reification)
            continue;

          if (r.reification == Class)
          {
            auto fields = r.reification / Fields;

            for (auto& field : *fields)
            {
              auto& ft = field->back();

              if (ft == Union)
              {
                auto pruned = prune_union(ft);

                if (pruned != ft)
                  field->replace(ft, pruned);
              }
            }
          }
          else if (r.reification == Type)
          {
            auto& body = r.reification->back();

            if (body == Union)
            {
              auto pruned = prune_union(body);

              if (pruned != body)
                r.reification->replace(body, pruned);
            }
          }
        }
      }
    }

    Node resolve_receiver_typeid(const Node& type_id)
    {
      if (!type_id || (type_id != TypeId))
        return {};

      auto type_id_loc = type_id->location().view();

      for (auto& key : map_order)
      {
        for (auto& r : map[key])
        {
          if (!r.id || (r.id->location().view() != type_id_loc))
            continue;

          if (!r.reification && (r.def == TypeAlias))
            reify_typealias(r);

          if (r.reification && (r.reification == Type))
            return clone(r.reification->back());

          return {};
        }
      }

      return {};
    }

    Node resolve_reified_typeids(const Node& type)
    {
      if (!type)
        return {};

      if (type == TypeId)
      {
        auto resolved = resolve_receiver_typeid(type);
        return resolved ? resolve_reified_typeids(resolved) : clone(type);
      }

      if (type == Type)
        return Type << resolve_reified_typeids(type->front());

      if (type == Union)
      {
        Node result = Union;

        for (auto& child : *type)
          result << resolve_reified_typeids(child);

        return result;
      }

      if (type->in({Ref, Array, Cown}))
        return type->type() << resolve_reified_typeids(type->front());

      return clone(type);
    }

    // Extract individual type ids from a reified type. Dyn (or an unresolved
    // shape TypeId) means "all classes/primitives". A resolved shape TypeId is
    // expanded to its concrete implementations. Wrapper receivers are expanded
    // across the resolved inner receiver set.
    ReceiverSet extract_receivers(const Node& reified_type)
    {
      std::function<ReceiverSet(const Node&)> collect = [&](const Node& type) {
        if (!type || (type == Dyn))
          return ReceiverSet{true, {}};

        if (type == Type)
          return collect(type->front());

        if (type == TypeId)
        {
          auto resolved = resolve_receiver_typeid(type);
          return resolved ? collect(resolved) : ReceiverSet{true, {}};
        }

        if (type == Union)
        {
          ReceiverSet explicit_result{false, {}};
          ReceiverSet expanded_result{false, {}};
          bool saw_expanded = false;

          auto add_unique = [&](ReceiverSet& result, Node recv) {
            if (!result.contains(recv))
              result.types.push_back(clone(recv));
          };

          for (auto& child : *type)
          {
            auto child_set = collect(child);

            if (child_set.all)
              return ReceiverSet{true, {}};

            auto& result =
              contains_typeid(child) ? expanded_result : explicit_result;
            saw_expanded |= contains_typeid(child);

            for (auto& recv : child_set.types)
              add_unique(result, recv);
          }

          if (saw_expanded && !explicit_result.types.empty())
          {
            if (std::all_of(
                  explicit_result.types.begin(),
                  explicit_result.types.end(),
                  [&](auto& recv) { return expanded_result.contains(recv); }))
              return explicit_result;
          }

          for (auto& recv : expanded_result.types)
            add_unique(explicit_result, recv);

          return explicit_result;
        }

        if (type->in({Ref, Array, Cown}))
        {
          auto inner_set = collect(type->front());

          if (inner_set.all)
            return ReceiverSet{true, {}};

          ReceiverSet result{false, {}};

          for (auto& inner : inner_set.types)
          {
            Node wrapper = type->type() << clone(inner);
            result.types.push_back(wrapper);
          }

          return result;
        }

        return ReceiverSet{false, {clone(type)}};
      };

      return collect(reified_type);
    }

    // Check if a MethodInvocation targets a specific class reification.
    bool mi_targets(const MethodInvocation& mi, Node class_id)
    {
      return mi.receivers.contains(class_id);
    }

    // Resolve a TypeArg through the current substitution map. If the TypeArg
    // is a Type wrapping a TypeName that resolves to a TypeParam in the subst,
    // return the substituted value. Otherwise, return the original TypeArg.
    Node resolve_typearg(const Node& arg, const NodeMap<Node>& subst)
    {
      bool wrapped = (arg == Type);
      auto inner = arg;

      // Unwrap Type node.
      if (wrapped)
        inner = inner->front();

      if (inner == Union)
      {
        Node r = Union;
        bool has_non_dyn = false;

        for (auto& child : *inner)
        {
          auto resolved = resolve_typearg(child, subst);
          auto resolved_inner =
            (resolved == Type) ? resolved->front() : resolved;
          if (resolved_inner != Dyn)
            has_non_dyn = true;
          r << clone(resolved_inner);
        }

        if (has_non_dyn)
        {
          Node filtered = Union;
          for (auto& child : *r)
          {
            if (child != Dyn)
              filtered << clone(child);
          }
          r = filtered;
        }

        if (r->size() == 1)
          return wrapped ? (Type << clone(r->front())) : clone(r->front());

        return wrapped ? (Type << r) : r;
      }

      if (inner != TypeName)
        return arg;

      // Navigate the FQ name to see if the last element is a TypeParam.
      auto def = find_def(top, inner);

      if (!def)
        return arg;

      if (def == TypeParam)
      {
        // It's a TypeParam. Look it up in the subst map.
        auto find = subst.find(def);

        if (find != subst.end())
        {
          if (wrapped)
            return clone(find->second);

          return (find->second == Type) ? clone(find->second->front()) :
                                          clone(find->second);
        }

        // TypeParam not in subst — return Dyn to prevent self-referential
        // substitution cycles (where a TypeParam maps to a reference to
        // itself, causing infinite recursion in reify_type).
        return wrapped ? (Type << Dyn) : Dyn;
      }

      // Not a bare TypeParam. Recursively resolve TypeParams in any nested
      // TypeArgs (e.g., array[T] → array[i32] when T → i32 is in subst).
      bool changed = false;
      Node resolved_name = TypeName;

      for (auto& elem : *inner)
      {
        auto ta = elem / TypeArgs;

        if (ta->empty())
        {
          resolved_name << clone(elem);
          continue;
        }

        Node new_ta = TypeArgs;

        for (auto& a : *ta)
        {
          auto resolved = resolve_typearg(a, subst);

          if (resolved != a)
            changed = true;

          new_ta << clone(resolved);
        }

        resolved_name << (NameElement << clone(elem / Ident) << new_ta);
      }

      if (!changed)
        return arg;

      // Re-wrap in Type if the original was wrapped.
      if (wrapped)
        return Type << resolved_name;

      return resolved_name;
    }

    // Check whether two substitution maps are equivalent under invariance.
    bool subst_equal(const NodeMap<Node>& a, const NodeMap<Node>& b)
    {
      if (a.size() != b.size())
        return false;

      return std::equal(
        a.begin(), a.end(), b.begin(), b.end(), [&](auto& lhs, auto& rhs) {
          return (lhs.first == rhs.first) &&
            Subtype.invariant(top, lhs.second, rhs.second);
        });
    }

    // Compare two vectors of resolved types for invariant equality.
    bool typeargs_equal(const std::vector<Node>& a, const std::vector<Node>& b)
    {
      return std::equal(
        a.begin(), a.end(), b.begin(), b.end(), [&](auto& lhs, auto& rhs) {
          return Subtype.invariant(top, lhs, rhs);
        });
    }

    // Find or create a method reification index for the given base method id
    // and type arguments resolved through the call-site substitution.
    size_t find_method_index(
      const std::string& base_id,
      const Node& typeargs,
      const NodeMap<Node>& call_subst)
    {
      std::vector<Node> resolved;

      for (auto& ta : *typeargs)
        resolved.push_back((ta == Type) ? reify_type(ta, call_subst) : Dyn);

      auto& entries = method_index[base_id];

      for (size_t i = 0; i < entries.size(); i++)
      {
        if (typeargs_equal(entries[i], resolved))
          return i;
      }

      entries.push_back(std::move(resolved));
      return entries.size() - 1;
    }

    // Find an existing reification of def with the given subst (invariant
    // subtype equivalence), or create one, schedule it, and return its id.
    //
    // For primitive and wrapper builtins, dedup uses structural id equality
    // (make_id fully resolves element types without using the index). For all
    // other defs, dedup uses substitution map equality (the index embedded in
    // generic ClassId strings would vary per call, breaking id comparison).
    Node
    find_or_push(const Node& def, NodeMap<Node> subst, Node resolved_name = {})
    {
      auto it = map.find(def);
      bool is_new_key = (it == map.end());
      auto& r_vec = map[def];

      if (is_new_key)
        map_order.push_back(def);

      if (is_under_builtin(def))
      {
        auto name = (def / Ident)->location().view();

        if (
          primitive_types.find(name) != primitive_types.end() ||
          ffi_primitive_types.find(name) != ffi_primitive_types.end() ||
          wrapper_types.find(name) != wrapper_types.end())
        {
          // Primitives and wrappers: make_id produces index-free structural
          // ids (bare tokens or Wrapper << elem_type), so id equality works.
          auto id = make_id(def, r_vec.size(), subst);

          for (auto& existing : r_vec)
          {
            if (existing.id->equals(id))
            {
              // If the existing entry has an empty subst but the new call
              // has a non-empty one, update the existing entry's subst.
              // This happens when ensure_ref_reified creates a wrapper with
              // empty subst, and get_reification later provides proper subst.
              if (existing.subst.empty() && !subst.empty())
              {
                existing.subst = subst;

                // Re-register methods now that subst is available.
                if (existing.reification)
                {
                  for (auto& mi : method_invocations)
                  {
                    if (mi_targets(mi, existing.id))
                      register_method(mi, existing);
                  }
                }
              }

              if (!existing.resolved_name && resolved_name)
                existing.resolved_name = resolved_name;

              return clone(existing.id);
            }
          }

          r_vec.push_back(
            {def,
             std::move(subst),
             std::move(id),
             {},
             std::move(resolved_name)});
          worklist.push_back(&r_vec.back());
          return clone(r_vec.back().id);
        }
      }

      // All other defs: dedup using substitution map equality.
      // Compare entries for TypeParams owned by this def AND by the
      // enclosing class. Nested classes/shapes can reference the parent's
      // TypeParams in their own signatures/bodies, so different bindings for
      // the enclosing class must produce different reifications.
      auto own_tps = def / TypeParams;
      auto parent_cls = def->parent(ClassDef);
      auto parent_tps = parent_cls ? parent_cls / TypeParams : Node{};

      for (auto& existing : r_vec)
      {
        bool match = true;

        auto check_tp = [&](const Node& tp) {
          auto a_it = existing.subst.find(tp);
          auto b_it = subst.find(tp);

          if (a_it == existing.subst.end() && b_it == subst.end())
            return;

          if (a_it == existing.subst.end() || b_it == subst.end())
          {
            match = false;
            return;
          }

          if (!Subtype.invariant(top, a_it->second, b_it->second))
          {
            match = false;
          }
        };

        for (auto& tp : *own_tps)
        {
          check_tp(tp);
          if (!match)
            break;
        }

        if (match && parent_tps)
        {
          for (auto& tp : *parent_tps)
          {
            check_tp(tp);
            if (!match)
              break;
          }
        }

        if (match)
        {
          if (!existing.resolved_name && resolved_name)
            existing.resolved_name = resolved_name;
          return clone(existing.id);
        }
      }

      auto id = make_id(def, r_vec.size(), subst);

      r_vec.push_back(
        {def, std::move(subst), std::move(id), {}, std::move(resolved_name)});
      worklist.push_back(&r_vec.back());
      return clone(r_vec.back().id);
    }

    void reify_class(Reification& r)
    {
      // Shape reification is handled post-worklist: build a Type node
      // mapping the shape's TypeId to a Union of matching concrete classes.
      if ((r.def / Shape) == Shape)
        return;

      // Skip creation if already reified (e.g., early call from
      // reify_make_callback). Method invocation registration below
      // still runs.
      if (!r.reification)
      {
        if (r.id != ClassId)
        {
          // Primitive or wrapper type.
          r.reification = Primitive << clone(r.id) << Methods;
        }
        else
        {
          // User-defined class.
          Node fields = Fields;

          auto find_create_param_type = [&](const Node& field_def) -> Node {
            auto field_name = (field_def / Ident)->location().view();

            for (auto& child : *(r.def / ClassBody))
            {
              if (child != Function)
                continue;

              if ((child / Ident)->location().view() != "create")
                continue;

              auto* create_reif = find_function_reification(child, r.subst);

              if (!create_reif || !create_reif->reification)
                continue;

              auto params = create_reif->reification / Params;
              auto def_params = child / Params;

              for (size_t i = 0; i < def_params->size(); i++)
              {
                if (
                  (def_params->at(i) / Ident)->location().view() != field_name)
                  continue;

                return clone(params->at(i) / Type);
              }
            }

            return {};
          };

          auto has_create_param = [&](const Node& field_def) {
            auto field_name = (field_def / Ident)->location().view();

            for (auto& child : *(r.def / ClassBody))
            {
              if (child != Function)
                continue;

              if ((child / Ident)->location().view() != "create")
                continue;

              for (auto& param : *(child / Params))
              {
                if ((param / Ident)->location().view() == field_name)
                  return true;
              }
            }

            return false;
          };

          for (auto& f : *(r.def / ClassBody))
          {
            if (f != FieldDef)
              continue;

            auto field_type = find_create_param_type(f);

            if (!field_type)
            {
              field_type = has_create_param(f) ?
                reify_type(f / Type, r.subst) :
                reify_emitted_type(f / Type, r.subst, f / Ident, "field type");
            }

            fields << (Field << (FieldId ^ (f / Ident)) << field_type);
          }

          r.reification = Class << r.id << fields << Methods;
        }
      }

      // Register existing method invocations that target this class.
      for (auto& mi : method_invocations)
      {
        if (mi_targets(mi, r.id))
          register_method(mi, r);
      }

      // Register @final method if the class has a `final` function.
      for (auto& f : *(r.def / ClassBody))
      {
        if (f != Function)
          continue;

        if ((f / Ident)->location().view() != "final")
          continue;

        if ((f / Lhs)->type() != Rhs)
          continue;

        if (!((f / TypeParams)->empty()))
          continue;

        if ((f / Params)->size() != 1)
          continue;

        // Reify the final function and register as @final.
        auto funcid = find_or_push(f, r.subst);
        auto methods = r.reification / Methods;
        methods << (Method << (MethodId ^ "@final") << funcid);
        break;
      }
    }

    void reify_typealias(Reification& r)
    {
      // Store the reified type alias as a Type entry (TypeAlias is
      // resolved during reification; the IR uses Type for lookups).
      r.reification =
        Type << r.id
             << reify_emitted_type(
                  r.def / Type, r.subst, r.def / Ident, "type alias");
    }

    // Find a function reification by FunctionId and return its reified
    // return type. Returns empty Node if not found or not yet reified.
    Node find_func_return_type(const Node& funcid)
    {
      auto funcid_loc = funcid->location().view();

      for (auto& key : map_order)
      {
        if (key != Function)
          continue;

        for (auto& reif : map[key])
        {
          if (reif.id && (reif.id->location().view() == funcid_loc))
          {
            if (reif.reification)
              return clone(reif.reification / Type);

            // Not yet reified — eagerly compute the return type from the
            // function's ClassDef so that receiver-type tracking for
            // downstream Lookup/CallDyn sites doesn't fall back to the
            // conservative "all classes" receiver set.
            auto def_type = reif.def / Type;
            if (def_type->front() == TypeVar)
              return {};

            return reify_emitted_type(
              def_type, reif.subst, reif.def / Ident, "return type");
          }
        }
      }

      return {};
    }

    Reification* find_function_reification(const Node& funcid)
    {
      auto funcid_loc = funcid->location().view();

      for (auto& key : map_order)
      {
        if (key != Function)
          continue;

        for (auto& reif : map[key])
        {
          if (reif.id && (reif.id->location().view() == funcid_loc))
            return &reif;
        }
      }

      return nullptr;
    }

    Reification*
    find_function_reification(const Node& def, const NodeMap<Node>& subst)
    {
      for (auto& key : map_order)
      {
        if (key != Function)
          continue;

        for (auto& reif : map[key])
        {
          if ((reif.def == def) && subst_equal(reif.subst, subst))
            return &reif;
        }
      }

      return nullptr;
    }

    Reification* find_class_reification(const Node& classid)
    {
      auto classid_loc = classid->location().view();

      for (auto& key : map_order)
      {
        if (key != ClassDef)
          continue;

        for (auto& reif : map[key])
        {
          if (reif.id && (reif.id->location().view() == classid_loc))
            return &reif;
        }
      }

      return nullptr;
    }

    std::vector<Reification*> find_method_targets(
      Node recv_type,
      const std::string& method_id,
      const Node& args,
      bool behavior_args)
    {
      auto recv_set = extract_receivers(recv_type);
      std::vector<Reification*> targets;

      for (auto& key : map_order)
      {
        if (key != ClassDef)
          continue;

        for (auto& r : map[key])
        {
          if (!r.reification || !r.id)
            continue;

          if ((r.def / Shape) == Shape)
            continue;

          if (!recv_set.contains(r.id))
            continue;

          auto methods = r.reification / Methods;

          for (auto& m : *methods)
          {
            if ((m / MethodId)->location().view() != method_id)
              continue;

            auto* target = find_function_reification(m / FunctionId);

            if (
              target &&
              method_target_accepts_args(*target, args, behavior_args) &&
              std::none_of(targets.begin(), targets.end(), [&](auto* existing) {
                return existing == target;
              }))
            {
              targets.push_back(target);
            }

            break;
          }
        }
      }

      return targets;
    }

    Node merge_refined_type(Node current, Node actual)
    {
      if (!current)
        return clone(actual);

      if (!actual)
        return clone(current);

      if (current->equals(actual))
        return clone(current);

      if (current == Dyn)
        return clone(actual);

      if (actual == Dyn)
        return clone(current);

      Node merged = Union;

      auto add_type = [&](Node type) {
        auto add_one = [&](Node single) {
          for (auto& existing : *merged)
          {
            if (existing->equals(single))
              return;
          }

          merged << clone(single);
        };

        if (type == Union)
        {
          for (auto& child : *type)
            add_one(child);
        }
        else
        {
          add_one(type);
        }
      };

      add_type(current);
      add_type(actual);

      if (merged->size() == 1)
        return clone(merged->front());

      return merged;
    }

    Node behavior_arg_type(Node type)
    {
      if (!type)
        return {};

      if (type == Union)
      {
        Node converted = Union;

        auto add_unique = [&](Node candidate) {
          if (!candidate)
            return;

          auto add_one = [&](Node single) {
            for (auto& existing : *converted)
            {
              if (existing->equals(single))
                return;
            }

            converted << clone(single);
          };

          if (candidate == Union)
          {
            for (auto& child : *candidate)
              add_one(child);
          }
          else
          {
            add_one(candidate);
          }
        };

        for (auto& child : *type)
          add_unique(behavior_arg_type(child));

        if (converted->empty())
          return {};

        if (converted->size() == 1)
          return clone(converted->front());

        return converted;
      }

      if (type != Cown)
        return clone(type);

      auto inner = clone(type->front());

      if (inner != Dyn)
        ensure_ref_reified(inner);

      return Ref << inner;
    }

    Node actual_arg_type(const Node& arg, bool behavior_arg)
    {
      auto src = arg->back();
      auto it = local_types.find(src->location());

      if (it == local_types.end())
        return {};

      if (!behavior_arg)
        return clone(it->second);

      return behavior_arg_type(it->second);
    }

    bool method_target_accepts_args(
      Reification& target, const Node& args, bool behavior_args)
    {
      if (!target.reification)
        return false;

      auto params = target.reification / Params;

      if (params->size() != args->size())
        return false;

      for (size_t i = 1; i < args->size(); i++)
      {
        auto actual = actual_arg_type(args->at(i), behavior_args);

        if (!actual)
          continue;

        auto param = params->at(i) / Type;
        auto resolved_actual = resolve_reified_typeids(actual);
        auto resolved_param = resolve_reified_typeids(param);

        if (
          contains_dyn(resolved_actual) || contains_typeid(resolved_actual) ||
          contains_dyn(resolved_param) || contains_typeid(resolved_param))
          continue;

        if (!virc::IRSubtype(top, resolved_actual, resolved_param))
          return false;
      }

      return true;
    }

    Node receiver_type_from_targets(const std::vector<Reification*>& targets)
    {
      Node union_node = Union;

      auto add_unique = [&](Node type) {
        for (auto& existing : *union_node)
        {
          if (existing->equals(type))
            return;
        }

        union_node << clone(type);
      };

      for (auto* target : targets)
      {
        if (!target || !target->reification)
          continue;

        auto params = target->reification / Params;
        if (params->empty())
          continue;

        add_unique(params->front() / Type);
      }

      if (union_node->empty())
        return {};

      if (union_node->size() == 1)
        return clone(union_node->front());

      return union_node;
    }

    bool refine_receiver_type(
      const Node& func,
      const Location& recv_loc,
      Node& current_type,
      const std::vector<Reification*>& targets)
    {
      auto refined = receiver_type_from_targets(targets);

      if (!refined || current_type->equals(refined))
        return false;

      current_type = clone(refined);

      for (auto& param : *(func / Params))
      {
        if ((param / LocalId)->location() != recv_loc)
          continue;

        auto current = param / Type;
        if (!current->equals(refined))
          param->replace(current, clone(refined));
        break;
      }

      return true;
    }

    bool receiver_is_param(const Node& func, const Location& recv_loc) const
    {
      for (auto& param : *(func / Params))
      {
        if ((param / LocalId)->location() == recv_loc)
          return true;
      }

      return false;
    }

    bool refine_function_params(
      Reification& target, const Node& args, bool behavior_args)
    {
      if (!target.reification)
        return false;

      auto params = target.reification / Params;
      auto def_params = target.def / Params;

      if (params->size() != args->size())
        return false;

      bool changed = false;

      auto find_create_field_def = [&](const Node& def_param) -> Node {
        auto parent_cls = target.def->parent(ClassDef);
        if (!parent_cls)
          return {};
        if ((target.def / Ident)->location().view() != "create")
          return {};

        auto field_name = (def_param / Ident)->location().view();
        for (auto& child : *(parent_cls / ClassBody))
        {
          if (child != FieldDef)
            continue;
          if ((child / Ident)->location().view() != field_name)
            continue;
          return child;
        }

        return {};
      };

      auto sync_create_field_type =
        [&](const Node& def_param, const Node& refined_type) {
          auto parent_cls = target.def->parent(ClassDef);
          if (!parent_cls)
            return false;
          if ((target.def / Ident)->location().view() != "create")
            return false;

          auto class_id = target.reification / Type;
          if (class_id != ClassId)
            return false;

          auto* class_reif = find_class_reification(class_id);
          if (!class_reif || !class_reif->reification)
            return false;

          auto field_name = (def_param / Ident)->location().view();
          auto field_def = find_create_field_def(def_param);

          if (
            !field_def ||
            !(contains_typeparam_ref(field_def / Type) ||
              has_unresolved_type(field_def / Type, class_reif->subst)))
            return false;

          auto fields = class_reif->reification / Fields;
          for (auto& field : *fields)
          {
            if ((field / FieldId)->location().view() != field_name)
              continue;

            auto current_field = field / Type;
            auto refined = clone(refined_type);

            if (current_field->equals(refined))
              return false;

            field->replace(current_field, refined);
            return true;
          }

          return false;
        };

      for (size_t i = 0; i < args->size(); i++)
      {
        auto actual = actual_arg_type(args->at(i), behavior_args);

        if (!actual)
          continue;

        auto param = params->at(i);
        auto def_param = def_params->at(i);
        auto current = param / Type;
        bool generic_origin = contains_typeparam_ref(def_param / Type);
        auto unresolved_seed = reify_type(def_param / Type, target.subst);
        bool is_create = (target.def / Ident)->location().view() == "create";
        auto field_def = find_create_field_def(def_param);
        bool generic_create_field = field_def &&
          (contains_typeparam_ref(field_def / Type) ||
           has_unresolved_type(field_def / Type, target.subst));
        bool constructor_seed = is_create && generic_create_field &&
          contains_typeid(current) && !contains_typeid(actual) &&
          virc::IRSubtype(top, actual, current);
        bool replacing_seed = unresolved_seed &&
          current->equals(unresolved_seed) && current->in({TypeId, Union, Dyn});

        // A TypeId that was resolved from a class-level TypeParam is a valid
        // concrete type. Don't replace it — the class's subst already
        // determined the correct type. Method-level TypeParams should still
        // be refined from call-site argument types.
        if (replacing_seed && (current == TypeId) && generic_origin)
        {
          auto def_type = def_param / Type;
          auto tp_name = (def_type == Type) ? def_type->front() : def_type;

          if (tp_name == TypeName)
          {
            auto last_elem = tp_name->back();
            auto tp_ident = last_elem / Ident;
            auto parent_cls = target.def->parent(ClassDef);

            if (parent_cls)
            {
              for (auto& tp : *(parent_cls / TypeParams))
              {
                if ((tp / Ident)->location() == tp_ident->location())
                {
                  replacing_seed = false;
                  break;
                }
              }
            }
          }
        }

        if (
          !generic_origin &&
          !has_unresolved_type(def_param / Type, target.subst) &&
          !constructor_seed)
          continue;

        Node merged =
          (contains_dyn(current) || replacing_seed || constructor_seed) ?
          clone(actual) :
          merge_refined_type(current, actual);

        if (!current->equals(merged))
        {
          param->replace(current, merged);
          changed = true;
        }

        changed |= sync_create_field_type(def_param, merged);
      }

      return changed;
    }

    Node find_method_return_type(const std::vector<Reification*>& targets)
    {
      Nodes ret_types;

      for (auto* target : targets)
      {
        if (!target || !target->reification)
          continue;

        Node ret = target->reification / Type;

        if (!ret)
          continue;

        bool dup = false;

        for (auto& existing : ret_types)
        {
          if (existing->equals(ret))
          {
            dup = true;
            break;
          }
        }

        if (!dup)
          ret_types.push_back(ret);
      }

      if (ret_types.empty())
        return {};

      if (ret_types.size() == 1)
        return clone(ret_types.front());

      Node union_node = Union;

      for (auto& ret : ret_types)
        union_node << clone(ret);

      return union_node;
    }

    // Given a reified receiver type (possibly a union) and a MethodId string,
    // find the method's function return type by searching matching class
    // reifications. If multiple receivers contribute distinct return types,
    // return their union.
    Node find_method_return_type(Node recv_type, const std::string& method_id)
    {
      auto recv_set = extract_receivers(recv_type);
      Nodes ret_types;
      auto map_order_size = map_order.size();

      for (size_t i = 0; i < map_order_size; i++)
      {
        if (map_order[i] != ClassDef)
          continue;

        for (auto& r : map[map_order[i]])
        {
          if (!r.reification || !r.id)
            continue;

          if ((r.def / Shape) == Shape)
            continue;

          if (!recv_set.contains(r.id))
            continue;

          auto methods = r.reification / Methods;

          for (auto& m : *methods)
          {
            if ((m / MethodId)->location().view() == method_id)
            {
              auto ret = find_func_return_type(m / FunctionId);

              if (!ret)
                break;

              bool dup = false;

              for (auto& existing : ret_types)
              {
                if (existing->equals(ret))
                {
                  dup = true;
                  break;
                }
              }

              if (!dup)
                ret_types.push_back(ret);

              break;
            }
          }
        }
      }

      if (ret_types.empty())
        return {};

      if (ret_types.size() == 1)
        return clone(ret_types.front());

      Node union_node = Union;

      for (auto& ret : ret_types)
        union_node << clone(ret);

      return union_node;
    }

    bool reify_function(Reification& r)
    {
      // Clear per-function local type tracking.
      local_types.clear();
      lookup_info.clear();

      // Reify the function signature.
      auto def_type = r.def / Type;
      bool typevar_return = (def_type->front() == TypeVar);
      Node r_type;

      if (!typevar_return)
        r_type =
          reify_emitted_type(def_type, r.subst, r.def / Ident, "return type");
      else
        r_type = {}; // Will be inferred from Return terminals after body.
      Node params = Params;

      for (auto& p : *(r.def / Params))
      {
        auto p_type = reify_type(p / Type, r.subst);
        local_types[(p / Ident)->location()] = p_type;
        params << (Param << (LocalId ^ (p / Ident)) << p_type);
      }

      // Reify the function body.
      Node vars = Vars;
      std::vector<Location> var_locs;
      Node labels = clone(r.def / Labels);

      for (auto& l : *labels)
      {
        Node body = l / Body;
        Nodes remove;
        Nodes splat_expand;

        // No structural changes required: CallDyn, math ops on existing
        // values.

        body->traverse([&](Node& n) {
          if (n == body)
            return true;

          if (n->in({Const, Convert}))
          {
            reify_primitive(n / Type);
            // Track type: Const/Convert produce the primitive type token.
            local_types[(n / LocalId)->location()] = clone(n / Type);
          }
          else if (n == ConstStr)
          {
            Node u8_type = TypeName
              << (NameElement << (Ident ^ "_builtin") << TypeArgs)
              << (NameElement << (Ident ^ "u8") << TypeArgs);
            ensure_array_reified(u8_type, {});
            local_types[(n / LocalId)->location()] = Array << U8;
          }
          else if (n->in({Eq, Ne, Lt, Le, Gt, Ge}))
          {
            reify_primitive(Bool);
            local_types[(n / LocalId)->location()] = Bool;
          }
          else if (n->in({Const_E, Const_Pi, Const_Inf, Const_NaN}))
          {
            reify_primitive(F64);
            local_types[(n / LocalId)->location()] = F64;
          }
          else if (n == Bits)
          {
            reify_primitive(U64);
            local_types[(n / LocalId)->location()] = U64;
          }
          else if (n == Len)
          {
            reify_primitive(USize);
            local_types[(n / LocalId)->location()] = USize;
          }
          else if (n == MakePtr)
          {
            reify_primitive(Ptr);
            local_types[(n / LocalId)->location()] = Ptr;
          }
          else if (n == MakeCallback)
          {
            reify_primitive(Ptr);
            local_types[(n / LocalId)->location()] = Ptr;

            // Find the lambda's type and register its @callback method.
            // First check local_types (works when source was from New or from a
            // typed function wrapper that may need deferred shape resolution).
            auto src_loc = (n / Rhs)->location();
            auto src_it = local_types.find(src_loc);
            Node callback_type;

            if (src_it != local_types.end())
              callback_type = clone(src_it->second);

            if (!callback_type)
            {
              // local_types doesn't have it (e.g., assigned via Call).
              // Trace back through Copy/Move to find the original source.
              auto body_node = n->parent();

              // Pass 1: follow Copy/Move chain to root source.
              auto trace_loc = src_loc;
              bool changed = true;

              while (changed)
              {
                changed = false;

                for (auto& stmt : *body_node)
                {
                  if (&stmt == &n)
                    break;

                  if (
                    stmt->in({Copy, Move}) &&
                    ((stmt / LocalId)->location() == trace_loc))
                  {
                    trace_loc = (stmt / Rhs)->location();
                    changed = true;
                  }
                }
              }

              // Pass 2: find the definition of the root source.
              Node call_enc;

              for (auto& stmt : *body_node)
              {
                if (&stmt == &n)
                  break;

                if (
                  stmt->in({New, Stack}) &&
                  ((stmt / LocalId)->location() == trace_loc))
                {
                  callback_type = clone(stmt / ClassId);
                }
                else if (
                  (stmt == MemoSlot) &&
                  ((stmt / LocalId)->location() == trace_loc))
                {
                  callback_type = empty_class_type(stmt / MemoId);
                }
                else if (
                  (stmt == Call) && ((stmt / LocalId)->location() == trace_loc))
                {
                  // Find the Function reification for this Call, then
                  // trigger reification of its enclosing ClassDef.
                  auto funcid_loc = (stmt / FunctionId)->location().view();
                  NodeMap<Node> class_subst;

                  for (auto& key : map_order)
                  {
                    if (key != Function)
                      continue;

                    for (auto& reif : map[key])
                    {
                      if (reif.id && (reif.id->location().view() == funcid_loc))
                      {
                        auto enc = reif.def->parent(ClassDef);

                        if (enc)
                        {
                          // Extract the class's TypeParam substitutions
                          // from the function's substitution context.
                          for (auto& tp : *(enc / TypeParams))
                          {
                            auto sit = reif.subst.find(tp);

                            if (sit != reif.subst.end())
                              class_subst[sit->first] = clone(sit->second);
                          }

                          call_enc = enc;
                        }

                        break;
                      }
                    }

                    if (call_enc)
                      break;
                  }

                  // Trigger class reification AFTER the map_order loop
                  // to avoid iterator invalidation (find_or_push may
                  // append to map_order).
                  if (call_enc)
                    callback_type =
                      find_or_push(call_enc, std::move(class_subst));
                }
              }
            }

            if (callback_type)
              reify_make_callback(n, callback_type);
            else
              n->parent()->replace(
                n, err(n, "make_callback: cannot determine lambda type"));
          }
          else if (n == CodePtrCallback)
          {
            reify_primitive(Ptr);
            local_types[(n / LocalId)->location()] = Ptr;
          }
          else if (n->in({
                     FreeCallback,
                     Pin,
                     Unpin,
                     Merge,
                     AddExternal,
                     RemoveExternal,
                   }))
          {
            reify_primitive(None);
            local_types[(n / LocalId)->location()] = None;
          }
          else if (n == Freeze)
          {
            // Propagate type from source to destination.
            auto src_it = local_types.find((n / Rhs)->location());

            if (src_it != local_types.end())
              local_types[(n / LocalId)->location()] = clone(src_it->second);
          }
          else if (n->in({ArrayCopy, ArrayFill}))
          {
            reify_primitive(None);
            local_types[(n / LocalId)->location()] = None;
          }
          else if (n == ArrayCompare)
          {
            reify_primitive(I64);
            local_types[(n / LocalId)->location()] = I64;
          }
          else if (n == TypedMove)
          {
            auto dst_loc = (n / LocalId)->location();
            auto type = reify_type(n / Type, r.subst);
            n->replace(n / Type, type);
            local_types[dst_loc] = clone(type);
          }
          else if (n->in({Copy, Move}))
          {
            // Propagate type from source to destination.
            auto src_it = local_types.find((n / Rhs)->location());

            if (src_it != local_types.end())
              local_types[(n / LocalId)->location()] = clone(src_it->second);
          }
          else if (n == RegisterRef)
          {
            // RegisterRef: dst = &src. Result type is Ref << type(src).
            auto src_it = local_types.find((n / Rhs)->location());

            if (src_it != local_types.end())
            {
              ensure_ref_reified(src_it->second);
              local_types[(n / LocalId)->location()] = Ref
                << clone(src_it->second);
            }
          }
          else if (n == FieldRef)
          {
            // FieldRef: dst = &(arg.field). Result type is
            // Ref << reified field type.
            auto obj_loc = (n / Arg / Rhs)->location();
            auto obj_it = local_types.find(obj_loc);

            if (obj_it != local_types.end() && (obj_it->second == ClassId))
            {
              auto ft = find_field_type(obj_it->second, n / FieldId);

              if (ft)
              {
                ensure_ref_reified(ft);
                local_types[(n / LocalId)->location()] = Ref << ft;
              }
            }
          }
          else if (n->in({ArrayRef, ArrayRefConst}))
          {
            // ArrayRef/ArrayRefConst: dst = &(arr[i]). Result type is
            // Ref << element type.
            auto arr_loc = (n / Arg / Rhs)->location();
            auto arr_it = local_types.find(arr_loc);

            if (arr_it != local_types.end())
            {
              Node elem;

              if (arr_it->second == TupleType && n == ArrayRefConst)
              {
                // TupleType is a peer of Array: extract per-element type
                // by constant index.
                auto idx = from_chars_sep_v<size_t>(n / Rhs);

                if (idx < arr_it->second->size())
                  elem = clone(arr_it->second->at(idx));
                else
                  elem = Dyn;
              }
              else if (arr_it->second == TupleType)
              {
                // Runtime-indexed access on a TupleType: element type is
                // unknown at compile time.
                elem = Dyn;
              }
              else if (arr_it->second == Array)
              {
                elem = clone(arr_it->second->front());
              }

              if (elem)
              {
                ensure_ref_reified(elem);
                local_types[(n / LocalId)->location()] = Ref << elem;
              }
            }
          }
          else if (n == ArrayRefFromEnd)
          {
            // Compute element type and collect for post-traversal expansion.
            auto arr_loc = (n / Arg / Rhs)->location();
            auto arr_it = local_types.find(arr_loc);

            if (arr_it != local_types.end() && arr_it->second == TupleType)
            {
              auto from_end = from_chars_sep_v<size_t>(n / Rhs);
              auto arity = arr_it->second->size();

              if (from_end >= 1 && from_end <= arity)
              {
                auto real_idx = arity - from_end;
                auto elem = clone(arr_it->second->at(real_idx));
                ensure_ref_reified(elem);
                local_types[(n / LocalId)->location()] = Ref << elem;
              }
            }

            splat_expand.push_back(n);
          }
          else if (n == SplatOp)
          {
            // Compute result type and collect for post-traversal expansion.
            auto arr_loc = (n / Arg / Rhs)->location();
            auto arr_it = local_types.find(arr_loc);

            if (arr_it != local_types.end() && arr_it->second == TupleType)
            {
              auto before = from_chars_sep_v<size_t>(n / Lhs);
              auto after = from_chars_sep_v<size_t>(n / Rhs);
              auto arity = arr_it->second->size();

              if (before + after <= arity)
              {
                auto remaining = arity - before - after;

                if (remaining == 0)
                {
                  reify_primitive(clone(None));
                  local_types[(n / LocalId)->location()] = clone(None);
                }
                else if (remaining == 1)
                {
                  local_types[(n / LocalId)->location()] =
                    clone(arr_it->second->at(before));
                }
                else
                {
                  Node ttype = TupleType;

                  for (size_t i = before; i < before + remaining; i++)
                    ttype << clone(arr_it->second->at(i));

                  local_types[(n / LocalId)->location()] = clone(ttype);
                }
              }
            }

            splat_expand.push_back(n);
          }
          else if (n == Load)
          {
            // Load: dst = *src. Unwrap Ref to get inner type.
            auto src_it = local_types.find((n / Rhs)->location());

            if (src_it != local_types.end() && (src_it->second == Ref))
              local_types[(n / LocalId)->location()] =
                clone(src_it->second->front());
          }
          else if (n == Store)
          {
            // Store: dst = old *src, *src = arg. Result is old value type.
            auto src_it = local_types.find((n / Rhs)->location());

            if (src_it != local_types.end() && (src_it->second == Ref))
              local_types[(n / LocalId)->location()] =
                clone(src_it->second->front());
          }
          else if (n == Var)
          {
            var_locs.push_back((n / Ident)->location());
            remove.push_back(n);
          }
          else if (n->in({New, Stack}))
          {
            // Save the type before reify_new transforms the node.
            auto new_type = reify_emitted_type(
              n / Type, r.subst, n / Type, "constructed type");
            reify_new(n, r.subst);
            // After reify_new, dst is first child.
            auto dst_loc = (n / LocalId)->location();
            local_types[dst_loc] = new_type;

            // Empty class: load its memoized instance.
            if ((n / Args)->empty())
            {
              auto s = MemoSlot << clone(n / LocalId)
                                << ensure_empty_class_memo(n / ClassId);
              n->parent()->replace(n, s);
              n = s;
            }
          }
          else if (n == Lookup)
          {
            // Save receiver location before reify_lookup transforms the node.
            auto recv_loc = (n / Rhs)->location();
            reify_lookup(n, r.subst, r.id);
            // After reify_lookup: Lookup << dst << src << MethodId.
            auto mid = (n / MethodId)->location().view();
            lookup_info[(n / LocalId)->location()] = {
              std::string(mid), recv_loc};
          }
          else if (n == Call)
          {
            reify_call(n, r.subst);
            if (n == Call)
              record_call_edge(r.id, n / FunctionId, n);

            // Track Call return type from the function reification.
            auto ret = find_func_return_type(n / FunctionId);
            if (ret)
              local_types[(n / LocalId)->location()] = ret;
          }
          else if (n->in({CallDyn, TryCallDyn}))
          {
            // Track CallDyn return type by resolving the method on
            // the receiver's reified class.
            auto src_loc = (n / Rhs)->location();
            auto li = lookup_info.find(src_loc);
            if (li != lookup_info.end())
            {
              auto recv_it = local_types.find(li->second.recv_loc);
              if (recv_it != local_types.end())
              {
                auto ret = find_method_return_type(
                  recv_it->second, li->second.method_id);

                // Structural fallback for ref deref: ref[X].*(ref[X])
                // returns X. This handles cases where ensure_ref_reified
                // created the ref wrapper without proper subst for method
                // registration.
                if (!ret && recv_it->second == Ref)
                {
                  auto& mid = li->second.method_id;
                  if (mid.starts_with("*::"))
                    ret = clone(recv_it->second->front());
                }

                if (ret)
                  local_types[(n / LocalId)->location()] = ret;
              }
            }
          }
          else if (n == NewArray)
          {
            auto arr_type =
              Array << reify_emitted_type(
                n / Type, r.subst, n / Type, "array element type");
            local_types[(n / LocalId)->location()] = clone(arr_type);
            n / Type = arr_type;
          }
          else if (n == NewArrayConst)
          {
            auto inner = reify_emitted_type(
              n / Type, r.subst, n / Type, "array element type");

            if (inner == TupleType)
            {
              // TupleType is a peer of Array, not wrapped in it.
              local_types[(n / LocalId)->location()] = clone(inner);
              n / Type = inner;
            }
            else
            {
              // Save original Type before reification overwrites it.
              // ensure_array_reified needs the TypeName form for subst.
              auto orig_type = clone(n / Type);
              auto arr_type = Array << inner;
              local_types[(n / LocalId)->location()] = clone(arr_type);
              n / Type = arr_type;

              // For array literals, trigger reification of the array class
              // so method invocations (size, apply) can be resolved.
              auto loc_view = (n / LocalId)->location().view();
              bool is_array_lit =
                loc_view.size() >= 5 && loc_view.substr(0, 5) == "array";

              if (is_array_lit)
                ensure_array_reified(orig_type, r.subst);
            }
          }
          else if (n == FFI)
          {
            reify_ffi(n, r);
          }
          else if (n == FFIStruct)
          {
            auto layout_type = reify_emitted_type(
              n / Type, r.subst, n / Type, "FFI layout type");
            auto result_type = reify_emitted_type(
              ffi_struct_result_type(),
              r.subst,
              n / Type,
              "FFI struct result type");
            n / Type = layout_type;
            local_types[(n / LocalId)->location()] = clone(result_type);
          }
          else if (n == FFILoad)
          {
            auto field_type =
              reify_emitted_type(n / Type, r.subst, n / Type, "FFI field type");
            n / Type = field_type;
            local_types[(n / LocalId)->location()] = clone(field_type);
          }
          else if (n == FFIStore)
          {
            auto field_type =
              reify_emitted_type(n / Type, r.subst, n / Type, "FFI field type");
            reify_primitive(None);
            n / Type = field_type;
            local_types[(n / LocalId)->location()] = None;
          }
          else if (n == When)
          {
            n->parent()->replace(n, reify_when(n, r));
          }
          else if (n == Typetest)
          {
            n / Type = reify_type(n / Type, r.subst);
          }

          return false;
        });

        for (auto& n : remove)
          n->parent()->replace(n);

        // Expand ArrayRefFromEnd and SplatOp nodes.
        for (auto& n : splat_expand)
        {
          if (n == ArrayRefFromEnd)
          {
            // Convert to ArrayRefConst with computed index.
            auto arr_loc = (n / Arg / Rhs)->location();
            auto arr_it = local_types.find(arr_loc);

            if (arr_it != local_types.end() && arr_it->second == TupleType)
            {
              auto from_end = from_chars_sep_v<size_t>(n / Rhs);
              auto arity = arr_it->second->size();
              auto real_idx = arity - from_end;

              Node replacement = ArrayRefConst
                << clone(n / LocalId) << clone(n / Arg)
                << (Int ^ std::to_string(real_idx));

              body->replace(n, replacement);
            }
            else
            {
              assert(false && "ArrayRefFromEnd source must be TupleType");
            }
          }
          else if (n == SplatOp)
          {
            auto arr_loc = (n / Arg / Rhs)->location();
            auto arr_it = local_types.find(arr_loc);

            if (arr_it != local_types.end() && arr_it->second == TupleType)
            {
              auto before = from_chars_sep_v<size_t>(n / Lhs);
              auto after = from_chars_sep_v<size_t>(n / Rhs);
              auto arity = arr_it->second->size();

              if (before + after > arity)
              {
                body->replace(
                  n,
                  err(
                    n,
                    "tuple has " + std::to_string(arity) +
                      " elements, but destructuring requires at least " +
                      std::to_string(before + after)));
                continue;
              }

              auto remaining = arity - before - after;
              auto dst_loc = (n / LocalId)->location();

              if (remaining == 0)
              {
                // No remaining elements: produce a None constant.
                Node replacement = Const << (LocalId ^ dst_loc) << clone(None)
                                         << clone(None);

                body->replace(n, replacement);
              }
              else if (remaining == 1)
              {
                // One element: ArrayRefConst + Load.
                auto ref_loc = top->fresh(Location("splat"));

                Node aref = ArrayRefConst << (LocalId ^ ref_loc)
                                          << clone(n / Arg)
                                          << (Int ^ std::to_string(before));

                Node load = Load << (LocalId ^ dst_loc) << (LocalId ^ ref_loc);

                Nodes replacements = {aref, load};
                auto it = body->find(n);
                auto pos = body->erase(it, std::next(it));
                body->insert(pos, replacements.begin(), replacements.end());
              }
              else
              {
                // Two or more elements: create a new tuple.
                Node ttype = TupleType;

                for (size_t i = before; i < before + remaining; i++)
                  ttype << clone(arr_it->second->at(i));

                Nodes replacements;

                // NewArrayConst to allocate the tuple.
                replacements.push_back(
                  NewArrayConst << (LocalId ^ dst_loc) << clone(ttype)
                                << (Int ^ std::to_string(remaining)));

                // Copy each element from source to destination.
                for (size_t i = 0; i < remaining; i++)
                {
                  auto src_ref = top->fresh(Location("splat"));
                  auto val_loc = top->fresh(Location("splat"));
                  auto dst_ref = top->fresh(Location("splat"));
                  auto old_val = top->fresh(Location("splat"));

                  // Get ref to source element.
                  replacements.push_back(
                    ArrayRefConst << (LocalId ^ src_ref) << clone(n / Arg)
                                  << (Int ^ std::to_string(before + i)));

                  // Load source value.
                  replacements.push_back(
                    Load << (LocalId ^ val_loc) << (LocalId ^ src_ref));

                  // Get ref to destination element.
                  replacements.push_back(
                    ArrayRefConst << (LocalId ^ dst_ref)
                                  << (Arg << ArgCopy << (LocalId ^ dst_loc))
                                  << (Int ^ std::to_string(i)));

                  // Store value into destination.
                  replacements.push_back(
                    Store << (LocalId ^ old_val) << (LocalId ^ dst_ref)
                          << (Arg << ArgCopy << (LocalId ^ val_loc)));
                }

                auto it = body->find(n);
                auto pos = body->erase(it, std::next(it));
                body->insert(pos, replacements.begin(), replacements.end());
              }
            }
            else
            {
              assert(false && "SplatOp source must be TupleType");
            }
          }
        }

        // Reify the Type child of Raise terminators.
        auto term = l / Return;

        if (term == Raise)
          term / Type = reify_type(term / Type, r.subst);
      }

      // Infer TypeVar return type from Return terminals after body
      // processing. By now, local_types has been populated for all
      // statements in the body. Collect all distinct return types and
      // build a union if there are multiple.
      if (typevar_return)
      {
        Nodes ret_types;

        for (auto& l : *labels)
        {
          auto term = l / Return;

          if (term != Return)
            continue;

          auto ret_loc = (term / LocalId)->location();
          auto it = local_types.find(ret_loc);

          if (it == local_types.end())
            continue;

          // Check if this type is already covered.
          bool dup = false;

          for (auto& existing : ret_types)
          {
            if (existing->equals(it->second))
            {
              dup = true;
              break;
            }
          }

          if (!dup)
            ret_types.push_back(clone(it->second));
        }

        if (ret_types.size() == 1)
        {
          r_type = ret_types.front();
        }
        else if (ret_types.size() > 1)
        {
          Node union_node = Union;

          for (auto& rt : ret_types)
            union_node << clone(rt);

          r_type = union_node;
        }

        // If we still don't have a type, check if all exits are
        // Raise/Jump — the function never returns normally.
        if (!r_type)
        {
          bool all_nonlocal = true;

          for (auto& l : *labels)
          {
            auto term = l / Return;

            if (term->in({Jump, Cond}))
              continue;

            if (term != Raise)
            {
              all_nonlocal = false;
              break;
            }
          }

          if (all_nonlocal)
          {
            reify_primitive(clone(None));
            r_type = clone(None);
          }
        }

        // Last resort: mark as Dyn and let the deferred second pass try to
        // refine it once all callees are reified.
        if (!r_type)
          r_type = Dyn;
      }

      // Implicit none return: when a function returns none, append a
      // none constant and use it as the return value for any Return
      // terminator. This lets users omit the trailing `none` in
      // functions and lambdas that return none.
      if (r_type->type() == None)
      {
        for (auto& l : *labels)
        {
          auto term = l / Return;

          if (term != Return)
            continue;

          auto none_loc = top->fresh(Location("none"));
          auto body = l / Body;
          body << (Const << (LocalId ^ none_loc) << None << None);
          term->replace(term->front(), LocalId ^ none_loc);
        }
      }

      // Build VarDef nodes with types from local_types.
      for (auto& loc : var_locs)
      {
        auto it = local_types.find(loc);
        Node var_type = (it != local_types.end()) ? clone(it->second) : Dyn;
        vars << (VarDef << (LocalId ^ loc) << var_type);
      }

      r.reification = Func << r.id << params << r_type << vars << labels;

      if ((r.def / Lhs) == Once)
        once_funcs.insert(r.id->location());

      // Ensure every AtTeardown callback has @callback registered so the
      // runtime can invoke it.
      for (auto& l : *labels)
      {
        auto body_node = l / Body;

        for (auto& teardown : *body_node)
        {
          if (teardown != AtTeardown)
            continue;

          auto ret_loc = (teardown / LocalId)->location();

          // Trace through Copy/Move to find the original source.
          bool changed = true;

          while (changed)
          {
            changed = false;

            for (auto& stmt : *body_node)
            {
              if (
                stmt->in({Copy, Move}) &&
                ((stmt / LocalId)->location() == ret_loc))
              {
                ret_loc = (stmt / Rhs)->location();
                changed = true;
                break;
              }
            }
          }

          // Check local_types first (works for New/Stack and typed function
          // wrappers that resolve to a concrete lambda class).
          auto it = local_types.find(ret_loc);

          if (it != local_types.end())
          {
            if (register_callback_type(it->second, {}, false, r.id, teardown))
              continue;
          }

          // Check if the callback comes from a Call (e.g., create).
          for (auto& stmt : *body_node)
          {
            if ((stmt == Call) && ((stmt / LocalId)->location() == ret_loc))
            {
              // Find the Function reification and its enclosing ClassDef.
              auto funcid_loc = (stmt / FunctionId)->location().view();
              Node call_enc;
              NodeMap<Node> class_subst;

              for (auto& key : map_order)
              {
                if (key != Function)
                  continue;

                for (auto& reif : map[key])
                {
                  if (reif.id && (reif.id->location().view() == funcid_loc))
                  {
                    auto enc = reif.def->parent(ClassDef);

                    if (enc)
                    {
                      for (auto& tp : *(enc / TypeParams))
                      {
                        auto sit = reif.subst.find(tp);

                        if (sit != reif.subst.end())
                          class_subst[sit->first] = clone(sit->second);
                      }

                      call_enc = enc;
                    }

                    break;
                  }
                }

                if (call_enc)
                  break;
              }

              if (call_enc)
              {
                auto class_id = find_or_push(call_enc, std::move(class_subst));
                register_callback_type(class_id, {}, false, r.id, teardown);
              }

              break;
            }
          }
        }
      }

      return true;
    }

    // Turn a type into an IR type. The IR doesn't have intersection types,
    // structural types, or tuple types.
    Node reify_type(const Node& type, const NodeMap<Node>& subst)
    {
      if (type == Type)
        return reify_type(type->front(), subst);

      // Already-reified IR type. Return a clone so the caller can safely
      // insert it into a new part of the AST.
      if (type == Dyn)
        return clone(type);

      if (type->in(
            {ClassId,
             TypeId,
             None,
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
             Ptr}))
      {
        return clone(type);
      }

      if (type == Array)
        return Array << reify_type(type->front(), subst);

      if (type == Ref)
        return Ref << reify_type(type->front(), subst);

      if (type == Cown)
        return Cown << reify_type(type->front(), subst);

      // TypeVar that wasn't resolved during inference (e.g., in a generic
      // context). Return Dyn as a fallback — the caller should handle this
      // or the second pass will resolve it.
      if (type == TypeVar)
        return Dyn;

      // Preserve TupleType with reified element types.
      if (type == TupleType)
      {
        Node r = TupleType;

        for (auto& child : *type)
          r << reify_type(child, subst);

        return r;
      }

      if (type == Union)
      {
        Node r = Union;

        for (auto& t : *type)
        {
          auto rt = reify_type(t, subst);

          // A union that contains a dynamic type is just dynamic. A union that
          // contains a union is flattened. TypeId entries for shapes with no
          // implementors are dropped (the code paths requiring them are
          // unreachable).
          if (rt == Dyn)
            return Dyn;
          else if (rt == Union)
            r << *rt;
          else if (rt == TypeId)
          {
            // Check if this TypeId corresponds to a shape with no
            // implementors. If the shape's reification hasn't been emitted
            // yet (null), keep it — resolve_shapes may add implementors
            // later. If the shape has been resolved to an empty union,
            // drop it. The post-shape cleanup pass (prune_empty_shape_unions)
            // handles any TypeIds that turn out to be empty after
            // resolve_shapes runs.
            auto has_empty_shape = [&](const Node& tid) -> bool {
              for (auto& key : map_order)
              {
                for (auto& cr : map[key])
                {
                  if (cr.id && same_reification_id(cr.id, tid))
                  {
                    if (cr.reification && (cr.reification == Type))
                    {
                      auto& body = cr.reification->back();
                      return (body == Union) && body->empty();
                    }

                    // Pending (null) or non-Type reification — keep.
                    return false;
                  }
                }
              }

              // Not in the map at all — treat as empty.
              return true;
            };

            if (!has_empty_shape(rt))
              r << rt;
          }
          else
            r << rt;
        }

        if (r->empty())
          return Dyn;

        if (r->size() == 1)
          return r->front();

        return r;
      }

      if (type == Isect)
      {
        Node r = Dyn;

        for (auto& t : *type)
        {
          auto rt = reify_type(t, subst);

          // Encapsulate rt in a union.
          if (rt != Union)
            rt = Union << rt;

          if (r == Dyn)
          {
            // A dynamic result means all types, so the intersection is rt.
            r = rt;
          }
          else
          {
            // Intersect the existing union with this one.
            Node nr = Union;

            for (auto& existing : *r)
            {
              // Keep this existing type if it also exists in rt. Dynamic types
              // in the intersection are ignored.
              bool found = std::any_of(rt->begin(), rt->end(), [&](auto& c) {
                return (c != Dyn) && existing->equals(c);
              });

              // Keep only unique types.
              if (found && std::none_of(nr->begin(), nr->end(), [&](auto& u) {
                    return u->equals(existing);
                  }))
              {
                nr << existing;
              }
            }

            r = nr;
          }
        }

        return r;
      }

      if (type == TypeName)
        return reify_typename(type, subst);

      assert(false);
      return {};
    }

    // Get the reification and return the ClassId or TypeId.
    Node reify_typename(const Node& tn, const NodeMap<Node>& subst)
    {
      return get_reification(
        tn, subst, [](auto& def) { return def->in({ClassDef, TypeAlias}); });
    }

    // Ensure a primitive type is reified. Delegates to find_or_push which
    // deduplicates and schedules via the worklist.
    void reify_primitive(const Node& type)
    {
      // Check flat primitives (_builtin::name).
      for (auto& [k, v] : primitive_types)
      {
        if (type->type() != v->type())
          continue;

        auto defs = builtin->look(Location(std::string(k)));
        assert(defs.size() == 1);

        Node prim_name = TypeName
          << (NameElement << (Ident ^ "_builtin") << TypeArgs)
          << (NameElement << (Ident ^ std::string(k)) << TypeArgs);

        find_or_push(defs.front(), {}, prim_name);
        return;
      }

      // Check ffi primitives (_builtin::ffi::name).
      for (auto& [k, v] : ffi_primitive_types)
      {
        if (type->type() != v->type())
          continue;

        auto ffi_defs = builtin->look(Location("ffi"));
        assert(ffi_defs.size() == 1);
        auto ffi_def = ffi_defs.front();

        auto defs = ffi_def->lookdown(Location(std::string(k)));
        assert(defs.size() == 1);

        Node prim_name = TypeName
          << (NameElement << (Ident ^ "_builtin") << TypeArgs)
          << (NameElement << (Ident ^ "ffi") << TypeArgs)
          << (NameElement << (Ident ^ std::string(k)) << TypeArgs);

        find_or_push(defs.front(), {}, prim_name);
        return;
      }
    }

    // Look up a field's reified type from a ClassId.  Finds the Reification
    // matching `classid`, locates the FieldDef by name, and reifies the field
    // type using the class's substitution map.
    Node find_field_type(Node classid, const Node& field_id)
    {
      auto field_name = field_id->location().view();

      for (auto& key : map_order)
      {
        for (auto& r : map[key])
        {
          if (!same_reification_id(r.id, classid) || (r.def != ClassDef))
            continue;

          if (r.reification)
          {
            auto fields = r.reification / Fields;
            for (auto& field : *fields)
            {
              if ((field / FieldId)->location().view() != field_name)
                continue;
              return clone(field / Type);
            }
          }

          for (auto& f : *(r.def / ClassBody))
          {
            if (f != FieldDef)
              continue;

            if ((f / Ident)->location().view() != field_name)
              continue;
            return reify_type(f / Type, r.subst);
          }

          return {};
        }
      }

      return {};
    }

    // Ensure that a Ref wrapper primitive with the given inner IR type is
    // reified.  Checks for an existing entry by structural id equality and
    // creates one via the worklist if absent.
    void ensure_ref_reified(const Node& inner_ir_type)
    {
      if (!inner_ir_type || (inner_ir_type == Dyn))
        return;

      auto ref_defs = builtin->look(Location("ref"));
      assert(!ref_defs.empty());
      auto ref_def = ref_defs.front();
      auto tps = ref_def / TypeParams;
      assert(tps->size() == 1);

      Node expected_id = Ref << clone(inner_ir_type);
      NodeMap<Node> subst;
      subst[tps->at(0)] = clone(inner_ir_type);

      auto it = map.find(ref_def);
      bool is_new_key = (it == map.end());
      auto& r_vec = map[ref_def];

      if (is_new_key)
        map_order.push_back(ref_def);

      for (auto& existing : r_vec)
      {
        if (!existing.id->equals(expected_id))
          continue;

        if (existing.subst.empty())
        {
          existing.subst = subst;

          if (existing.reification)
          {
            for (auto& mi : method_invocations)
            {
              if (mi_targets(mi, existing.id))
                register_method(mi, existing);
            }
          }
        }

        return;
      }

      r_vec.push_back(
        {ref_def, std::move(subst), std::move(expected_id), {}, {}});
      worklist.push_back(&r_vec.back());
    }

    // Ensure that the array wrapper class is reified for a given element type.
    // Called from array literal processing so that method invocations (size,
    // apply) have a class reification to bind against.
    void ensure_array_reified(
      const Node& elem_type, const NodeMap<Node>& outer_subst)
    {
      auto array_defs = builtin->look(Location("array"));
      assert(!array_defs.empty());
      auto array_def = array_defs.front();

      auto tps = array_def / TypeParams;
      assert(tps->size() == 1);

      // Build subst mapping the array's TypeParam T to the element type.
      // Include outer_subst so any TypeParam refs in elem_type are resolved.
      NodeMap<Node> subst = outer_subst;
      subst[tps->at(0)] = clone(elem_type);

      find_or_push(array_def, std::move(subst));
    }

    void reify_call(Node& call, const NodeMap<Node>& subst)
    {
      auto hand = (call / Lhs)->type();
      auto arity = (call / Args)->size();

      auto funcid = get_reification(call / FuncName, subst, [&](auto& def) {
        return (def == Function) && ((def / Params)->size() == arity) &&
          (((def / Lhs) == hand) || ((def / Lhs) == Once && hand == Rhs));
      });

      if (!funcid || (funcid == Dyn))
      {
        // Can't resolve function — type arguments may need to be explicit.
        auto funcname = call / FuncName;
        call->parent()->replace(
          call,
          err(
            funcname,
            "Cannot resolve function — type arguments may need to be "
            "specified explicitly"));
        return;
      }

      auto dst = call / LocalId;
      auto args = call / Args;
      call->erase(call->begin(), call->end());
      call << dst << funcid << args;
    }

    void reify_new(Node& n, const NodeMap<Node>& subst)
    {
      auto dst = n / LocalId;
      auto type_node = n / Type;
      auto newargs = n / NewArgs;

      // Navigate the TypeName to find the ClassDef for field ordering.
      Node def = top;
      auto tn = (type_node == Type) ? type_node->front() : type_node;

      if (tn == TypeName)
      {
        for (auto& elem : *tn)
        {
          auto defs = def->look((elem / Ident)->location());

          if (!defs.empty())
            def = defs.front();
        }
      }

      // Reify the type to get a ClassId.
      auto classid =
        reify_emitted_type(type_node, subst, type_node, "constructed type");

      // Convert NewArgs to Args, ordered by field position in the class.
      Node args = Args;

      if (def == ClassDef)
      {
        for (auto& f : *(def / ClassBody))
        {
          if (f != FieldDef)
            continue;

          auto field_name = (f / Ident)->location().view();

          for (auto& na : *newargs)
          {
            if ((na / Ident)->location().view() == field_name)
            {
              args << (Arg << ArgCopy << clone(na->at(1)));
              break;
            }
          }
        }
      }

      // Fallback: if we couldn't match fields, just use NewArgs order.
      if (args->empty())
      {
        for (auto& na : *newargs)
          args << (Arg << ArgCopy << clone(na->at(1)));
      }

      n->erase(n->begin(), n->end());
      n << dst << classid << args;
    }

    void reify_lookup(
      Node& n, const NodeMap<Node>& call_subst, const Node& caller_id)
    {
      auto dst = n / LocalId;
      auto src = n / Rhs;
      auto hand = n / Lhs;
      auto ident = n / Ident;
      auto typeargs = n / TypeArgs;
      auto arity_node = n / Int;

      // Build method ID: "name::arity[::ref]::index"
      auto name = std::string(ident->location().view());
      auto arity = from_chars_sep_v<size_t>(arity_node);
      auto base_id =
        std::format("{}::{}{}", name, arity, hand == Lhs ? "::ref" : "");

      // Find or create a reification index for these resolved type arguments.
      auto index = find_method_index(base_id, typeargs, call_subst);
      auto method_id_str = std::format("{}::{}", base_id, index);

      // Determine receiver types from the source local's tracked type.
      ReceiverSet receivers{true, {}};
      auto src_it = local_types.find(src->location());
      Node receiver_type = Dyn;

      if (src_it != local_types.end())
      {
        receiver_type = clone(src_it->second);
        receivers = extract_receivers(src_it->second);
      }

      // Record this method invocation for method registration.
      method_invocations.push_back(
        {caller_id->location(),
         n,
         std::move(receiver_type),
         method_id_str,
         name,
         arity,
         hand->type(),
         clone(typeargs),
         call_subst,
         std::move(receivers)});

      // Register this new MI on existing class reifications that match.
      // Iterate via map_order (insertion order) rather than map (pointer order)
      // to ensure deterministic function reification ordering across runs.
      // Use index-based loop because register_method -> find_or_push can
      // push_back to map_order, invalidating range-for iterators.
      auto& mi = method_invocations.back();
      auto map_order_size = map_order.size();

      for (size_t i = 0; i < map_order_size; i++)
      {
        if (map_order[i] != ClassDef)
          continue;

        for (auto& r : map[map_order[i]])
        {
          if (r.reification && mi_targets(mi, r.id))
            register_method(mi, r);
        }
      }

      auto mid = MethodId ^ method_id_str;

      n->erase(n->begin(), n->end());
      n << dst << src << mid;
    }

    // Register a single MethodInvocation on a single class Reification.
    // If the class has a matching function, reify it and add a Method entry.
    void register_method(const MethodInvocation& mi, Reification& r)
    {
      assert(r.def == ClassDef);

      if ((r.def / Shape) == Shape)
        return;

      // Skip method registration if the class has TypeParams but the
      // subst doesn't include them (e.g., wrapper classes created by
      // ensure_ref_reified with empty subst). Methods will be registered
      // when the subst is updated via find_or_push.
      auto class_tps = r.def / TypeParams;

      if (!class_tps->empty() && r.subst.empty())
        return;

      auto mid_node = MethodId ^ mi.method_id;

      for (auto& f : *(r.def / ClassBody))
      {
        if (f != Function)
          continue;

        if ((f / Ident)->location().view() != mi.name)
          continue;

        if ((f / Lhs)->type() != mi.hand)
          continue;

        if ((f / Params)->size() != mi.arity)
          continue;

        auto func_tps = f / TypeParams;

        if (func_tps->size() != mi.typeargs->size())
          continue;

        // Build func_subst: class subst + method TypeParams -> resolved
        // TypeArgs.  Resolve TypeArgs through both call-site and class
        // substitution contexts (class subst takes priority for class
        // TypeParams).
        NodeMap<Node> combined = mi.call_subst;

        for (auto& [k, v] : r.subst)
          combined.insert_or_assign(k, v);

        NodeMap<Node> func_subst = r.subst;

        for (size_t i = 0; i < func_tps->size(); i++)
        {
          auto ta = mi.typeargs->at(i);
          Node resolved = (ta == Type) ? reify_type(ta, combined) : Dyn;
          func_subst[func_tps->at(i)] = resolved;
        }

        auto funcid = find_or_push(f, func_subst);

        // Check if this Method entry already exists.
        auto methods = r.reification / Methods;
        bool already =
          std::any_of(methods->begin(), methods->end(), [&](auto& existing) {
            return ((existing / MethodId)->location().view() == mi.method_id) &&
              ((existing / FunctionId)->location().view() ==
               funcid->location().view());
          });

        if (!already)
        {
          dynamic_call_targets[mi.method_id].push_back(
            {clone(r.id), clone(funcid)});
          methods << (Method << clone(mid_node) << funcid);
        }
      }
    }

    Nodes resolve_callback_targets(Node type)
    {
      if (!type)
        return {};

      if (type == Type)
        type = type->front();

      if (type == ClassId)
        return {clone(type)};

      if (type == Union)
      {
        Nodes targets;

        for (auto& child : *type)
        {
          auto resolved = resolve_callback_targets(child);
          targets.insert(targets.end(), resolved.begin(), resolved.end());
        }

        return targets;
      }

      if (type != TypeId)
        return {};

      auto type_id_loc = type->location().view();

      for (auto& key : map_order)
      {
        for (auto& r : map[key])
        {
          if (!r.id || (r.id->location().view() != type_id_loc))
            continue;

          if (!r.reification && (r.def == TypeAlias))
            reify_typealias(r);

          if (r.reification && (r.reification == Type))
            return resolve_callback_targets(r.reification->back());

          return {};
        }
      }

      return {};
    }

    // Register @callback on a class and return its reified apply function.
    // If match_count_out and has_generic_out are provided, they report
    // details about the apply method search.
    Node ensure_callback_method(
      const Node& class_id,
      size_t* match_count_out = nullptr,
      bool* has_generic_out = nullptr)
    {
      // Find the Reification for the lambda's class.
      auto class_id_loc = class_id->location().view();
      Reification* target_r = nullptr;

      for (auto& key : map_order)
      {
        if (key != ClassDef)
          continue;

        for (auto& r : map[key])
        {
          if (r.id && (r.id->location().view() == class_id_loc))
          {
            target_r = &r;
            break;
          }
        }

        if (target_r)
          break;
      }

      if (!target_r)
        return {};

      // Ensure the class has been reified (it may have just been
      // added to the worklist by find_or_push and not yet processed).
      if (!target_r->reification)
        reify_class(*target_r);

      // Scan the ClassDef for a unique non-generic `apply` method.
      Node found_func;
      size_t match_count = 0;
      bool has_generic = false;

      for (auto& f : *(target_r->def / ClassBody))
      {
        if (f != Function)
          continue;

        if ((f / Ident)->location().view() != "apply")
          continue;

        if ((f / Lhs)->type() != Rhs)
          continue;

        if (!((f / TypeParams)->empty()))
        {
          has_generic = true;
          continue;
        }

        found_func = f;
        match_count++;
      }

      if (match_count_out)
        *match_count_out = match_count;
      if (has_generic_out)
        *has_generic_out = has_generic;

      if (match_count != 1)
        return {};

      // Reify the apply function with the class's substitution context.
      auto funcid = find_or_push(found_func, target_r->subst);

      // Register the @callback Method on the class.
      auto methods = target_r->reification / Methods;
      auto mid_node = MethodId ^ "@callback";
      bool already = false;

      for (auto& existing : *methods)
      {
        if (
          ((existing / MethodId)->location().view() == "@callback") &&
          ((existing / FunctionId)->location().view() ==
           funcid->location().view()))
        {
          already = true;
          break;
        }
      }

      if (!already)
        methods << (Method << clone(mid_node) << funcid);

      return funcid;
    }

    void emit_make_callback_error(
      Node& n, size_t match_count, bool has_generic) const
    {
      if (match_count == 0)
      {
        auto msg = has_generic ?
          "make_callback requires a non-generic 'apply' method" :
          "make_callback requires a type with an 'apply' method";
        n->parent()->replace(n, err(n, msg));
        return;
      }

      if (match_count > 1)
      {
        n->parent()->replace(
          n, err(n, "make_callback requires exactly one 'apply' overload"));
      }
    }

    bool register_callback_type(
      const Node& type,
      Node site = {},
      bool required = false,
      Node registrant = {},
      Node registration_site = {})
    {
      auto targets = resolve_callback_targets(type);

      if (targets.empty())
      {
        auto inner = type;

        if (inner && (inner == Type))
          inner = inner->front();

        if (inner && (inner == TypeId))
        {
          pending_callbacks.push_back(
            {site,
             clone(type),
             required,
             registrant ? clone(registrant) : Node{},
             registration_site ? clone(registration_site) : Node{}});
          return true;
        }

        if (required && site)
        {
          site->parent()->replace(
            site, err(site, "make_callback: cannot determine lambda type"));
        }

        return false;
      }

      size_t match_count = 0;
      bool has_generic = false;

      for (auto& class_id : targets)
      {
        auto funcid =
          ensure_callback_method(class_id, &match_count, &has_generic);

        if (!funcid)
        {
          if (required && site)
            emit_make_callback_error(site, match_count, has_generic);

          return false;
        }

        if (registrant)
          record_call_edge(registrant, funcid, registration_site);
      }

      return true;
    }

    void process_pending_callbacks(bool final_pass)
    {
      std::vector<PendingCallback> remaining;

      for (auto& pending : pending_callbacks)
      {
        auto targets = resolve_callback_targets(pending.type);

        if (targets.empty())
        {
          if (final_pass && pending.required && pending.site)
          {
            pending.site->parent()->replace(
              pending.site,
              err(pending.site, "make_callback: cannot determine lambda type"));
          }
          else
          {
            remaining.push_back(
              {pending.site,
               clone(pending.type),
               pending.required,
               pending.registrant ? clone(pending.registrant) : Node{},
               pending.registration_site ? clone(pending.registration_site) :
                                           Node{}});
          }

          continue;
        }

        size_t match_count = 0;
        bool has_generic = false;
        bool ok = true;

        for (auto& class_id : targets)
        {
          auto funcid =
            ensure_callback_method(class_id, &match_count, &has_generic);

          if (!funcid)
          {
            ok = false;
            break;
          }

          if (pending.registrant)
            record_call_edge(
              pending.registrant, funcid, pending.registration_site);
        }

        if (ok)
          continue;

        if (final_pass && pending.required && pending.site)
          emit_make_callback_error(pending.site, match_count, has_generic);
      }

      pending_callbacks = std::move(remaining);
    }

    void reify_make_callback(Node& n, const Node& type)
    {
      register_callback_type(type, n, true);
    }

    Node find_ffi_initializer(const Node& cls, const Node& lib)
    {
      auto defs = cls->lookdown(ffi_init_id(lib));

      if (defs.empty())
        return {};

      assert(defs.size() == 1);
      assert(defs.front() == Function);
      return defs.front();
    }

    void reify_ffi(Node& n, Reification& r)
    {
      auto sym_id = n / SymbolId;
      auto sym_name = sym_id->location();

      // Walk up from the function definition to find the Lib that defines
      // this symbol.
      auto def = r.def;
      auto parent = def->parent(ClassDef);

      while (parent)
      {
        for (auto& child : *(parent / ClassBody))
        {
          if (child != Lib)
            continue;

          for (auto& sym : *(child / Symbols))
          {
            if (sym != Symbol)
              continue;

            if ((sym / SymbolId)->location() == sym_name)
            {
              auto initializer = find_ffi_initializer(parent, child);

              if (initializer)
              {
                auto initializer_id = find_or_push(initializer, r.subst);
                record_call_edge(r.id, initializer_id, n);
              }

              // Found the matching symbol in this Lib.
              // Get or create the reified Lib.
              auto lib_loc = (child / String)->location();
              auto find = libs.find(lib_loc);
              Node reified_lib;

              if (find == libs.end())
              {
                reified_lib = Lib << clone(child / String) << Symbols;
                libs[lib_loc] = reified_lib;
              }
              else
              {
                reified_lib = find->second;
              }

              // Reify the types in the symbol.
              Node ffi_params = FFIParams;

              for (auto& p : *(sym / FFIParams))
                ffi_params << reify_emitted_type(
                  p, r.subst, sym / SymbolId, "FFI parameter type");

              auto ret_type = reify_emitted_type(
                sym / Type, r.subst, sym / SymbolId, "FFI return type");

              // Add the reified symbol. Duplicate detection and type
              // compatibility checking is done in the VIRC assign_ids pass.
              auto reified_symbols = reified_lib / Symbols;
              reified_symbols
                << (Symbol << clone(sym / SymbolId) << clone(sym / Lhs)
                           << clone(sym / Rhs) << clone(sym / Vararg)
                           << ffi_params << ret_type);

              return;
            }
          }
        }

        parent = parent->parent(ClassDef);
      }
    }

    Node reify_when(Node& n, Reification& r)
    {
      auto when_type = n / Type;
      Node inner_type;

      if (when_type->front() != TypeVar)
      {
        inner_type = reify_type(when_type, r.subst);
      }
      else
      {
        // TypeVar return: try to resolve from the lambda apply's
        // registered method reification. The When's Rhs is the Lookup
        // result for 'apply' on the lambda.
        auto src_loc = (n / Rhs)->location();
        auto li = lookup_info.find(src_loc);

        if (li != lookup_info.end())
        {
          auto recv_it = local_types.find(li->second.recv_loc);

          if (recv_it != local_types.end())
          {
            auto ret =
              find_method_return_type(recv_it->second, li->second.method_id);

            if (ret)
              inner_type = ret;
          }
        }

        // Fallback: emit Dyn if we couldn't resolve.
        if (!inner_type)
          inner_type = Dyn;
      }

      auto dst_loc = (n / LocalId)->location();
      auto result = WhenDyn << (n / LocalId) << (n / Rhs) << (n / Args)
                            << (Cown << inner_type);

      // Track the When result as Cown << inner_type.
      local_types[dst_loc] = Cown << clone(inner_type);
      return result;
    }

    // Given a TypeName or FuncName and a substitution map, find or create a
    // reification and return the ClassId, TypeId, or FunctionId. The accept
    // function is used to filter the final definition, such as looking for a
    // function with a specific arity and handedness.
    template<typename F>
    Node get_reification(const Node& name, const NodeMap<Node>& subst, F accept)
    {
      assert(name->in({TypeName, FuncName}));
      Node def = top;

      // Navigate the fully qualified name from Top, collecting TypeParam
      // substitutions from TypeArgs along the way.
      // r.subst only contains entries for TypeParams encountered during
      // navigation (the def's own params), not the caller's context.
      // resolve_subst combines both for resolving TypeArg references.
      Reification r{top, {}, {}, {}, {}};
      NodeMap<Node> resolve_subst = subst;

      for (auto it = name->begin(); it != name->end(); ++it)
      {
        auto& elem = *it;
        assert(elem == NameElement);
        auto ident = elem / Ident;
        auto ta = elem / TypeArgs;
        bool is_last = (it + 1 == name->end());

        auto defs = def->look(ident->location());

        if (defs.empty())
        {
          if (def == Top)
            return err(elem, "No top-level definition found");

          return err(
                   elem,
                   "Identifier not found: " +
                     std::string(ident->location().view()))
            << errmsg("Resolving here:") << errloc(def / Ident);
        }

        if (is_last)
        {
          // If the definition is a TypeParam, look it up in the substitution
          // map and reify the substituted type directly.
          for (auto& d : defs)
          {
            if (d == TypeParam)
            {
              auto find = resolve_subst.find(d);

              if (find != resolve_subst.end())
                return reify_type(find->second, resolve_subst);

              return Dyn;
            }
          }

          // Use the accept filter to find the right def.
          bool found = false;

          for (auto& d : defs)
          {
            if (accept(d))
            {
              def = d;
              found = true;
              break;
            }
          }

          if (!found)
          {
            return err(elem, "No matching definition found")
              << errmsg("Resolving here:") << errloc(defs.front() / Ident);
          }
        }
        else
        {
          // Intermediate elements must resolve to a scope (ClassDef).
          // When multiple defs exist (e.g., function overloads), pick
          // the one that contains the next element in the path.
          if (defs.size() > 1)
          {
            auto next_ident = (*(it + 1)) / Ident;
            def = {};

            for (auto& d : defs)
            {
              auto next_defs = d->look(next_ident->location());

              if (!next_defs.empty())
              {
                def = d;
                break;
              }
            }

            if (!def)
              def = defs.front();
          }
          else
          {
            def = defs.front();
          }

          if (def == TypeParam)
          {
            // Look up the TypeParam in the substitution map and resolve
            // through the substituted type to find the ClassDef.
            auto find = resolve_subst.find(def);

            if (find == resolve_subst.end())
              return err(elem, "TypeParam has no substitution");

            auto sub = find->second;

            // Unwrap Type node.
            if (sub == Type)
              sub = sub->front();

            if (sub == Dyn)
              return err(
                elem,
                "Cannot resolve type — type arguments may need to be "
                "specified explicitly");

            if (sub != TypeName)
            {
              return err(
                elem, "TypeParam substitution must be a type name here");
            }

            // Navigate from the substituted TypeName to find the ClassDef.
            def = top;

            for (auto& se : *sub)
            {
              auto si = se / Ident;
              auto sta = se / TypeArgs;
              auto sdefs = def->look(si->location());

              if (sdefs.empty())
                return err(se, "Definition not found in TypeParam resolution");

              def = sdefs.front();

              if (!sta->empty())
              {
                auto stps = def / TypeParams;

                for (size_t i = 0; i < stps->size(); i++)
                {
                  r.subst[stps->at(i)] = sta->at(i);
                  resolve_subst[stps->at(i)] = sta->at(i);
                }
              }
            }

            if (def != ClassDef)
            {
              return err(
                elem,
                "TypeParam substitution must resolve to a class for "
                "intermediate navigation");
            }
          }
          else if (!def->in({ClassDef, Function}))
          {
            return err(elem, "Intermediate name must be a class or function")
              << errmsg("Resolving here:") << errloc(def / Ident);
          }
        }

        // Build substitution from TypeArgs when provided.
        auto tps = def / TypeParams;

        if (!ta->empty())
        {
          if (ta->size() != tps->size())
          {
            return err(
                     elem,
                     std::format(
                       "Expected {} type arguments, got {}",
                       tps->size(),
                       ta->size()))
              << errmsg("Resolving here:") << errloc(def / Ident);
          }

          for (size_t i = 0; i < tps->size(); i++)
          {
            // Substitute any TypeParam references in the TypeArg using the
            // full resolution context, to avoid self-referential cycles.
            auto arg = ta->at(i);
            auto resolved = resolve_typearg(arg, resolve_subst);
            r.subst[tps->at(i)] = resolved;
            resolve_subst[tps->at(i)] = resolved;
          }
        }
        else if (!tps->empty())
        {
          // No TypeArgs but the def has TypeParams (e.g., bare class name
          // as return type from within a generic class). Inherit any
          // existing substitutions from the resolution context.
          // If not found in resolve_subst, check existing reifications of
          // this def for a TypeParam binding (handles nested classes of
          // generic outer classes, where the outer subst isn't in the
          // immediate resolution context).
          for (auto& tp : *tps)
          {
            auto find = resolve_subst.find(tp);

            if (find != resolve_subst.end())
            {
              r.subst[tp] = find->second;
              resolve_subst[tp] = find->second;
            }
            else
            {
              auto map_it = map.find(def);
              if (map_it != map.end() && !map_it->second.empty())
              {
                auto& existing = map_it->second.front();
                auto existing_find = existing.subst.find(tp);
                if (existing_find != existing.subst.end())
                {
                  r.subst[tp] = existing_find->second;
                  resolve_subst[tp] = existing_find->second;
                }
              }
            }
          }
        }
      }

      // Build a resolved TypeName with all TypeParam refs substituted.
      // This is stored on the Reification for use in shape checking.
      Node resolved_name;
      resolved_name = name->type();

      for (auto& elem : *name)
      {
        Node new_ta = TypeArgs;

        for (auto& a : *(elem / TypeArgs))
          new_ta << clone(resolve_typearg(a, resolve_subst));

        resolved_name << (NameElement << clone(elem / Ident) << new_ta);
      }

      // Shapes produce Dyn in function bodies (preserving method dispatch
      // behavior), but we record a map entry so the post-worklist phase
      // can build a Type << TypeId << Union of matching concrete classes.
      if ((def == ClassDef) && ((def / Shape) == Shape))
      {
        // _builtin::any is the universal shape — remains pure Dyn.
        if (
          (def->parent(ClassDef) == builtin) &&
          ((def / Ident)->location().view() == "any"))
          return Dyn;

        return find_or_push(def, std::move(r.subst), resolved_name);
      }

      return find_or_push(def, std::move(r.subst), resolved_name);
    }

    Node make_id(const Node& def, size_t index, const NodeMap<Node>& subst)
    {
      if (is_under_builtin(def) && (def == ClassDef))
      {
        // Check for a bare primitive type.
        auto find = primitive_types.find((def / Ident)->location().view());

        if (find != primitive_types.end())
          return find->second;

        // Check for an ffi primitive type.
        auto ffi_find =
          ffi_primitive_types.find((def / Ident)->location().view());

        if (ffi_find != ffi_primitive_types.end())
          return ffi_find->second;

        // Check for a wrapper type (array[T], cown[T], ref[T]).
        auto wrap_find = wrapper_types.find((def / Ident)->location().view());

        if (wrap_find != wrapper_types.end())
        {
          auto tps = def / TypeParams;
          assert(tps->size() == 1);
          auto tp_find = subst.find(tps->at(0));
          Node elem_type =
            (tp_find != subst.end()) ? reify_type(tp_find->second, subst) : Dyn;

          if (
            (tp_find == subst.end()) ||
            has_unresolved_type(tp_find->second, subst))
            emit_unresolved_type_error(def / Ident, "wrapper element type");

          return wrap_find->second << elem_type;
        }
      }

      // Identifiers take the form `a::b::c::3`.
      assert(def->in({ClassDef, TypeAlias, Function}));
      auto id = std::string((def / Ident)->location().view());
      auto parent = def->parent({Top, ClassDef, TypeAlias, Function});

      while (parent && parent != Top)
      {
        id = std::format("{}::{}", (parent / Ident)->location().view(), id);
        parent = parent->parent({Top, ClassDef, TypeAlias, Function});
      }

      if (def == Function)
      {
        // A function adds arity and handedness.
        id = std::format(
          "{}::{}{}",
          id,
          (def / Params)->size(),
          (def / Lhs) == Lhs ? "::ref" : "");
      }

      id = std::format("{}::{}", id, index);

      if (def == ClassDef)
      {
        if ((def / Shape) == Shape)
          return TypeId ^ id;
        return ClassId ^ id;
      }
      else if (def == TypeAlias)
        return TypeId ^ id;
      else if (def == Function)
        return FunctionId ^ id;

      assert(false);
      return {};
    }
  };

  PassDef reify()
  {
    PassDef p{"reify", wfIR, dir::bottomup, {}};

    p.pre([=](auto top) {
      Reifier reifier;
      reifier.run(top);
      return lower_once(top, reifier.once_functions(), reifier.calls());
    });

    return p;
  }
}
