#include "../infix.h"

#include <deque>
#include <iostream>
#include <map>
#include <unordered_set>
#include <vector>

#include <trieste/nodeworker.h>

namespace infix {

Node clone_type_lookup_prefix(const Node &type_lookup, size_t count) {
  Node cloned = TypeLookup;
  for (size_t i = 0; i < count && i < type_lookup->size(); ++i) {
    cloned->push_back(type_lookup->at(i)->clone());
  }
  return cloned;
}

// Clone a node and add count Parent tokens at the beginning of any TypeLookup
// nodes. Uses bottom_up_map to perform the cloning.
Node prepend_parents(Node source, size_t count) {
  return bottom_up_map(source, [count](Node n, const Node &) {
    if (!(n == TypeLookup))
      return n;
    n->insert(n->begin(), count, Parent);
    return n;
  });
}

auto ambiguous_lookup_error(Node symtab, Node node) {
  auto results = symtab->look(node->location());

  if (results.size() == 0)
    std::cerr
        << "Internal error: ambiguous_lookup_error called with no results "
        << std::endl
        << node->location().str() << std::endl;

  Node error_node = Error << (ErrorMsg ^ "Ambiguous lookup:")
                          << (ErrorMsg ^ node->location().str())
                          << (ErrorMsg ^ " found ");
  bool first = true;
  for (auto &res : results) {
    error_node << (ErrorMsg ^ res->location().str());
    if (!first) {
      error_node << (ErrorMsg ^ "and");
    } else {
      first = false;
    }
  }

  return symtab << error_node;
}

// Return the index of `child` within its parent, if any.
std::optional<size_t> child_index_in_parent(const Node &child) {
  Node parent = child ? child->parent() : nullptr;
  if (!parent)
    return std::nullopt;

  size_t idx = 0;
  for (auto &c : *parent) {
    if (c == child)
      return idx;
    ++idx;
  }

  return std::nullopt;
}

struct ResolutionState : NodeWorkerState {
  // Range [0, resolved_end) within type_lookup's children is the resolved
  // portion. This includes any leading Parent tokens.
  size_t resolved_end{0};

  // The final resolved scope node for symbol lookup.
  Node node{nullptr};

  // Expand aliases - true for `use` statement bodies
  bool expand_aliases{false};

  // Used to detect if we have already waited for all subterms.
  bool blocked_on_subterms{false};

  // Check if this path is uninitialized (sentinel for "search all scopes").
  bool is_uninitialized() const { return node == nullptr && resolved_end == 0; }
};

// Core algorithm: maintain a worklist of type lookups to resolve and track
// dependents via NodeWorkerState. Blocking/unblocking is managed by
// NodeWorker.
struct ResolveWork {
  using State = ResolutionState;

  static Node use_to_type_lookup(const Node &use_node) {
    Node type_node = use_node / Type;
    assert(type_node->size() == 1);
    Node type_lookup = type_node->at(0);
    assert(type_lookup == TypeLookup);
    return type_lookup;
  }

  // Normalize a path by processing Parents and substituting TypeParams.
  // Takes a base node for scope lookups.
  // A Parent after a segment consumes that segment (A::B::.. -> A::).
  // A Parent at the front stays at the front and moves up the scope chain.
  // When a TypeParam is found, it's substituted with the corresponding type
  // arg.
  //
  // Invariant: segment_count tracks the number of TypeReference segments in
  // entry[0..i). When segment_count > 0, the last segment is at entry[i-1].
  // This works because a Parent always immediately follows a segment (after
  // normalization), so when we remove a segment+Parent pair, the new last
  // segment is still at i-1 (after adjusting i).
  static bool normalize_path(Node entry, const Node &base) {
    bool performed_work = false;
    Node scope = base->scope();
    assert(scope);
    size_t segment_count = 0;
    size_t i = 0;

    while (i < entry->size()) {
      Node elem = entry->at(i);

      if (elem == Parent) {
        // Always move up scope chain
        scope = scope->scope();
        assert(scope);
        i++;

        if (segment_count == 0)
          continue;

        performed_work = true;
        // Delete the previous segment (at i-2) and this Parent
        entry->erase(entry->begin() + i - 2, entry->begin() + i);
        segment_count--;
        i -= 2;
        continue;
      }

      assert(elem == TypeReference);

      assert(scope);

      // Check if it's a TypeParam
      Node name = elem / Name;
      auto lookups = scope->look(name->location());

      if (lookups.size() == 1 && lookups.front() == TypeParam) {
        auto index = child_index_in_parent(lookups.front());
        assert(index.has_value());

        // Get type args from previous segment (at i-1)
        Node type_args = nullptr;
        if (segment_count > 0 && entry->at(i - 1) == TypeReference) {
          type_args = entry->at(i - 1) / TypeArgs;
        }

        if (!type_args || type_args->size() <= index.value()) {
          std::cout << "[normalize_path] error: not enough generic arguments"
                    << std::endl;
          assert(false);
        }
        performed_work = true;

        Node arg = type_args->at(index.value());
        assert(arg == Type);
        Node lookup_arg = arg->at(0);
        assert(lookup_arg == TypeLookup);

        // Replace prefix (0 to i inclusive) with type arg content
        entry->erase(entry->begin(), entry->begin() + i + 1);
        entry->insert(entry->begin(), lookup_arg->begin(), lookup_arg->end());

        // Restart from the beginning
        scope = base->scope();
        segment_count = 0;
        i = 0;
        continue;
      }

      if (lookups.size() == 1) {
        scope = lookups.front();
      }

      segment_count++;
      i++;
    }

    return performed_work;
  }

  static Node rebase_path(const Node &base, size_t prefix_count, Node source) {
    // Only rebase type lookups.
    if (source != TypeLookup) {
      return source;
    }
    Node prefix = clone_type_lookup_prefix(base, prefix_count);
    source->insert(source->begin(), prefix->begin(), prefix->end());
    normalize_path(source, base);
    return source;
  }

  // Find a name by searching up the scope chain.
  // Modifies the entry TypeLookup in-place, inserting Parent tokens and/or
  // segments from use statements. The worker state for this node accordingly.
  // Returns true if found, false if blocked waiting on unresolved uses.
  bool lookup_levels_up(Node entry, NodeWorker<ResolveWork> &worker) const {

    Node scope = entry->scope();
    auto &state = worker.state(entry);
    assert(state.resolved_end == 0);
    Node name = entry->front() / Name;
    Nodes unresolved_use_types;
    size_t levels = 0;
    while (scope) {
      auto results = scope->look(name->location());
      if (results.size() > 1) {
        ambiguous_lookup_error(scope, name);
        break;
      }
      if (results.size() == 1) {
        // Found directly in scope chain.
        entry->insert(entry->begin(), levels, Parent);
        state.resolved_end = levels; // Only the Parents are "resolved" so far
        state.node = scope;
        return true;
      }
      // Not found, check the resolved `use` statements in this scope.
      auto current_using = scope->includes();
      for (const auto &u : current_using) {
        // Check if current use has been fully resolved
        // including all sub type lookups.
        Node u_type = u / Type;
        if (!worker.is_resolved(u_type)) {
          // Not resolved add to list to block on.
          unresolved_use_types.push_back(u_type);
          continue;
        }

        auto u_lookup = use_to_type_lookup(u);
        // It should not be possible for the surrounding type lookup to be
        // resolved, without all the sub terms being resolved.
        assert(worker.is_resolved(u_lookup));

        auto &u_lookup_state = worker.state(u_lookup);
        auto found = u_lookup_state.node->look(name->location());
        if (found.size() > 1) {
          ambiguous_lookup_error(u_lookup_state.node, name);
          continue;
        }

        if (found.size() == 1) {
          // Found via a use statement.
          // Copy the use's resolved path into entry, plus extra Parents.
          Node cloned_path = prepend_parents(u_lookup, levels);
          // Insert all children from cloned path
          entry->insert(entry->begin(), cloned_path->begin(),
                        cloned_path->end());
          state.resolved_end = cloned_path->size();
          state.node = u_lookup_state.node;
          return true;
        }
      }

      scope = scope->scope();
      levels++;
    }

    // We reach the top without finding it; wait on unresolved uses. If
    // none are pending, resolution will ultimately fail when processing
    // completes.
    worker.block_on_any(entry, unresolved_use_types);
    return false; // Not found yet
  }

  void seed(const Node &n, State &state) {}

  bool wait_on_subterms(const Node &n, NodeWorker<ResolveWork> &worker) {
    // Wait on all subterms to be resolved before processing this node. This
    // ensures that when we process a type lookup, all the type lookups it
    // contains are already resolved, which simplifies handling of nested type
    // lookups.
    std::vector<Node> subterms_to_wait_on;
    n->traverse([&](Node &current) {
      if (current == n)
        return true;
      if (current == Type || current == TypeLookup) {
        if (!worker.is_resolved(current)) {
          subterms_to_wait_on.push_back(current);
        }
        // Sub terms are responsible for their own subterms.
        return false;
      }
      return true;
    });
    if (!subterms_to_wait_on.empty()) {
      worker.block_on_all(n, subterms_to_wait_on);
      return true;
    }
    return false;
  }

  bool process(const Node &entry, NodeWorker<ResolveWork> &worker) {
    auto &state = worker.state(entry);

    // Check if we have already blocked on subterms, if we haven't
    // wait for all subterms to be resolved.
    if (!state.blocked_on_subterms) {
      state.blocked_on_subterms = true;
      if (wait_on_subterms(entry, worker)) {
        return false;
      }
    }

    if (entry == Type) {
      // All type lookups are resolved, we can consider this entry
      // resolved.
      return true;
    }

    assert(entry == TypeLookup);

    // Helper to count remaining unresolved elements
    auto remaining = [&]() { return entry->size() - state.resolved_end; };

    // Main resolution loop
    while (remaining() > 0) {
      Node current = entry->at(state.resolved_end);

      if (current == Parent) {
        if (state.node == nullptr)
          state.node = entry->scope();
        // Parents at the front just move up the scope chain, they don't
        // need to be resolved.
        state.resolved_end++;
        state.node = state.node->scope();
        continue;
      }

      // Should be a TypeReference at this point
      assert(current == TypeReference);

      // If path is uninitialized, look up the first name
      if (state.node == nullptr) {
        if (!lookup_levels_up(entry, worker)) {
          // Not found in currently resolved scopes; lookup_levels_up will
          // have added unresolved `use` statements to wait on.
          return false;
        }
        continue;
      }

      assert(state.node != nullptr);
      Node head = current / Name;
      // Now look up the current segment
      auto found = state.node->look(head->location());
      // Should find either a module/struct, a type alias, or a type
      // parameter.
      if (found.size() != 1) {
        ambiguous_lookup_error(state.node, head);
        return false;
      }

      if (found.front() == TypeAlias) {
        Node alias_type = found.front() / Type;
        Node name = found.front() / Name;

        // Can only expand if the alias body is a single TypeLookup
        bool can_expand =
            (alias_type->size() == 1) && (alias_type->at(0) == TypeLookup);

        if (!can_expand || ((remaining() == 1) && !state.expand_aliases)) {
          // Don't expand the alias if:
          // 1) The alias body is not a simple path (e.g., it's a union
          // type), OR 2) It's the last element of a TypeLookup and we're
          // not in expand mode.
          state.resolved_end++;
          state.node = found.front();
          continue;
        }

        if (worker.block_on(entry, alias_type))
          return false;

        Node alias_body = alias_type->at(0);

        // The alias body must be resolved now, as we have waited for the
        // enclosing type.
        assert(worker.is_resolved(alias_body));

        // For rebasing, we need the path UP TO (but not including) the
        // alias reference. The current reference (Foo) will be replaced by
        // the alias body. Create a temporary path that includes the alias
        // reference for rebase context (needed for generic substitution).
        state.resolved_end++; // Include the alias reference temporarily
        state.node = found.front();

        // Rebase the alias body into the current context
        Node rebased_alias =
            bottom_up_map(alias_body, [&](Node n, const Node &) {
              return rebase_path(entry, state.resolved_end, n);
            });

        // The rebased alias REPLACES the entire resolved prefix plus the
        // alias reference. The rebase_prefix included resolved_end+1, so we
        // erase the entire path up to and including the alias reference.
        entry->erase(entry->begin(), entry->begin() + state.resolved_end);
        // Insert the rebased alias children at the beginning
        // (no clone needed - rebased_alias is already a fresh copy from
        // bottom_up_map)
        entry->insert(entry->begin(), rebased_alias->begin(),
                      rebased_alias->end());

        // The rebased alias is already normalized (Parents at front).
        // After rebasing, we need to re-resolve from the start of the new
        // path
        state.resolved_end = 0;
        state.node = entry->scope();
        continue;
      }

      if (found.front() == Module || found.front() == Struct) {
        state.resolved_end++;
        state.node = found.front();
        continue;
      }

      if (found.front() == TypeParam) {
        // TODO if we add bounds/assumptions on a TypeParam, then we may
        // have to change this.

        // We don't allow lookup on a type parameter.
        if (remaining() != 1) {
          head << (Error << (ErrorMsg ^
                             "Cannot resolve type lookup with additional "
                             "segments after type parameter")
                         << (ErrorMsg ^ head->location().str()));
          return false;
        }

        if ((state.resolved_end == 0) ||
            (entry->at(state.resolved_end - 1) == Parent)) {
          state.resolved_end++;
          continue;
        }

        bool performed_work = normalize_path(entry, entry);
        if (performed_work) {
          // If we performed any normalization work, we need to restart
          // resolution from the beginning of the path, since the
          // normalization may have changed the path structure (e.g., by
          // consuming Parents or substituting type args).
          state.resolved_end = 0;
          state.node = entry->scope();
          continue;
        }
      }

      // Unhandled candidate type will fail resolution at this level.
      return false;
    }

    assert(!worker.is_resolved(entry));

    // The entry has been modified in-place; no final substitution needed.
    assert(!ast_has_cycle(entry));
    return true;
  }
};

PassDef get_resolve_types_pass() {
  auto worker = std::make_shared<NodeWorker<ResolveWork>>(ResolveWork{});

  PassDef pass{"resolve_types",
               wf_function_parse,
               dir::bottomup | dir::once,
               {
                   T(Use) << (T(Type) << (T(TypeLookup)[TypeLookup])) >>
                       [worker](auto &_) -> Node {
                     // For each use statement, we need to resolve the type
                     // lookup in its type. This will allow us to rebase the
                     // used module's contents into the current scope when
                     // processing the use.
                     Node type_lookup = _(TypeLookup);
                     worker->add(type_lookup);
                     // For use statements, we want to expand the aliases
                     // even at the head of the term.
                     worker->state(type_lookup).expand_aliases = true;
                     return NoChange;
                   },

                   T(TypeLookup)[TypeLookup] >> [worker](auto &_) -> Node {
                     // For each type lookup, we need to resolve it to the
                     // final scope node it refers to. This will allow us to
                     // rebase type aliases correctly when processing the
                     // type lookup.
                     Node type_lookup = _(TypeLookup);
                     worker->add(type_lookup);
                     return NoChange;
                   },
               }};

  // Debug: dump the gathered bodies after the pass finishes.
  pass.post([worker](Node) {
    worker->run();

    for (const auto &pair : worker->states()) {
      auto &name = pair.first;
      auto &status = pair.second;
      if (pair.first != TypeLookup) {
        continue;
      }

      if (status.kind != WorkerStatus::Resolved) {
        name << (Error << (ErrorMsg ^ "Failed to resolve name:")
                       << (ErrorMsg ^ name->location().str()));

        continue;
      }
    }

    return static_cast<size_t>(0);
  });

  return pass;
}

} // namespace infix