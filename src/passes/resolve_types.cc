#include "../infix.h"

#include <deque>
#include <iostream>
#include <map>
#include <unordered_set>
#include <vector>

#include <trieste/nodeworker.h>

/**
 * To handle generics correctly, we need to handle resolutions to be arbitrary
 * types, we need to perform a rebase operation, that takes two types: 1) The
 * first has to be a type lookup 2) Can be an arbitrary type.
 *
 * We then need to effectively substitute inside the second type all type
 * lookups to initially have the path of the first type lookup.
 *
 * For example:
 *
 *    ..::..::A[..::B]   +++   ..::C::D[..::E]  ==>
 * ..::..::A[..::B]::..::C::D[..::..::A[..::B]::E]
 *    ==>  ..::..::C::D[..::B]
 *
 * This is pervasive through the whole reolution.
 *
 * This should be implemented as a function on a clone of a Node, that returns a
 * new Node.
 */

namespace infix {



// A view into a TypeLookup's children representing a resolved path.
// The range includes any leading Parent tokens which represent scope levels.
// This avoids copying path segments during resolution.
struct RelativePath {
  // The TypeLookup node this path references (may be null for empty paths).
  Node type_lookup{nullptr};
  // Range [0, resolved_end) within type_lookup's children is the resolved portion.
  // This includes any leading Parent tokens.
  size_t resolved_end{0};
  // The final resolved scope node for symbol lookup.
  Node node{nullptr};

  // Check if this path is uninitialized (sentinel for "search all scopes").
  bool is_uninitialized() const { return node == nullptr && resolved_end == 0; }

  // Count leading Parent tokens in the resolved portion.
  size_t parent_count() const {
    if (!type_lookup)
      return 0;
    size_t count = 0;
    for (size_t i = 0; i < resolved_end && i < type_lookup->size(); ++i) {
      if (type_lookup->at(i) == Parent)
        count++;
      else
        break;
    }
    return count;
  }

  // Get the first non-Parent index in the resolved portion.
  size_t first_segment_index() const { return parent_count(); }

  // Get a segment at a given index within the resolved portion (after Parents).
  Node segment_at(size_t idx) const {
    size_t actual_idx = first_segment_index() + idx;
    if (actual_idx < resolved_end && type_lookup)
      return type_lookup->at(actual_idx);
    return nullptr;
  }

  // Number of non-Parent segments in the resolved portion.
  size_t segment_count() const {
    size_t parents = parent_count();
    return (resolved_end > parents) ? (resolved_end - parents) : 0;
  }

  // Get the last segment (for accessing TypeArgs during generic substitution).
  Node back_segment() const {
    if (segment_count() == 0)
      return nullptr;
    return type_lookup->at(resolved_end - 1);
  }

  // Check if the entire path has been resolved (resolved_end covers all elements).
  bool is_fully_resolved() const {
    return type_lookup && resolved_end == type_lookup->size();
  }
};

std::ostream &operator<<(std::ostream &os, RelativePath const &rp) {
  os << "RelativePath(parents=" << rp.parent_count()
     << ", segments=" << rp.segment_count() << ", resolved_end=" << rp.resolved_end
     << ", node=" << (rp.node ? rp.node->str() : "<null>") << ")";
  return os;
}

// Prepend Parents and optionally segments from a path to a TypeLookup.
// Inserts `extra_parents` Parent tokens, plus the contents of `path` if provided.
// Returns the total number of elements inserted.
size_t prepend_to_type_lookup(Node entry, size_t extra_parents,
                              const RelativePath *path = nullptr) {
  size_t inserted = 0;
  assert(path == nullptr || path->is_fully_resolved());

  // Insert segments from path (in reverse to maintain order after insertions at begin)
  if (path) {
    for (size_t i = path->segment_count(); i > 0; --i) {
      // We need to clone the segment and apply extra_parents to and TypeLookup inside it, if present.
      Node t = bottom_up_map(path->segment_at(i - 1), [&](Node n, const Node &) {
        if (n == TypeLookup) {
          // Insert extra Parents into the segment's TypeLookup
          for (size_t j = 0; j < extra_parents; ++j) {
            n->insert(n->begin(), Parent);
          }
        }
        return n;
      });
      entry->insert(entry->begin(), t);
      inserted++;
    }
    // Insert Parents from path
    for (size_t i = 0; i < path->parent_count(); ++i) {
      entry->insert(entry->begin(), Parent);
      inserted++;
    }
  }

  // Insert extra Parents
  for (size_t i = 0; i < extra_parents; ++i) {
    entry->insert(entry->begin(), Parent);
    inserted++;
  }

  return inserted;
}

// Initialize path by counting leading Parents and walking up the scope chain.
void init_path(Node entry, RelativePath &path) {
  path.resolved_end = 0;
  Node scope = entry->scope();
  
  for (auto &child : *entry) {
    if (child != Parent)
      break;
    path.resolved_end++;
    if (scope) scope = scope->scope();
  }
  
  path.node = scope;
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
  // The resolution of the name. path.resolved_end tracks how much of the
  // TypeLookup has been resolved; the rest is the pending suffix.
  RelativePath path;
  // Expand aliases - true for `use` statement bodies
  bool expand_aliases{false};
  // Used to detect if we have already waited for all subterms.
  bool blocked_on_subterms{false};
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
// When a TypeParam is found, it's substituted with the corresponding type arg.
//
// Invariant: segment_count tracks the number of TypeReference segments in
// entry[0..i). When segment_count > 0, the last segment is at entry[i-1].
// This works because a Parent always immediately follows a segment (after
// normalization), so when we remove a segment+Parent pair, the new last
// segment is still at i-1 (after adjusting i).
static void normalize_path(Node entry, const Node &base) {
  Node scope = base->scope();
  size_t segment_count = 0;
  size_t i = 0;
  
  while (i < entry->size()) {
    Node elem = entry->at(i);
    
    if (elem == Parent) {
      // Always move up scope chain
      if (scope) scope = scope->scope();
      
      if (segment_count > 0) {
        // Delete the previous segment (at i-1) and this Parent
        entry->erase(entry->begin() + i - 1, entry->begin() + i + 1);
        segment_count--;
        i--;
        continue;
      }
      i++;
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
}

  static Node rebase_path(const Node &base, const RelativePath &prefix_,
                          Node source) {
    // Only rebase type lookups.
    if (source != TypeLookup) {
      return source;
    }
    // Step 1: Prepend the prefix to source
    for (size_t i = prefix_.resolved_end; i > 0; --i) {
      source->insert(source->begin(), prefix_.type_lookup->at(i - 1)->clone());
    }
    
    // Step 2: Normalize (handles Parents and TypeParam substitution)
    normalize_path(source, base);

    return source;
  }

  // Find a name by searching up the scope chain.
  // Modifies the entry TypeLookup in-place, inserting Parent tokens and/or
  // segments from use statements. Updates state.path accordingly.
  // Returns true if found, false if blocked waiting on unresolved uses.
  bool lookup_levels_up(const Node &name, State &state,
                        NodeWorker<ResolveWork> &worker) const {
    Node entry = state.path.type_lookup;
    Node scope = name->scope();
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
        prepend_to_type_lookup(entry, levels, nullptr);
        state.path.resolved_end = levels; // Only the Parents are "resolved" so far
        state.path.node = scope;
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
        auto found = u_lookup_state.path.node->look(name->location());
        if (found.size() > 1) {
          ambiguous_lookup_error(u_lookup_state.path.node, name);
          continue;
        }

        if (found.size() == 1) {
          // Found via a use statement.
          // Copy the use's resolved path into entry, plus extra Parents.
          const RelativePath &use_path = u_lookup_state.path;
          assert(use_path.is_fully_resolved());
          state.path.resolved_end = prepend_to_type_lookup(entry, levels, &use_path);
          state.path.node = u_lookup_state.path.node;
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

  void seed(const Node &n, State &state) {
    if (n == Type)
      // Type are just containers for TypeLookup nodes.
      // They just form joins in the worklist.
      return;

    assert(n == TypeLookup);

    // Initialize the path to reference this TypeLookup
    state.path.type_lookup = n;
    state.path.resolved_end = 0; // Nothing resolved yet

    if (n->parent() && n->parent()->type() == Type) {
      if (n->parent()->parent() && n->parent()->parent()->type() == Use) {
        // This is a type lookup that is the body of a `use`. We want to
        // expand it completely to work out what names are in scope.
        state.expand_aliases = true;
      }
    }
  }

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
    auto remaining = [&]() {
      return entry->size() - state.path.resolved_end;
    };

    // Helper to get the next unresolved element
    auto current = [&]() -> Node {
      if (state.path.resolved_end < entry->size())
        return entry->at(state.path.resolved_end);
      return nullptr;
    };

    // Main resolution loop
    while (remaining() > 0) {
      Node reference = current();
      
      // Should be a TypeReference at this point
      if (reference != TypeReference) {
        return false;
      }

      Node head = reference / Name;

      // If path is uninitialized, look up the first name
      if (state.path.is_uninitialized()) {
        if (!lookup_levels_up(head, state, worker)) {
          // Not found in currently resolved scopes; lookup_levels_up will
          // have added unresolved `use` statements to wait on.
          return false;
        }
        // lookup_levels_up may have inserted Parents at the beginning,
        // so we need to skip past them
        while (current() == Parent) {
          state.path.resolved_end++;
        }
        // Also skip any segments that lookup_levels_up copied from a use
        // (resolved_end was already set by lookup_levels_up)
      }

      // Now look up the current segment
      auto found = state.path.node->look(head->location());
      // Should find either a module/struct, a type alias, or a type parameter.
      if (found.size() != 1) {
        ambiguous_lookup_error(state.path.node, head);
        return false;
      }

      if (found.front() == TypeAlias) {
        Node alias_type = found.front() / Type;
        Node name = found.front() / Name;

        // Can only expand if the alias body is a single TypeLookup
        bool can_expand = (alias_type->size() == 1) && 
                          (alias_type->at(0) == TypeLookup);
        
        if (!can_expand || ((remaining() == 1) && !state.expand_aliases)) {
          // Don't expand the alias if:
          // 1) The alias body is not a simple path (e.g., it's a union type), OR
          // 2) It's the last element of a TypeLookup and we're not in expand mode.
          state.path.resolved_end++;
          state.path.node = found.front();
          continue;
        }

        if (worker.block_on(entry, alias_type))
          return false;

        Node alias_body = alias_type->at(0);

        // The alias body must be resolved now, as we have waited for the
        // enclosing type.
        assert(worker.is_resolved(alias_body));

        // For rebasing, we need the path UP TO (but not including) the alias
        // reference. The current reference (Foo) will be replaced by the alias body.
        // Create a temporary path that includes the alias reference for rebase
        // context (needed for generic substitution).
        RelativePath rebase_prefix = state.path;
        rebase_prefix.resolved_end++; // Include the alias reference temporarily
        rebase_prefix.node = found.front();
        
        // Rebase the alias body into the current context
        Node rebased_alias =
            bottom_up_map(alias_body, [&](Node n, const Node &) {
              return rebase_path(entry, rebase_prefix, n);
            });
        
        // The rebased alias REPLACES the entire resolved prefix plus the alias reference.
        // The rebase_prefix included resolved_end+1, so we erase the entire path
        // up to and including the alias reference.
        entry->erase(entry->begin(), entry->begin() + rebase_prefix.resolved_end);
        // Insert the rebased alias children at the beginning
        // (no clone needed - rebased_alias is already a fresh copy from bottom_up_map)
        entry->insert(entry->begin(), rebased_alias->begin(), rebased_alias->end());
        
        // The rebased alias is already normalized (Parents at front).
        init_path(entry, state.path);

        continue;
      }

      if (found.front() == Module || found.front() == Struct) {
        state.path.resolved_end++;
        state.path.node = found.front();
        continue;
      }

      if (found.front() == TypeParam) {
        // TODO if we add bounds/assumptions on a TypeParam, then we may have to
        // change this.

        // We don't allow lookup on a type parameter.
        if (remaining() != 1) {
          head << (Error << (ErrorMsg ^
                             "Cannot resolve type lookup with additional "
                             "segments after type parameter")
                         << (ErrorMsg ^ head->location().str()));
          return false;
        }

        state.path.resolved_end++;
        continue;
      }

      // Unhandled candidate type will fail resolution at this level.
      return false;
    }

    assert(!worker.is_resolved(entry));

    // The entry has been modified in-place; no final substitution needed.
    assert(!ast_has_cycle(entry));
    assert(state.path.is_fully_resolved());

    return true;
  }
};

PassDef get_resolve_types_pass() {
  // Replace names used in types and uses with the fully relative qualified
  // names of their definitions. We only perform this for using definitions:
  //
  // module A[T]
  //   module B
  //     type Foo = T::F
  // module C[U]
  //   use A[U]
  //   type Da = B
  // module D[V]
  //   use C[V]::Da
  //   type F = Foo
  // module E
  //   type G = D[D[Int]]::F
  //
  // In this example we need to resolve generic parameters, but some are also
  // shared.  This allows us to normalise the paths correctly, and share names
  // where possible.  The following illustrates the various steps each name
  // should go through:
  //
  // module A[T]
  //   module B
  //     type Foo = T::F
  // module C[U]
  //   use ..::A[U]
  //   type Da = B
  // module D[V]
  //   use ..::C[V]::Da
  //   use ..::C[V]::B
  //   use ..::C[V]::..::A[..::C[V]::U]::B
  //   use ..::A[..::C[V]::U]::B
  //   use ..::A[V]::B
  //   type F = Foo
  // module E
  //   type G = D[D[Int]]::F
  //
  // Should we update the aliases we resolve on the way?
  // Implementation note:
  //  The pass requires a complex handling of cyclic dependences between name
  //  resolution of using statements and type aliases. Consider the following:
  //
  // module A
  //   use B::Foo
  //   type Bar = Foo
  //
  // module B
  //   use A::Bar
  //   type Foo = Bar
  //
  // In this case there is a cycle between the two modules A and B, and their
  // type aliases Foo and Bar. The resolution must ensure that both type aliases
  // are resolved to the same final type. To handle this we may need to perform
  // multiple passes until a fixed point is reached.
  //
  // module A
  //   use ..::B::Foo
  //   use ..::A::Bar
  //   use ..::B::Foo
  //   use ..::A::Bar
  //   ...
  //   type Bar = ..::B::Foo
  //
  // module B
  //   use ..::A::Bar
  //   ...
  //   type Foo = ..::A::Bar
  //
  // There are cases where this does not lead to a cycle, for instance.

  auto worker = std::make_shared<NodeWorker<ResolveWork>>(ResolveWork{});

  PassDef pass{"resolve_types",
               wf_function_parse,
               dir::bottomup | dir::once,
               {
                   // Capture the path body for every `use` statement.
                   T(TypeLookup)[TypeLookup] >> [worker](auto &_) -> Node {
                     worker->add(_(TypeLookup));
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