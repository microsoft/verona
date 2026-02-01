#include "../infix.h"

#include <deque>
#include <iostream>
#include <map>
#include <unordered_set>
#include <vector>

#include "node_worker.h"

/**
 * To handle generics correctly, we need to handle resolutions to be arbitrary types,
 * we need to perform a rebase operation, that takes two types:
 *  1) The first has to be a type lookup
 *  2) Can be an arbitrary type.
 * 
 * We then need to effectively substitute inside the second type all type lookups to
 * initially have the path of the first type lookup.
 * 
 * For example:
 * 
 *    ..::..::A[..::B]   +++   ..::C::D[..::E]  ==>   ..::..::A[..::B]::..::C::D[..::..::A[..::B]::E]
 *    ==>  ..::..::C::D[..::B]
 *
 * This is pervasive through the whole reolution.
 *
 * This should be implemented as a function on a clone of a Node, that returns a new Node.
 */

namespace infix {
struct RelativePath {
  // How many levels up to search; -1 means search all enclosing scopes.
  int lookup_levels{-1};
  // Resolved portion of the path so far (list of path segment nodes).
  Nodes prefix;
  // The final resolved node, if resolution is complete.  This is where the
  // symbol table for the resolved name can be found.  The prefix is required to
  // understand the meaning of generic parameters.  That is this is the module
  // or struct.
  Node node{nullptr};
};

std::ostream &operator<<(std::ostream &os, RelativePath const &rrn) {
  os << "RelativeResolvedName(lookup_levels=" << rrn.lookup_levels
     << ", prefix=[";
  for (auto &p : rrn.prefix) {
    os << p->str();
  }
  os << "])";
  return os;
}

auto ambiguous_lookup_error(Node symtab, Node node) {
  auto results = symtab->look(node->location());

  if (results.size() == 0)
    std::cerr << "Internal error: ambiguous_lookup_error called with no results " << std::endl
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

std::string type_lookup_to_str(Node type_lookup) {
  std::ostringstream oss;
  bool first = true;
  assert(type_lookup == TypeLookup);
  for (auto &child : *type_lookup) {
    if (!first) {
      oss << "::";
    } else {
      first = false;
    }
    oss << child->str();
  }
  return oss.str();
}

// Worklist of unresolved `use` and `type` bodies gathered during the pass.
std::deque<Node> lookup_worklist;

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

  PassDef pass{"resolve_types",
               wf_function_parse,
               dir::bottomup | dir::once,
               {
                   // Capture the path body for every `use` statement.
                   T(TypeLookup)[TypeLookup] >> [](auto &_) -> Node {
                     lookup_worklist.push_back(_(TypeLookup));
                     return NoChange;
                   },
               }};

  // Debug: dump the gathered bodies after the pass finishes.
  pass.post([](Node) {
    struct ResolutionState : NodeWorkerState {
      // Remaining portion that still needs resolution (list of path segment
      // nodes). This augments the base NodeWorkerState used by NodeWorker.
      std::deque<Node> pending_suffix;
      // The resolution of the name.
      RelativePath path;
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

      static void rebase_path(RelativePath &target,
                              const RelativePath &source) {
        // Rebase alias resolution into the current context, adjusting lookup
        // levels and prefix to account for the relative jump.
        if (target.prefix.size() > source.lookup_levels) {
          for (size_t i = 0; i < source.lookup_levels; i++) {
            target.prefix.pop_back();
          }
        } else {
          auto prefix_size = target.prefix.size();
          target.lookup_levels += source.lookup_levels - prefix_size;
          target.prefix.clear();
        }

        // if ((target.prefix.size() != 0) && (source.prefix.size() == 1)) {
        //   // Check if this is type parameter
        //   Node first = source.prefix.front();
        //   Nodes lookup = source.node->look(first->location());
        //   if (lookup.size() == 1 && lookup.front() == TypeParam) {
        //     Node type_params = source.node / TypeParams;
        //     for (size_t i = 0; i < type_params->size(); i++) {
        //       Node tp = type_params->at(i);
        //       if (tp == first) {
        //         // Find the i-th type parameter of this term, and replace the target with this term.
        //         auto t_args = target.prefix.back() / TypeArgs;
        //         t_args[i]

        //         return;
        //       }
        //     }
        //     // This is probably an error.

        //     return;
        //   }
        // }

        target.prefix.insert(target.prefix.end(), source.prefix.begin(),
                             source.prefix.end());
        target.node = source.node;
      }

      RelativePath lookup_levels_up(const Node &name, const Node &entry,
                                    NodeWorker<ResolveWork> &worker) const {
        Node scope = name->scope();
        Nodes unresolved_use_types;
        int levels = 0;
        while (scope) {
          auto results = scope->look(name->location());
          if (results.size() > 1) {
            ambiguous_lookup_error(scope, name);
            break;
          }
          if (results.size() == 1) {
            RelativePath result;
            result.lookup_levels = levels;
            result.prefix = {};
            result.node = scope;
            return result;
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
            // It should not be possible for the surrounding type lookup to be resolved,
            // without all the sub terms being resolved.
            assert(worker.is_resolved(u_lookup));

            auto &u_lookup_state = worker.state(u_lookup);
            auto found = u_lookup_state.path.node->look(name->location());
            if (found.size() > 1) {
              ambiguous_lookup_error(u_lookup_state.path.node, name);
              continue;
            }

            if (found.size() == 1) {
              // Return found result, and adjust levels according to how far up
              // the using statement was, then add the found name to the prefix.
              RelativePath result = u_lookup_state.path;
              result.lookup_levels += levels;
              return result;
            }
          }

          scope = scope->scope();
          levels++;
        }

        // We reach the top without finding it; wait on unresolved uses. If
        // none are pending, resolution will ultimately fail when processing
        // completes.
        worker.block_on_any(entry, unresolved_use_types);
        return {};
      }

      void seed(const Node &n, State &state) {
        if (n == Type)
          // Type are just containers for TypeLookup nodes.
          // They just form joins in the worklist.
          return;

        for (auto &child : *n) {
          state.pending_suffix.push_back(child);
        }
        // TODO sanity checking here for different terms, and handle generics.
      }

      bool process(const Node &entry, NodeWorker<ResolveWork> &worker) {
        auto &state = worker.state(entry);
        // If terms is a Type node, then find all type lookups inside and wait for them all to be resolved.
        if (entry == Type)
        {
          // use traverse to find all TypeLookup nodes inside entry
          std::vector<Node> type_lookups;
          entry->traverse([&](Node &current) {
            if (current == TypeLookup) {
              if (!worker.is_resolved(current)) {
                type_lookups.push_back(current);
              }
            }
            return true;
          });
          if (!type_lookups.empty()) {
            worker.block_on_all(entry, type_lookups);
            return false;
          }
          // All type lookups are resolved, we can consider this entry resolved.
          return true;
        }

        assert(entry == TypeLookup);
        while (!state.pending_suffix.empty()) {
          Node reference = state.pending_suffix.front();
          Node head = reference / Name;

          if (state.path.lookup_levels == -1) {
            state.path = lookup_levels_up(head, entry, worker);
            if (state.path.lookup_levels == -1) {
              // Not found in currently resolved scopes; lookup_level will have
              // added unresolved `use` statements to wait on.
              return false;
            }
          }

          auto found = state.path.node->look(head->location());
          // Should find either a module/struct, a type alias, or a type
          // parameter.
          if (found.size() != 1) {
            ambiguous_lookup_error(state.path.node, head);
            return false;
          }

          if (found.front() == TypeAlias) {
            // This effectively `rebase`s the alias body into the current
            // context. Resolve the type alias to continue lookups.
            Node alias_type = found.front() / Type;

            std::cout << "Resolving type alias during lookup: "
                      << type_lookup_to_str(entry) << " alias body: "
                      << alias_type->str() << std::endl;

            if ((alias_type->size() != 1)) {
              head << (Error << (ErrorMsg ^ "Invalid type alias body for lookup")
                             << (ErrorAst << head) << (ErrorAst << alias_type));
              return false;
            }

            if (worker.block_on(entry, alias_type))
              return false;

            Node alias_body = alias_type->at(0);
            // If alias not yet resolved, attempt to resolve it; otherwise
            // block until it completes.
            if (worker.block_on_any(entry, {alias_body})) {
              return false;
            }

            auto &resolved_alias = worker.state(alias_body);
            // Alias is resolved, add it to the resolved prefix and continue.
            // This involves substitution and adjustments for relative paths.
            // TODO, we need to apply rebase_path to all the type lookups inside the
            // resolved alias body.

            rebase_path(state.path, resolved_alias.path);
//            state.pending_suffix.pop_front();
            continue;
          }

          if (found.front() == Module || found.front() == Struct) {
            state.path.prefix.push_back(reference);
            state.pending_suffix.pop_front();
            state.path.node = found.front();
            continue;
          }

          if (found.front() == TypeParam) {
            // We don't allow lookup on a type parameter.
            if (state.pending_suffix.size() != 1) {
              head << (Error << (ErrorMsg ^ "Cannot resolve type lookup with additional segments after type parameter")
                             << (ErrorAst << head));
              return false;
            }
            // Type parameters aren't fields of the current scope.
            if (state.path.prefix.size() > 0) {
              head << (Error << (ErrorMsg ^ "Cannot resolve type lookup with prefix before type parameter")
                             << (ErrorAst << head));
              return false;
            }

            state.path.prefix.push_back(reference);
            state.pending_suffix.pop_front();
            continue;
          }

          // Unhandled candidate type will fail resolution at this level.
          // TODO: extend to functions/variables if resolution is unified.
          return false;
        }

        return true;
      }
    };

    NodeWorker<ResolveWork> worker(ResolveWork{});

    for (const auto &item : lookup_worklist) {
      worker.add(item);
    }

    worker.run();

    for (const auto &pair : worker.states()) {
      auto &name = pair.first;
      auto &status = pair.second;
      if (status.kind != WorkerStatus::Resolved) {
        name << (Error << (ErrorMsg ^ "Failed to resolve name:")
                       << (ErrorMsg ^ name->location().str()));

        continue;
      }

      if (name != TypeLookup)
        continue;
      name->erase(name->begin(), name->end());
      for (size_t i = 0; i < status.path.lookup_levels; i++) {
        name << (Parent);
      }
      for (const auto &seg : status.path.prefix) {
        name << seg->clone();
      }
    }

    return static_cast<size_t>(0);
  });

  return pass;
}

} // namespace infix