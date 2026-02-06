#include "../infix.h"

#include <unordered_set>
#include <vector>

namespace infix {

// Bottom-up map over the AST using trieste's non-recursive traverse.
// The mapper receives (1) a mutable clone of the current node's header with
// already-mapped children attached, and (2) the original node for context. It
// should return the replacement node (or nullptr to delete this sub-tree).
// Returns the mapped root (or nullptr if the root is removed).
Node bottom_up_map(Node root,
                   const std::function<Node(Node, const Node &)> &fn) {
  if (!root || !fn)
    return root;

  std::vector<Node> stack;
  Node result;

  root->traverse(
      [&](Node &current) {
        // Pre-order: allocate a partial node matching the current node's
        // header; children will be filled in on post-order.
        stack.push_back(NodeDef::create(current->type(), current->location()));
        return true;
      },
      [&](Node &current) {
        Node partial = std::move(stack.back());
        stack.pop_back();

        Node mapped = fn(partial, current);

        if (stack.empty()) {
          result = mapped;
          return;
        }

        if (mapped)
          stack.back()->push_back(mapped);
      });

  return result;
}

bool ast_has_cycle(Node root) {
  if (!root)
    return false;

  std::unordered_set<NodeDef *> in_path;
  bool has_cycle = false;

  root->traverse(
      [&](Node &current) {
        NodeDef *ptr = current.get();
        if (in_path.find(ptr) != in_path.end()) {
          has_cycle = true;
          return false; // Detected a cycle; do not descend further.
        }
        in_path.insert(ptr);
        return true;
      },
      [&](Node &current) {
        in_path.erase(current.get());
      });

  return has_cycle;
}

Nodes lookup_all(Node n) {
  Nodes result;
  Nodes includes = n->lookup();

  for (auto &inc : includes) {
    // If not a use statement, just keep it.
    if (inc != Use) {
      result.push_back(inc);
      continue;
    }

    // TODO: Initially assumed all children are just module names.
    Node start = inc->at(0)->at(0);

    bool first = true;
    for (auto &child : *inc->at(0)) {
      // Check the child is a module name.
      if (child->type() == Name) {
        // TODO Could be from earlier Use statement.
        Nodes child_module;
        if (first) {
          first = false;
          child_module = child->lookup();
          // Filter out Use nodes
          child_module.erase(
              std::remove_if(child_module.begin(), child_module.end(),
                             [](auto &n) { return n->type() == Use; }),
              child_module.end());
        } else {
          child_module = start->lookdown(child->location());
        }
        if (child_module.empty())
          continue;
        if (child_module.size() > 1) {
          // Error: module not found.
          abort();
        }
        if (child_module.front() != Module) {
          // Unhandled case: not a module.
          // TODO: Need to handle type aliases and structs.
          abort();
        }
        // TODO: Handle type parameters here.
        start = child_module.front();
        continue;
      }
      abort();
    }
    auto candidates = start->lookdown(n->location());
    result.insert(result.end(), candidates.begin(), candidates.end());
  }

  return result;
}

std::optional<size_t> lookup_levels_up(Node n) {
  size_t levels = 0;
  auto scope = n->scope();

  while (scope) {
    Nodes matches;
    scope->get_symbols(n->location(), matches, [&](auto &node) {
      return (node->type() & flag::lookup) &&
             (!(scope->type() & flag::defbeforeuse) || node->precedes(n));
    });

    if (!matches.empty())
      return levels;

    scope = scope->scope();
    ++levels;
  }

  return std::nullopt;
}

} // namespace infix