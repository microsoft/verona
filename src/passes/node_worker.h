#pragma once

#include <algorithm>
#include <deque>
#include <unordered_set>
#include <utility>
#include <vector>
#include <trieste/trieste.h>

namespace infix {
using namespace trieste;

/// Lifecycle states for nodes managed by `NodeWorker`.
enum class WorkerStatus {
  Uninitialized, ///< Not yet seen by the worker.
  Active,        ///< In the worklist, currently eligible for processing.
  Blocked,       ///< Waiting on one or more other nodes to resolve.
  Resolved,      ///< Completed successfully.
};

/// Per-node base state used by `NodeWorker`. Extend in `Work::State` for
/// algorithm-specific data.
struct NodeWorkerState {
  WorkerStatus kind{WorkerStatus::Uninitialized};
  /// Count of remaining prerequisites before unblocking. 0 means wake on the
  /// next signal.
  size_t blocked_on_count{0};
  /// Dependents that should be woken when this node resolves.
  std::unordered_set<Node> dependents;
};
/// Generic worklist driver for node-based passes with dependency blocking.
///
/// Usage overview:
///   1) Define a Work type with:
///        using State = YourState;                // YourState derives from NodeWorkerState
///        void seed(const Node&, State&);         // Initialize per-node state when first seen
///        bool process(const Node&, NodeWorker&); // Do one step; return true when resolved
///   2) Instantiate NodeWorker<Work> worker{Work{...}}.
///   3) Call worker.add(node) to enqueue roots.
///   4) Call worker.run() to drive the worklist until completion.
///   5) Inspect worker.states() for final results (Resolved/Blocked/etc.).
///
/// Blocking semantics:
///   - In process(), call block_on / block_on_all / block_on_any when a node
///     depends on other nodes. Those calls mark the current node Blocked and
///     register it as a dependent of the prerequisites. When a prerequisite
///     resolves, the worker automatically re-queues dependents.
///   - process() should return true only when the current node is resolved; if
///     it returns false, the node remains Active unless you explicitly blocked
///     it, so callers must set kind to Blocked via the block_on helpers.
///
/// Example (blocks on child nodes):
///   struct Work {
///     struct State : NodeWorkerState {
///       bool children_added{false};
///     };
///     void seed(const Node &, State &) {}
///     bool process(const Node &n, NodeWorker<Work> &worker) {
///       auto &s = worker.state(n);
///       if (!s.children_added && !n->empty()) {
///         // Block on all children; they get added implicitly.
///         std::vector<Node> children(n->begin(), n->end());
///         if (worker.block_on_all(n, children)) {
///           s.children_added = true;
///           return false; // will resume when children resolve
///         }
///       }
///       return true; // resolved once children are done (or none existed)
///     }
///   };
///   NodeWorker<Work> worker{Work{}};
///   worker.add(root);
///   worker.run();
template <typename Work>
class NodeWorker {
public:
  using State = typename Work::State;

  explicit NodeWorker(Work work) : work_(std::move(work)) {}

  /// Access existing state; the node must already have been added.
  State &state(const Node &n) { 
    auto& result = state_.at(n);
    return result;
  }

  const State &state(const Node &n) const {
    const auto& result = state_.at(n);
    return result;
  }

  bool is_resolved(const Node &n) const {
    auto it = state_.find(n);
    return it != state_.end() && it->second.kind == WorkerStatus::Resolved;
  }

  const NodeMap<State> &states() const { return state_; }

  /// Add a node to the worker if unseen; seeds its state and enqueues it.
  void add(const Node &n) {
    State &s = state_[n];
    if (s.kind != WorkerStatus::Uninitialized)
      return;

    work_.seed(n, s);
    s.kind = WorkerStatus::Active;
    worklist_.push_back(n);
  }

  /// Drive the worklist until no Active nodes remain. Nodes that block on
  /// others will be re-enqueued automatically when unblocked.
  void run() {
    while (!worklist_.empty()) {
      Node current = worklist_.front();
      worklist_.pop_front();

      auto &s = state(current);
      if (s.kind == WorkerStatus::Resolved) {
        continue;
      }

      assert(s.kind == WorkerStatus::Active);

      const bool done = work_.process(current, *this);
      if (done) {
        s.kind = WorkerStatus::Resolved;
        unblock_dependents(current);
      }
    }
  }

  /// Block the dependent on a single origin; returns true if blocking occurs.
  bool block_on(const Node &dependent, const Node &origin) {
    add(origin);
    if (is_resolved(origin)) {
      return false;
    }

    state(origin).dependents.insert(dependent);
    state(dependent).kind = WorkerStatus::Blocked;
    return true;
  }

  /// Block until all origins resolve; returns true if any blocking was needed.
  bool block_on_all(const Node &dependent, const std::vector<Node> &origins) {
    size_t count = 0;
    for (const auto &origin : origins) {
      if (block_on(dependent, origin)) {
        count++;
      }
    }

    if (count == 0)
      return false;

    count--;
    size_t &blocked_on_count = state(dependent).blocked_on_count;
    if (blocked_on_count == 0) {
      blocked_on_count = count;
    } else {
      blocked_on_count = std::min(blocked_on_count, count);
    }
    return true;
  }

  /// Block until any origin resolves; returns true if any blocking was needed.
  bool block_on_any(const Node &dependent, const std::vector<Node> &origins) {
    bool has_blocking = false;
    for (const auto &origin : origins) {
      has_blocking |= block_on(dependent, origin);
    }
    state(dependent).blocked_on_count = 0;
    return has_blocking;
  }

private:
  void unblock_dependents(const Node &origin) {
    auto &waiting = state(origin).dependents;
    if (waiting.empty()) {
      return;
    }

    for (const auto &dependent : waiting) {
      auto &s = state(dependent);
      if (s.kind != WorkerStatus::Blocked) {
        continue;
      }

      if (s.blocked_on_count > 0) {
        s.blocked_on_count--;
        continue;
      }

      s.kind = WorkerStatus::Active;
      worklist_.push_back(dependent);
    }

    waiting.clear();
  }

  NodeMap<State> state_;
  std::deque<Node> worklist_;
  Work work_;
};

} // namespace infix
