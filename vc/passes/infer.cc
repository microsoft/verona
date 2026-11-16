#include "../lang.h"
#include "../subtype.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <format>
#include <iostream>
#include <unordered_set>

namespace vc
{
  static bool is_lambda_function(const Node& func);

  std::unordered_set<const void*> lambda_returns_omitted;

  static bool lambda_return_was_omitted(const Node& func)
  {
    return lambda_returns_omitted.count(func.get()) > 0;
  }

  // ===== Dispatch tables =====

  const std::map<std::string_view, Token> primitive_from_name = {
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

  const std::map<std::string_view, Token> ffi_primitive_from_name = {
    {"ptr", Ptr}};

  const std::initializer_list<Token> integer_types = {
    I8, I16, I32, I64, U8, U16, U32, U64, ILong, ULong, ISize, USize};

  const std::initializer_list<Token> float_types = {F32, F64};

  // Binary ops: result type = LHS type.
  const std::initializer_list<Token> propagate_lhs_ops = {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Pow,
    And,
    Or,
    Xor,
    Shl,
    Shr,
    Min,
    Max,
    LogBase,
    Atan2};

  // Unary ops: result type = operand type.
  const std::initializer_list<Token> propagate_rhs_ops = {
    Neg,  Abs,  Ceil, Floor, Exp,  Log,  Sqrt,  Cbrt,  Sin,   Cos,  Tan,
    Asin, Acos, Atan, Sinh,  Cosh, Tanh, Asinh, Acosh, Atanh, Read, Freeze};

  // Ops with fixed result types.
  const std::map<Token, Token> fixed_result_type = {
    {Eq, Bool},        {Ne, Bool},          {Lt, Bool},
    {Le, Bool},        {Gt, Bool},          {Ge, Bool},
    {IsInf, Bool},     {IsNaN, Bool},       {Not, Bool},
    {Bits, U64},       {Len, USize},        {Const_E, F64},
    {Const_Pi, F64},   {Const_Inf, F64},    {Const_NaN, F64},
    {GetRaise, U64},   {SetRaise, U64},     {FreeCallback, None},
    {Pin, None},       {Unpin, None},       {Merge, None},
    {FFIStore, None},  {AddExternal, None}, {RemoveExternal, None},
    {ArrayCopy, None}, {ArrayFill, None},   {ArrayCompare, I64},
  };

  const std::map<Token, Token> fixed_ffi_result_type = {
    {MakePtr, Ptr},
    {MakeCallback, Ptr},
    {CodePtrCallback, Ptr},
  };

  // ===== Type constructors =====

  Node primitive_type(const Token& tok)
  {
    return Type
      << (TypeName << (NameElement << (Ident ^ "_builtin") << TypeArgs)
                   << (NameElement << (Ident ^ tok.str()) << TypeArgs));
  }

  Node ffi_primitive_type(const Token& tok)
  {
    return Type
      << (TypeName << (NameElement << (Ident ^ "_builtin") << TypeArgs)
                   << (NameElement << (Ident ^ "ffi") << TypeArgs)
                   << (NameElement << (Ident ^ tok.str()) << TypeArgs));
  }

  Node primitive_or_ffi_type(const Token& tok)
  {
    if (tok == Ptr)
      return ffi_primitive_type(tok);

    return primitive_type(tok);
  }

  Node string_type()
  {
    return Type
      << (TypeName << (NameElement << (Ident ^ "_builtin") << TypeArgs)
                   << (NameElement << (Ident ^ "string") << TypeArgs));
  }

  Node ref_type(const Node& inner)
  {
    return Type
      << (TypeName << (NameElement << (Ident ^ "_builtin") << TypeArgs)
                   << (NameElement << (Ident ^ "ref")
                                   << (TypeArgs << clone(inner))));
  }

  Node cown_type(const Node& inner)
  {
    return Type
      << (TypeName << (NameElement << (Ident ^ "_builtin") << TypeArgs)
                   << (NameElement << (Ident ^ "cown")
                                   << (TypeArgs << clone(inner))));
  }

  // ===== Type predicates =====

  bool is_default_type(const Node& type)
  {
    return type && !type->empty() &&
      type->front()->in({DefaultInt, DefaultFloat});
  }

  bool contains_default_type(const Node& type)
  {
    if (!type)
      return false;

    bool found = false;
    type->traverse([&](auto node) {
      if (node->in({DefaultInt, DefaultFloat}))
      {
        found = true;
        return false;
      }
      return !found;
    });
    return found;
  }

  Node resolve_default(const Node& type)
  {
    if (!type || type->empty())
      return type;
    if (type->front() == DefaultInt)
      return primitive_type(U64);
    if (type->front() == DefaultFloat)
      return primitive_type(F64);
    return type;
  }

  bool is_any_type(const Node& type)
  {
    if (type != Type)
      return false;
    auto inner = type->front();
    if (inner != TypeName || inner->size() != 2)
      return false;
    if ((inner->front() / Ident)->location().view() != "_builtin")
      return false;
    return (inner->back() / Ident)->location().view() == "any";
  }

  bool is_dyn_type(const Node& type)
  {
    if (type != Type)
      return false;
    auto inner = type->front();
    if (inner != TypeName || inner->size() != 2)
      return false;
    if ((inner->front() / Ident)->location().view() != "_builtin")
      return false;
    return (inner->back() / Ident)->location().view() == "dyn";
  }

  bool is_uninformative_backward_type(const Node& type)
  {
    return is_any_type(type) || is_dyn_type(type) ||
      (type == Type && !type->empty() && type->front() == TypeSelf);
  }

  Node extract_wrapper_inner(const Node& type_node, std::string_view wrapper)
  {
    if (type_node != Type)
      return {};
    auto inner = type_node->front();
    if (inner != TypeName || inner->size() != 2)
      return {};
    if ((inner->front() / Ident)->location().view() != "_builtin")
      return {};
    if ((inner->back() / Ident)->location().view() != wrapper)
      return {};
    auto ta = inner->back() / TypeArgs;
    if (ta->size() != 1)
      return {};
    return clone(ta->front());
  }

  Node extract_ref_inner(const Node& type_node)
  {
    return extract_wrapper_inner(type_node, "ref");
  }

  Node extract_cown_inner(const Node& type_node)
  {
    return extract_wrapper_inner(type_node, "cown");
  }

  Node extract_primitive(const Node& type_node)
  {
    if (type_node != Type)
      return {};
    auto inner = type_node->front();
    if (inner == DefaultInt)
      return U64;
    if (inner == DefaultFloat)
      return F64;
    if (inner != TypeName)
      return {};
    auto first = (inner->front() / Ident)->location().view();
    if (first != "_builtin")
      return {};
    if (inner->size() == 2)
    {
      auto it =
        primitive_from_name.find((inner->back() / Ident)->location().view());
      return (it != primitive_from_name.end()) ? Node{it->second} : Node{};
    }
    if (inner->size() == 3)
    {
      if ((inner->at(1) / Ident)->location().view() != "ffi")
        return {};
      auto it =
        ffi_primitive_from_name.find((inner->at(2) / Ident)->location().view());
      return (it != ffi_primitive_from_name.end()) ? Node{it->second} : Node{};
    }
    return {};
  }

  Node extract_callable_primitive(const Node& type_node)
  {
    auto prim = extract_primitive(type_node);
    if (prim)
      return prim;
    if (type_node != Type)
      return {};
    auto inner = type_node->front();
    if (!inner->in({Union, Isect}))
      return {};
    Node candidate;
    for (auto& component : *inner)
    {
      auto p = extract_primitive(Type << clone(component));
      if (!p)
        continue;
      if (candidate && candidate->type() != p->type())
        return {};
      candidate = p;
    }
    return candidate;
  }

  Node exclude_tested_type(
    Node top, const Node& source_type, const Node& tested_type)
  {
    if (source_type != Type || tested_type != Type)
      return {};

    auto inner = source_type->front();
    if (inner != Union)
      return {};

    SequentCtx ctx{top, {}, {}};
    Node remaining = Union;
    for (auto& component : *inner)
    {
      auto component_type = Type << clone(component);
      if (!Subtype(ctx, component_type, tested_type))
        remaining << clone(component);
    }

    if (remaining->empty() || remaining->size() == inner->size())
      return {};

    return (remaining->size() == 1) ? Type << clone(remaining->front()) :
                                      Type << remaining;
  }

  Node extract_wrapper_primitive(const Node& type_node)
  {
    auto inner = extract_ref_inner(type_node);
    if (!inner)
      inner = extract_cown_inner(type_node);
    return inner ? extract_callable_primitive(inner) : Node{};
  }

  Node default_literal_type(const Node& lit)
  {
    if (lit->in({True, False}))
      return Bool;
    if (lit == None)
      return None;
    if (lit->in({Bin, Oct, Int, Hex, Char}))
      return DefaultInt;
    if (lit->in({Float, HexFloat}))
      return DefaultFloat;
    assert(false && "unhandled literal type");
    return {};
  }

  static bool contains_typevar(const Node& type_node)
  {
    if (!type_node)
      return false;
    bool found = false;
    type_node->traverse([&](auto node) {
      if (node == TypeVar)
        found = true;
      return !found;
    });
    return found;
  }

  static bool contains_self_type(const Node& type_node)
  {
    if (!type_node)
      return false;
    bool found = false;
    type_node->traverse([&](auto node) {
      if (node == TypeSelf)
        found = true;
      return !found;
    });
    return found;
  }

  Node direct_typeparam(Node top, const Node& type_node)
  {
    if (type_node != Type)
      return {};
    auto inner = type_node->front();
    if (inner != TypeName)
      return {};
    auto def = find_def(top, inner);
    return (def && def == TypeParam) ? def : Node{};
  }

  Node lookup_method_name(const Node& stmt)
  {
    Node ident = stmt / Ident;
    if (ident)
      return ident;
    return stmt / SymbolId;
  }

  using InferClock = std::chrono::steady_clock;

  struct InferProfileStats
  {
    std::string function_name;
    size_t labels = 0;
    size_t worklist_iterations = 0;
    size_t labels_processed = 0;
    size_t labels_skipped = 0;
    size_t entry_bindings = 0;
    size_t process_label_body_calls = 0;
    size_t dependency_cascade_calls = 0;
    size_t entry_update_calls = 0;
    size_t resolve_method_calls = 0;
    size_t resolve_method_cache_hits = 0;
    size_t resolve_method_cache_misses = 0;
    size_t resolve_method_scans = 0;
    size_t merge_type_calls = 0;
    size_t merge_env_calls = 0;
    size_t merge_type_existing_typevar = 0;
    size_t merge_type_both_typevar = 0;
    size_t merge_type_incoming_typevar = 0;
    size_t merge_type_pointer_equal = 0;
    size_t merge_type_structural_equal = 0;
    size_t merge_type_default_promote = 0;
    size_t merge_type_default_keep = 0;
    size_t merge_type_invariant_equal = 0;
    size_t merge_type_subtype_keep = 0;
    size_t merge_type_subtype_widen = 0;
    size_t merge_type_union_build = 0;
    size_t same_type_tree_calls = 0;
    size_t same_type_tree_equal = 0;
    size_t same_type_env_calls = 0;
    size_t same_type_env_equal = 0;
    size_t navigate_call_calls = 0;
    size_t apply_subst_calls = 0;
    size_t infer_typeargs_calls = 0;
    size_t push_arg_types_to_params_calls = 0;
    size_t propagate_call_node_calls = 0;
    size_t stmt_const_like = 0;
    size_t stmt_copy_like = 0;
    size_t stmt_ref_ops = 0;
    size_t stmt_tuple_ops = 0;
    size_t stmt_new_ops = 0;
    size_t stmt_call_ops = 0;
    size_t stmt_ffi_when_ops = 0;
    size_t stmt_typetest_ops = 0;
    size_t tuple_finalize_ops = 0;
    size_t requeue_succ = 0;
    size_t requeue_self_bwd = 0;
    size_t requeue_pred_bwd = 0;
    size_t entry_pred_merge_count = 0;
    size_t entry_self_bwd_merge_count = 0;
    size_t entry_succ_bwd_merge_count = 0;
    size_t entry_env_unchanged = 0;
    size_t entry_env_changed = 0;
    size_t cascade_env_bindings = 0;
    size_t cascade_body_statements = 0;
    size_t cascade_src_entries = 0;
    size_t cascade_const_defs = 0;
    size_t cascade_seed_locs = 0;
    size_t cascade_seed_type_nodes = 0;
    size_t cascade_max_type_nodes = 0;
    size_t cascade_work_pops = 0;
    size_t cascade_stmt_visits = 0;
    InferClock::duration process_function_time{};
    InferClock::duration entry_build_time{};
    InferClock::duration entry_pred_time{};
    InferClock::duration entry_self_bwd_time{};
    InferClock::duration entry_succ_bwd_time{};
    InferClock::duration process_label_body_time{};
    InferClock::duration resolve_method_time{};
    InferClock::duration dependency_cascade_time{};
    InferClock::duration entry_update_time{};
    InferClock::duration same_type_tree_time{};
    InferClock::duration same_type_env_time{};
    InferClock::duration navigate_call_time{};
    InferClock::duration apply_subst_time{};
    InferClock::duration infer_typeargs_time{};
    InferClock::duration push_arg_types_to_params_time{};
    InferClock::duration propagate_call_node_time{};
    InferClock::duration stmt_const_like_time{};
    InferClock::duration stmt_copy_like_time{};
    InferClock::duration stmt_ref_ops_time{};
    InferClock::duration stmt_tuple_ops_time{};
    InferClock::duration stmt_new_ops_time{};
    InferClock::duration stmt_call_ops_time{};
    InferClock::duration stmt_ffi_when_ops_time{};
    InferClock::duration stmt_typetest_ops_time{};
    InferClock::duration tuple_finalize_time{};
    InferClock::duration cascade_index_time{};
    InferClock::duration cascade_const_scan_time{};
    InferClock::duration cascade_seed_time{};
    InferClock::duration cascade_loop_time{};
  };

  static InferProfileStats* active_infer_profile = nullptr;
  static size_t* active_infer_transfer_epoch = nullptr;

  static bool infer_profile_enabled()
  {
    static const bool enabled = (std::getenv("VC_INFER_PROFILE") != nullptr);
    return enabled;
  }

  static double infer_duration_ms(const InferClock::duration& duration)
  {
    return std::chrono::duration<double, std::milli>(duration).count();
  }

  struct InferScopedTimer
  {
    InferClock::duration* slot;
    InferClock::time_point start;

    explicit InferScopedTimer(InferClock::duration* slot_) : slot(slot_)
    {
      if (slot != nullptr)
        start = InferClock::now();
    }

    ~InferScopedTimer()
    {
      if (slot != nullptr)
        *slot += InferClock::now() - start;
    }
  };

  static std::string infer_function_name(const Node& func)
  {
    std::string result;

    for (auto& scope : scope_path(func))
    {
      Node ident = scope / Ident;
      if (!ident)
        continue;

      if (!result.empty())
        result += "::";
      result += std::string(ident->location().view());
    }

    if (result.empty())
      result = "<function>";

    auto loc = func->location();
    if (loc.source && !loc.source->origin().empty())
      result += std::format("@{}", loc.source->origin());
    return result;
  }

  static size_t infer_tree_size(const Node& node)
  {
    if (!node)
      return 0;

    size_t count = 1;
    for (auto& child : *node)
      count += infer_tree_size(child);
    return count;
  }

  static void dump_infer_profile(const InferProfileStats& stats)
  {
    std::cerr
      << "infer-profile"
      << "\tfunc=" << stats.function_name << "\tlabels=" << stats.labels
      << "\twl_iters=" << stats.worklist_iterations
      << "\tlabel_runs=" << stats.labels_processed
      << "\tlabel_skips=" << stats.labels_skipped
      << "\tentry_bindings=" << stats.entry_bindings
      << "\tmerge_type=" << stats.merge_type_calls
      << "\tmerge_env=" << stats.merge_env_calls
      << "\tmt_exist_tv=" << stats.merge_type_existing_typevar
      << "\tmt_both_tv=" << stats.merge_type_both_typevar
      << "\tmt_in_tv=" << stats.merge_type_incoming_typevar
      << "\tmt_ptr_eq=" << stats.merge_type_pointer_equal
      << "\tmt_struct_eq=" << stats.merge_type_structural_equal
      << "\tmt_def_promote=" << stats.merge_type_default_promote
      << "\tmt_def_keep=" << stats.merge_type_default_keep
      << "\tmt_invariant=" << stats.merge_type_invariant_equal
      << "\tmt_sub_keep=" << stats.merge_type_subtype_keep
      << "\tmt_sub_widen=" << stats.merge_type_subtype_widen
      << "\tmt_union=" << stats.merge_type_union_build
      << "\tst_tree=" << stats.same_type_tree_calls
      << "\tst_tree_eq=" << stats.same_type_tree_equal
      << "\tst_env=" << stats.same_type_env_calls
      << "\tst_env_eq=" << stats.same_type_env_equal
      << "\tresolve=" << stats.resolve_method_calls
      << "\tresolve_hit=" << stats.resolve_method_cache_hits
      << "\tresolve_miss=" << stats.resolve_method_cache_misses
      << "\tresolve_scan=" << stats.resolve_method_scans
      << "\tnavigate_call=" << stats.navigate_call_calls
      << "\tapply_subst=" << stats.apply_subst_calls
      << "\tinfer_typeargs=" << stats.infer_typeargs_calls
      << "\tpush_arg_types=" << stats.push_arg_types_to_params_calls
      << "\tprop_call_node=" << stats.propagate_call_node_calls
      << "\tbody_calls=" << stats.process_label_body_calls
      << "\tcascade_calls=" << stats.dependency_cascade_calls
      << "\tentry_updates=" << stats.entry_update_calls
      << "\tstmt_const=" << stats.stmt_const_like
      << "\tstmt_copy=" << stats.stmt_copy_like
      << "\tstmt_ref=" << stats.stmt_ref_ops
      << "\tstmt_tuple=" << stats.stmt_tuple_ops
      << "\tstmt_new=" << stats.stmt_new_ops
      << "\tstmt_call=" << stats.stmt_call_ops
      << "\tstmt_ffi_when=" << stats.stmt_ffi_when_ops
      << "\tstmt_typetest=" << stats.stmt_typetest_ops
      << "\ttuple_finalize=" << stats.tuple_finalize_ops
      << "\treq_succ=" << stats.requeue_succ
      << "\treq_self_bwd=" << stats.requeue_self_bwd
      << "\treq_pred_bwd=" << stats.requeue_pred_bwd
      << "\tentry_pred_merges=" << stats.entry_pred_merge_count
      << "\tentry_self_bwd_merges=" << stats.entry_self_bwd_merge_count
      << "\tentry_succ_bwd_merges=" << stats.entry_succ_bwd_merge_count
      << "\tentry_same=" << stats.entry_env_unchanged
      << "\tentry_diff=" << stats.entry_env_changed
      << "\tcascade_env=" << stats.cascade_env_bindings
      << "\tcascade_body=" << stats.cascade_body_statements
      << "\tcascade_src=" << stats.cascade_src_entries
      << "\tcascade_consts=" << stats.cascade_const_defs
      << "\tcascade_seed=" << stats.cascade_seed_locs
      << "\tcascade_seed_nodes=" << stats.cascade_seed_type_nodes
      << "\tcascade_max_nodes=" << stats.cascade_max_type_nodes
      << "\tcascade_pops=" << stats.cascade_work_pops
      << "\tcascade_visits=" << stats.cascade_stmt_visits
      << "\ttotal_ms=" << infer_duration_ms(stats.process_function_time)
      << "\tentry_ms=" << infer_duration_ms(stats.entry_build_time)
      << "\tentry_pred_ms=" << infer_duration_ms(stats.entry_pred_time)
      << "\tentry_self_bwd_ms=" << infer_duration_ms(stats.entry_self_bwd_time)
      << "\tentry_succ_bwd_ms=" << infer_duration_ms(stats.entry_succ_bwd_time)
      << "\tentry_update_ms=" << infer_duration_ms(stats.entry_update_time)
      << "\tst_tree_ms=" << infer_duration_ms(stats.same_type_tree_time)
      << "\tst_env_ms=" << infer_duration_ms(stats.same_type_env_time)
      << "\tbody_ms=" << infer_duration_ms(stats.process_label_body_time)
      << "\tconst_ms=" << infer_duration_ms(stats.stmt_const_like_time)
      << "\tcopy_ms=" << infer_duration_ms(stats.stmt_copy_like_time)
      << "\tref_ms=" << infer_duration_ms(stats.stmt_ref_ops_time)
      << "\ttuple_ms=" << infer_duration_ms(stats.stmt_tuple_ops_time)
      << "\tnew_ms=" << infer_duration_ms(stats.stmt_new_ops_time)
      << "\tcall_ms=" << infer_duration_ms(stats.stmt_call_ops_time)
      << "\tffi_when_ms=" << infer_duration_ms(stats.stmt_ffi_when_ops_time)
      << "\ttypetest_ms=" << infer_duration_ms(stats.stmt_typetest_ops_time)
      << "\ttuple_finalize_ms=" << infer_duration_ms(stats.tuple_finalize_time)
      << "\tcascade_index_ms=" << infer_duration_ms(stats.cascade_index_time)
      << "\tcascade_const_scan_ms="
      << infer_duration_ms(stats.cascade_const_scan_time)
      << "\tcascade_seed_ms=" << infer_duration_ms(stats.cascade_seed_time)
      << "\tcascade_loop_ms=" << infer_duration_ms(stats.cascade_loop_time)
      << "\tresolve_ms=" << infer_duration_ms(stats.resolve_method_time)
      << "\tnavigate_call_ms=" << infer_duration_ms(stats.navigate_call_time)
      << "\tapply_subst_ms=" << infer_duration_ms(stats.apply_subst_time)
      << "\tinfer_typeargs_ms=" << infer_duration_ms(stats.infer_typeargs_time)
      << "\tpush_arg_types_ms="
      << infer_duration_ms(stats.push_arg_types_to_params_time)
      << "\tprop_call_node_ms="
      << infer_duration_ms(stats.propagate_call_node_time)
      << "\tcascade_ms=" << infer_duration_ms(stats.dependency_cascade_time)
      << '\n';
  }

  // ===== Type environment =====

  struct LocalTypeInfo
  {
    Node type;
    bool is_fixed;
    Node call_node; // Call/CallDyn that produced this (for cross-function prop)
  };

  using TypeEnv = std::map<Location, LocalTypeInfo>;

  struct PendingError
  {
    Node site;
    std::string msg;
  };

  static std::map<Location, PendingError> deferred_param_errors;

  // ===== Type lattice: merge =====

  // Forward declaration for use in extract_constraints.
  Node apply_subst(Node top, const Node& type_node, const NodeMap<Node>& subst);

  struct MethodInfo
  {
    Node func;
    NodeMap<Node> subst;
  };

  struct MethodOwner
  {
    Node class_def;
    Node subst_source;
    std::string key;
  };

  struct MethodLookupKey
  {
    std::string owner_key;
    std::string method_name;
    std::string hand_name;
    size_t arity;

    auto operator<=>(const MethodLookupKey&) const = default;
  };

  using MethodLookupCache = std::map<MethodLookupKey, Node>;

  static MethodLookupCache* active_method_cache = nullptr;

  struct InferProcessScope
  {
    InferProfileStats* current_profile;
    InferProfileStats* prev_profile;
    MethodLookupCache* prev_method_cache;
    size_t* prev_transfer_epoch;

    InferProcessScope(
      InferProfileStats* current_profile_,
      InferProfileStats* prev_profile_,
      MethodLookupCache* prev_method_cache_,
      size_t* prev_transfer_epoch_)
    : current_profile(current_profile_),
      prev_profile(prev_profile_),
      prev_method_cache(prev_method_cache_),
      prev_transfer_epoch(prev_transfer_epoch_)
    {}

    ~InferProcessScope()
    {
      active_method_cache = prev_method_cache;
      active_infer_transfer_epoch = prev_transfer_epoch;
      if (current_profile != nullptr)
        dump_infer_profile(*current_profile);
      active_infer_profile = prev_profile;
    }
  };

  MethodInfo resolve_method(
    Node top,
    const Node& receiver_type,
    const Node& method_ident,
    Token hand,
    size_t arity,
    const Node& method_typeargs);
  Node resolve_method_return_type(
    Node top,
    const Node& receiver_type,
    const Node& method_ident,
    Token hand,
    size_t arity,
    const Node& method_typeargs);
  MethodInfo resolve_callable_method(
    Node top,
    const Node& receiver_type,
    const Node& method_ident,
    Token hand,
    size_t arity,
    const Node& method_typeargs);

  static std::string typename_path_key(const Node& type_name)
  {
    if (type_name != TypeName)
      return {};

    std::string key;
    for (auto& elem : *type_name)
    {
      if (!key.empty())
        key += "::";
      key += std::string((elem / Ident)->location().view());
    }
    return key;
  }

  static std::string infer_type_name(const Node& type)
  {
    if (!type)
      return "<unknown>";
    if (type == Type)
    {
      if (!type->empty())
        return infer_type_name(type->front());
      return "type";
    }
    if (type == TypeName)
    {
      auto key = typename_path_key(type);
      if (!key.empty())
        return key;
    }
    if (type == Union)
    {
      std::string result = "Union(";
      bool first = true;
      for (auto& child : *type)
      {
        if (!first)
          result += ", ";
        result += infer_type_name(child);
        first = false;
      }
      result += ")";
      return result;
    }
    return std::string(type->type().str());
  }

  static MethodOwner resolve_method_owner(Node top, const Node& receiver_type)
  {
    if (receiver_type != Type)
      return {};

    auto inner = receiver_type->front();
    if (inner != TypeName)
      return {};

    auto class_def = find_def(top, inner);
    Node subst_source = inner;

    // Follow TypeAlias chains to the underlying ClassDef.
    while (class_def == TypeAlias)
    {
      auto alias_type = class_def / Type;
      if (alias_type->front() != TypeName)
        break;
      class_def = find_def(top, alias_type->front());
      if (!class_def)
        return {};
    }

    if ((!class_def || class_def != ClassDef) && inner->size() > 1)
    {
      Node base = TypeName;
      for (size_t i = 0; i + 1 < inner->size(); i++)
        base << clone(inner->at(i));

      auto base_def = find_def(top, base);
      if (base_def && base_def == ClassDef)
      {
        class_def = base_def;
        subst_source = base;
      }
    }

    if (!class_def || class_def != ClassDef)
      return {};

    return {class_def, subst_source, typename_path_key(subst_source)};
  }

  static Node resolve_class_method(
    const MethodOwner& owner,
    std::string_view method_name,
    Token hand,
    size_t arity)
  {
    if (!owner.class_def)
      return {};

    MethodLookupKey key{
      owner.key, std::string(method_name), std::string(hand.str()), arity};

    if (active_method_cache != nullptr)
    {
      auto it = active_method_cache->find(key);
      if (it != active_method_cache->end())
      {
        if (active_infer_profile != nullptr)
          active_infer_profile->resolve_method_cache_hits++;
        return it->second;
      }
    }

    if (active_infer_profile != nullptr)
    {
      active_infer_profile->resolve_method_cache_misses++;
      active_infer_profile->resolve_method_scans++;
    }

    Node func;
    for (auto& child : *(owner.class_def / ClassBody))
    {
      if (child != Function)
        continue;

      auto child_name = lookup_method_name(child);
      if (!child_name || child_name->location().view() != method_name)
        continue;
      if ((child / Lhs)->type() != hand)
        continue;
      if ((child / Params)->size() != arity)
        continue;

      func = child;
      break;
    }

    if (active_method_cache != nullptr)
      (*active_method_cache)[std::move(key)] = func;

    return func;
  }

  static bool same_type_tree(const Node& left, const Node& right)
  {
    if (active_infer_profile != nullptr)
      active_infer_profile->same_type_tree_calls++;
    InferScopedTimer timer(
      (active_infer_profile != nullptr) ?
        &active_infer_profile->same_type_tree_time :
        nullptr);

    if (left == right)
    {
      if (active_infer_profile != nullptr)
        active_infer_profile->same_type_tree_equal++;
      return true;
    }
    if (!left || !right)
      return false;
    if (left->type() != right->type())
      return false;
    if (left->size() != right->size())
      return false;

    if (left == Ident)
      return left->location().view() == right->location().view();

    for (size_t i = 0; i < left->size(); i++)
      if (!same_type_tree(left->at(i), right->at(i)))
        return false;

    if (active_infer_profile != nullptr)
      active_infer_profile->same_type_tree_equal++;
    return true;
  }

  static bool same_type_env(const TypeEnv& left, const TypeEnv& right)
  {
    if (active_infer_profile != nullptr)
      active_infer_profile->same_type_env_calls++;
    InferScopedTimer timer(
      (active_infer_profile != nullptr) ?
        &active_infer_profile->same_type_env_time :
        nullptr);

    if (left.size() != right.size())
      return false;

    auto lit = left.begin();
    auto rit = right.begin();
    while (lit != left.end())
    {
      if (lit->first != rit->first)
        return false;
      if (lit->second.is_fixed != rit->second.is_fixed)
        return false;
      if (lit->second.call_node != rit->second.call_node)
        return false;
      if (!same_type_tree(lit->second.type, rit->second.type))
        return false;
      ++lit;
      ++rit;
    }

    if (active_infer_profile != nullptr)
      active_infer_profile->same_type_env_equal++;
    return true;
  }

  static void note_infer_transfer_change()
  {
    if (active_infer_transfer_epoch != nullptr)
      (*active_infer_transfer_epoch)++;
  }

  static bool replace_if_changed(
    const Node& owner, const Node& old_child, const Node& new_child)
  {
    if (same_type_tree(old_child, new_child))
      return false;
    owner->replace(old_child, new_child);
    note_infer_transfer_change();
    return true;
  }

  static bool refine_const_local(
    TypeEnv& env,
    const std::map<Location, Node>& const_defs,
    const Location& loc,
    const Node& expected)
  {
    auto const_it = const_defs.find(loc);
    if (const_it == const_defs.end())
      return false;

    auto env_it = env.find(loc);
    if (env_it == env.end())
      return false;

    auto expected_prim = extract_primitive(expected);
    if (!expected_prim)
      return false;

    auto current_prim = extract_primitive(env_it->second.type);
    bool compatible = (env_it->second.type->front() == DefaultInt &&
                       expected_prim->in(integer_types)) ||
      (env_it->second.type->front() == DefaultFloat &&
       expected_prim->in(float_types)) ||
      (current_prim && current_prim->in(integer_types) &&
       expected_prim->in(integer_types)) ||
      (current_prim && current_prim->in(float_types) &&
       expected_prim->in(float_types));
    if (!compatible)
      return false;

    env_it->second.type = primitive_or_ffi_type(expected_prim->type());
    auto const_stmt = const_it->second;
    auto refine_const_stmt = [&](const Node& stmt) {
      if (stmt->size() == 3)
      {
        auto old_type = stmt->at(1);
        if (old_type->type() != expected_prim->type())
        {
          stmt->replace(old_type, expected_prim->type());
          note_infer_transfer_change();
          return true;
        }
      }
      else
      {
        auto dst = stmt->front();
        auto lit = stmt->back();
        stmt->erase(stmt->begin(), stmt->end());
        stmt << dst << expected_prim->type() << lit;
        note_infer_transfer_change();
        return true;
      }
      return false;
    };

    refine_const_stmt(const_stmt);
    return true;
  }

  static bool upsert_lookup_stmt(
    std::map<Location, Node>& lookup_stmts,
    const Location& loc,
    const Node& stmt)
  {
    auto [it, inserted] = lookup_stmts.insert({loc, stmt});
    if (inserted)
    {
      note_infer_transfer_change();
      return true;
    }
    if (it->second != stmt)
    {
      it->second = stmt;
      note_infer_transfer_change();
      return true;
    }
    return false;
  }

  static bool upsert_ref_to_tuple(
    std::map<Location, std::pair<Location, size_t>>& ref_to_tuple,
    const Location& loc,
    const std::pair<Location, size_t>& value)
  {
    auto [it, inserted] = ref_to_tuple.insert({loc, value});
    if (inserted)
    {
      note_infer_transfer_change();
      return true;
    }
    if (it->second != value)
    {
      it->second = value;
      note_infer_transfer_change();
      return true;
    }
    return false;
  }

  // Returns the merged type, or {} if no change from existing.
  static Node merge_type(const Node& existing, const Node& incoming, Node top)
  {
    if (active_infer_profile != nullptr)
      active_infer_profile->merge_type_calls++;

    if (!existing || existing->empty() || existing->front() == TypeVar)
    {
      if (active_infer_profile != nullptr)
        active_infer_profile->merge_type_existing_typevar++;

      // Both TypeVar → no change.
      if (incoming && !incoming->empty() && incoming->front() == TypeVar)
      {
        if (active_infer_profile != nullptr)
          active_infer_profile->merge_type_both_typevar++;
        return {};
      }
      return clone(incoming);
    }
    if (!incoming || incoming->empty() || incoming->front() == TypeVar)
    {
      if (active_infer_profile != nullptr)
        active_infer_profile->merge_type_incoming_typevar++;
      return {};
    }

    if (existing == incoming)
    {
      if (active_infer_profile != nullptr)
        active_infer_profile->merge_type_pointer_equal++;
      return {};
    }
    if (same_type_tree(existing, incoming))
    {
      if (active_infer_profile != nullptr)
        active_infer_profile->merge_type_structural_equal++;
      return {};
    }

    // Default yields to compatible concrete primitive.
    // Checked BEFORE Subtype.invariant because AxiomFalse makes
    // Subtype consider DefaultInt invariant with i32.
    if (is_default_type(existing) && !is_default_type(incoming))
    {
      auto prim = extract_primitive(incoming);
      if (prim)
      {
        bool compat =
          (existing->front() == DefaultInt && prim->in(integer_types)) ||
          (existing->front() == DefaultFloat && prim->in(float_types));
        if (compat)
        {
          if (active_infer_profile != nullptr)
            active_infer_profile->merge_type_default_promote++;
          return clone(incoming);
        }
      }

      // Default yields to a compatible primitive member of a union.
      // e.g., DefaultInt + Union(usize, none) → usize.
      if (incoming->front() == Union)
      {
        for (auto& member : *(incoming->front()))
        {
          auto mt = Type << clone(member);
          auto mp = extract_primitive(mt);
          if (mp)
          {
            bool compat =
              (existing->front() == DefaultInt && mp->in(integer_types)) ||
              (existing->front() == DefaultFloat && mp->in(float_types));
            if (compat)
            {
              if (active_infer_profile != nullptr)
                active_infer_profile->merge_type_default_promote++;
              return mt;
            }
          }
        }
      }
    }

    if (is_default_type(incoming) && !is_default_type(existing))
    {
      auto prim = extract_primitive(existing);
      if (prim)
      {
        bool compat =
          (incoming->front() == DefaultInt && prim->in(integer_types)) ||
          (incoming->front() == DefaultFloat && prim->in(float_types));
        if (compat)
        {
          if (active_infer_profile != nullptr)
            active_infer_profile->merge_type_default_keep++;
          return {};
        }
      }
    }

    bool existing_has_self = contains_self_type(existing);
    bool incoming_has_self = contains_self_type(incoming);
    if (existing_has_self != incoming_has_self)
    {
      if (existing_has_self)
        return clone(incoming);
      return {};
    }

    SequentCtx ctx{top, {}, {}};

    if (Subtype.invariant(ctx, existing, incoming))
    {
      if (active_infer_profile != nullptr)
        active_infer_profile->merge_type_invariant_equal++;
      return {};
    }

    if (Subtype(ctx, incoming, existing))
    {
      if (active_infer_profile != nullptr)
        active_infer_profile->merge_type_subtype_keep++;
      return {};
    }
    if (Subtype(ctx, existing, incoming))
    {
      if (active_infer_profile != nullptr)
        active_infer_profile->merge_type_subtype_widen++;
      return clone(incoming);
    }

    if (active_infer_profile != nullptr)
      active_infer_profile->merge_type_union_build++;

    // Build union.
    auto e_inner = existing->front();
    auto i_inner = incoming->front();
    Node u = Union;

    if (e_inner == Union)
      for (auto& c : *e_inner)
        u << clone(c);
    else
      u << clone(e_inner);

    // If incoming is a concrete primitive, replace compatible Default
    // members in the union (DefaultInt→integer, DefaultFloat→float).
    auto inc_prim = extract_primitive(incoming);
    if (inc_prim && !is_default_type(incoming))
    {
      auto it = u->begin();
      while (it != u->end())
      {
        bool is_def_int = (*it) == DefaultInt;
        bool is_def_float = (*it) == DefaultFloat;
        if (
          (is_def_int && inc_prim->in(integer_types)) ||
          (is_def_float && inc_prim->in(float_types)))
          it = u->erase(it, std::next(it));
        else
          ++it;
      }
    }

    bool covered = false;
    for (auto& m : *u)
    {
      if (Subtype(ctx, incoming, Type << clone(m)))
      {
        covered = true;
        break;
      }
    }

    if (!covered)
    {
      auto it = u->begin();
      while (it != u->end())
      {
        if (Subtype(ctx, Type << clone(*it), incoming))
          it = u->erase(it, std::next(it));
        else
          ++it;
      }
      u << clone(i_inner);
    }

    return (u->size() == 1) ? Type << clone(u->front()) : Type << u;
  }

  // Merge a type into env at loc. Returns true if the type changed.
  static bool merge_env(
    TypeEnv& env,
    const Location& loc,
    const Node& type,
    Node top,
    Node call_node = {})
  {
    if (active_infer_profile != nullptr)
      active_infer_profile->merge_env_calls++;

    auto it = env.find(loc);
    if (it == env.end())
    {
      env[loc] = {clone(type), false, call_node};
      return true;
    }
    if (it->second.is_fixed)
      return false;

    if (it->second.type == type)
    {
      if (call_node && !it->second.call_node)
      {
        it->second.call_node = call_node;
        return true;
      }

      return false;
    }

    auto merged = merge_type(it->second.type, type, top);
    if (!merged)
    {
      if (call_node && !it->second.call_node)
      {
        it->second.call_node = call_node;
        return true;
      }

      return false;
    }

    it->second.type = merged;
    if (call_node && !it->second.call_node)
      it->second.call_node = call_node;
    return true;
  }

  using SrcIndex = std::multimap<Location, Node>;

  static bool is_cascade_concrete(const Node& type)
  {
    return type && !type->empty() &&
      !type->front()->in({TypeVar, DefaultInt, DefaultFloat, Union});
  }

  static void enqueue_if_concrete(
    const TypeEnv& env,
    const Location& loc,
    std::deque<Location>& work,
    std::set<Location>& in_queue)
  {
    auto it = env.find(loc);
    if (it == env.end() || !is_cascade_concrete(it->second.type))
      return;
    if (in_queue.insert(loc).second)
      work.push_back(loc);
  }

  static SrcIndex build_src_index(const Node& body)
  {
    SrcIndex src_index;
    for (auto& stmt : *body)
    {
      if (stmt->in({Copy, Move}))
      {
        src_index.emplace((stmt / Rhs)->location(), stmt);
      }
      else if (stmt == Lookup)
      {
        src_index.emplace((stmt / Rhs)->location(), stmt);
      }
      else if (stmt->in({ArrayRef, ArrayRefConst, ArrayRefFromEnd, SplatOp}))
      {
        src_index.emplace(((stmt / Arg) / Rhs)->location(), stmt);
      }
      else if (stmt == Load)
      {
        src_index.emplace((stmt / Rhs)->location(), stmt);
      }
      else if (stmt->in({CallDyn, TryCallDyn}))
      {
        src_index.emplace((stmt / Rhs)->location(), stmt);
        for (auto& arg : *(stmt / Args))
          src_index.emplace((arg / Rhs)->location(), stmt);
      }
    }
    return src_index;
  }

  static std::set<Location> collect_label_defs(const Node& body)
  {
    std::set<Location> defs;
    for (auto& stmt : *body)
    {
      if (!stmt->empty() && stmt->front() == LocalId)
      {
        auto loc = stmt->front()->location();
        if (loc.view().starts_with("local$"))
          defs.insert(loc);
      }
    }
    return defs;
  }

  static std::set<Location> collect_label_uses(
    const Node& body, const Node& term, const std::set<Location>& defs)
  {
    std::set<Location> uses;

    auto collect = [&](const Node& root) {
      if (!root)
        return;
      root->traverse([&](auto& node) {
        if (node == LocalId)
          uses.insert(node->location());
        return true;
      });
    };

    collect(body);
    collect(term);

    for (auto& def : defs)
      uses.erase(def);

    return uses;
  }

  static std::set<Location> collect_label_kills(const Node& body)
  {
    std::set<Location> kills;
    for (auto& stmt : *body)
    {
      if (!stmt->empty() && stmt->front() == LocalId)
        kills.insert(stmt->front()->location());
    }
    return kills;
  }

  static bool prune_bwd_env(
    TypeEnv& env,
    const std::set<Location>& live_in,
    const std::set<Location>& defs)
  {
    bool changed = false;
    for (auto it = env.begin(); it != env.end();)
    {
      if ((live_in.count(it->first) == 0) && (defs.count(it->first) == 0))
      {
        it = env.erase(it);
        changed = true;
      }
      else
      {
        ++it;
      }
    }
    return changed;
  }

  static void run_dependency_cascade(
    TypeEnv& env,
    const Node& body,
    Node top,
    std::map<Location, Node>& lookup_stmts)
  {
    auto* profile = active_infer_profile;
    SrcIndex src_index;
    {
      InferScopedTimer timer(
        (profile != nullptr) ? &profile->cascade_index_time : nullptr);
      src_index = build_src_index(body);
    }
    if (profile != nullptr)
    {
      profile->cascade_env_bindings += env.size();
      profile->cascade_body_statements += body->size();
      profile->cascade_src_entries += src_index.size();
    }

    std::map<Location, Node> const_defs;
    std::deque<Location> work;
    std::set<Location> in_queue;

    {
      InferScopedTimer timer(
        (profile != nullptr) ? &profile->cascade_const_scan_time : nullptr);
      for (auto& stmt : *body)
        if (stmt == Const)
          const_defs[(stmt / LocalId)->location()] = stmt;
    }
    if (profile != nullptr)
      profile->cascade_const_defs += const_defs.size();

    auto refine_local_const =
      [&](const Location& loc, const Node& expected) -> bool {
      return refine_const_local(env, const_defs, loc, expected);
    };

    {
      InferScopedTimer timer(
        (profile != nullptr) ? &profile->cascade_seed_time : nullptr);
      for (auto& [loc, info] : env)
      {
        if (!is_cascade_concrete(info.type))
          continue;

        if (profile != nullptr)
        {
          auto type_nodes = infer_tree_size(info.type);
          profile->cascade_seed_locs++;
          profile->cascade_seed_type_nodes += type_nodes;
          profile->cascade_max_type_nodes =
            std::max(profile->cascade_max_type_nodes, type_nodes);
        }
        enqueue_if_concrete(env, loc, work, in_queue);
      }
    }

    {
      InferScopedTimer timer(
        (profile != nullptr) ? &profile->cascade_loop_time : nullptr);

      while (!work.empty())
      {
        auto loc = work.front();
        work.pop_front();
        in_queue.erase(loc);
        if (profile != nullptr)
          profile->cascade_work_pops++;

        auto [begin, end] = src_index.equal_range(loc);
        for (auto it = begin; it != end; ++it)
        {
          if (profile != nullptr)
            profile->cascade_stmt_visits++;

          auto stmt = it->second;

          if (stmt->in({Copy, Move}))
          {
            auto src_loc = (stmt / Rhs)->location();
            auto src_it = env.find(src_loc);
            if (src_it == env.end())
              continue;

            auto dst_loc = (stmt / LocalId)->location();
            if (
              merge_env(
                env,
                dst_loc,
                src_it->second.type,
                top,
                src_it->second.call_node))
              enqueue_if_concrete(env, dst_loc, work, in_queue);
          }
          else if (stmt == Lookup)
          {
            auto src_loc = (stmt / Rhs)->location();
            auto src_it = env.find(src_loc);
            if (src_it == env.end() || is_default_type(src_it->second.type))
              continue;

            auto hand = (stmt / Lhs)->type();
            auto method_ident = lookup_method_name(stmt);
            auto method_ta = stmt / TypeArgs;
            auto arity = from_chars_sep_v<size_t>(stmt / Int);
            auto ret = resolve_method_return_type(
              top, src_it->second.type, method_ident, hand, arity, method_ta);
            if (!ret)
              continue;

            auto dst_loc = (stmt / LocalId)->location();
            if (merge_env(env, dst_loc, ret, top))
              enqueue_if_concrete(env, dst_loc, work, in_queue);
          }
          else if (stmt->in({ArrayRef, ArrayRefConst}))
          {
            auto dst_loc = (stmt / LocalId)->location();
            auto src_loc = ((stmt / Arg) / Rhs)->location();
            auto src_it = env.find(src_loc);
            if (src_it == env.end())
              continue;

            if (stmt == ArrayRefConst)
            {
              auto index = from_chars_sep_v<size_t>(stmt / Rhs);
              auto inner = src_it->second.type->front();
              if (inner == TupleType && index < inner->size())
              {
                if (
                  merge_env(
                    env,
                    dst_loc,
                    ref_type(Type << clone(inner->at(index))),
                    top))
                  enqueue_if_concrete(env, dst_loc, work, in_queue);
                continue;
              }
            }

            if (
              merge_env(
                env, dst_loc, ref_type(clone(src_it->second.type)), top))
              enqueue_if_concrete(env, dst_loc, work, in_queue);
          }
          else if (stmt == ArrayRefFromEnd)
          {
            auto dst_loc = (stmt / LocalId)->location();
            auto src_loc = ((stmt / Arg) / Rhs)->location();
            auto src_it = env.find(src_loc);
            if (src_it == env.end())
              continue;

            auto offset = from_chars_sep_v<size_t>(stmt / Rhs);
            auto inner = src_it->second.type->front();
            if (inner == TupleType && offset > 0 && offset <= inner->size())
            {
              auto index = inner->size() - offset;
              if (
                merge_env(
                  env, dst_loc, ref_type(Type << clone(inner->at(index))), top))
                enqueue_if_concrete(env, dst_loc, work, in_queue);
            }
            else if (
              merge_env(
                env, dst_loc, ref_type(clone(src_it->second.type)), top))
            {
              enqueue_if_concrete(env, dst_loc, work, in_queue);
            }
          }
          else if (stmt == Load)
          {
            auto dst_loc = (stmt / LocalId)->location();
            auto src_it = env.find((stmt / Rhs)->location());
            if (src_it == env.end())
              continue;

            auto inner = extract_ref_inner(src_it->second.type);
            if (inner && merge_env(env, dst_loc, inner, top))
              enqueue_if_concrete(env, dst_loc, work, in_queue);
          }
          else if (stmt == SplatOp)
          {
            auto dst_loc = (stmt / LocalId)->location();
            auto src_loc = ((stmt / Arg) / Rhs)->location();
            auto src_it = env.find(src_loc);
            if (src_it == env.end())
              continue;

            auto inner = src_it->second.type->front();
            if (inner != TupleType)
              continue;

            auto before = from_chars_sep_v<size_t>(stmt / Lhs);
            auto after = from_chars_sep_v<size_t>(stmt / Rhs);
            if (before + after > inner->size())
              continue;

            auto remaining = inner->size() - before - after;
            Node splat_type;
            if (remaining == 0)
              splat_type = primitive_type(None);
            else if (remaining == 1)
              splat_type = Type << clone(inner->at(before));
            else
            {
              Node tup = TupleType;
              for (size_t i = before; i < before + remaining; i++)
                tup << clone(inner->at(i));
              splat_type = Type << tup;
            }

            if (merge_env(env, dst_loc, splat_type, top))
              enqueue_if_concrete(env, dst_loc, work, in_queue);
          }
          else if (stmt->in({CallDyn, TryCallDyn}))
          {
            auto dst_loc = (stmt / LocalId)->location();
            auto src_loc = (stmt / Rhs)->location();
            auto src_it = env.find(src_loc);
            if (src_it != env.end())
            {
              auto call_node = stmt;
              if (merge_env(env, dst_loc, src_it->second.type, top, call_node))
                enqueue_if_concrete(env, dst_loc, work, in_queue);
            }

            auto lookup_it = lookup_stmts.find(src_loc);
            if (lookup_it == lookup_stmts.end())
              continue;

            auto lookup_node = lookup_it->second;
            auto recv_loc = (lookup_node / Rhs)->location();
            auto recv_it = env.find(recv_loc);
            if (recv_it == env.end() || is_default_type(recv_it->second.type))
              continue;

            auto hand = (lookup_node / Lhs)->type();
            auto method_ident = lookup_method_name(lookup_node);
            auto method_ta = lookup_node / TypeArgs;
            auto arity = from_chars_sep_v<size_t>(lookup_node / Int);
            auto info = resolve_callable_method(
              top, recv_it->second.type, method_ident, hand, arity, method_ta);
            if (!info.func)
              continue;

            auto params = info.func / Params;
            auto args = stmt / Args;
            for (size_t i = 0; i < params->size() && i < args->size(); i++)
            {
              auto expected =
                apply_subst(top, params->at(i) / Type, info.subst);
              if (!expected || expected->front() == TypeVar)
                continue;

              auto arg_loc = (args->at(i) / Rhs)->location();
              refine_local_const(arg_loc, expected);

              if (merge_env(env, arg_loc, expected, top))
                enqueue_if_concrete(env, arg_loc, work, in_queue);
            }
          }
        }
      }
    }
  }

  // ===== Generic type inference =====

  NodeMap<Node>
  build_class_subst(const Node& class_def, const Node& typename_node)
  {
    NodeMap<Node> subst;
    auto tps = class_def / TypeParams;
    auto ta = typename_node->back() / TypeArgs;
    if (tps->size() == ta->size())
      for (size_t i = 0; i < tps->size(); i++)
        subst[tps->at(i)] = ta->at(i);
    return subst;
  }

  void extract_constraints(
    Node top,
    const Node& f_inner,
    const Node& a_inner,
    NodeMap<LocalTypeInfo>& constraints,
    bool is_default)
  {
    // Formal is a TypeParam reference.
    if (f_inner == TypeName)
    {
      auto def = find_def(top, f_inner);
      if (def && def == TypeParam)
      {
        Node actual_type = Type << clone(a_inner);
        auto existing = constraints.find(def);
        if (existing == constraints.end())
          constraints[def] = {actual_type, false, {}};
        else if (is_default_type(existing->second.type) && !is_default)
          constraints[def] = {actual_type, false, {}};
        return;
      }
    }

    // Both TypeNames: structural or shape-aware matching.
    if (f_inner == TypeName && a_inner == TypeName)
    {
      bool structural = (f_inner->size() == a_inner->size());
      if (structural)
      {
        for (size_t i = 0; i < f_inner->size(); i++)
        {
          if (
            (f_inner->at(i) / Ident)->location().view() !=
            (a_inner->at(i) / Ident)->location().view())
          {
            structural = false;
            break;
          }
          if (
            (f_inner->at(i) / TypeArgs)->size() !=
            (a_inner->at(i) / TypeArgs)->size())
          {
            structural = false;
            break;
          }
        }
      }

      if (structural)
      {
        for (size_t i = 0; i < f_inner->size(); i++)
        {
          auto f_ta = f_inner->at(i) / TypeArgs;
          auto a_ta = a_inner->at(i) / TypeArgs;
          for (size_t j = 0; j < f_ta->size(); j++)
            extract_constraints(
              top,
              f_ta->at(j)->front(),
              a_ta->at(j)->front(),
              constraints,
              is_default);
        }
        return;
      }

      // Shape-aware matching.
      auto f_def = find_def(top, f_inner);
      auto a_def = find_def(top, a_inner);
      if (
        f_def && a_def && f_def == ClassDef && (f_def / Shape) == Shape &&
        a_def == ClassDef)
      {
        auto shape_to_formal = build_class_subst(f_def, f_inner);
        if (!shape_to_formal.empty())
        {
          auto actual_subst = build_class_subst(a_def, a_inner);
          for (auto& sf : *(f_def / ClassBody))
          {
            if (sf != Function)
              continue;
            auto mname = (sf / Ident)->location().view();
            auto hand = (sf / Lhs)->type();
            auto arity = (sf / Params)->size();
            for (auto& af : *(a_def / ClassBody))
            {
              if (af != Function)
                continue;
              if ((af / Ident)->location().view() != mname)
                continue;
              if ((af / Lhs)->type() != hand)
                continue;
              if ((af / Params)->size() != arity)
                continue;
              auto formal_params = sf / Params;
              auto actual_params = af / Params;
              for (size_t j = 0; j < formal_params->size(); j++)
              {
                auto formal_param = apply_subst(
                  top, formal_params->at(j) / Type, shape_to_formal);
                auto actual_param =
                  apply_subst(top, actual_params->at(j) / Type, actual_subst);
                if (
                  !formal_param || !actual_param ||
                  actual_param->front() == TypeVar)
                {
                  continue;
                }
                extract_constraints(
                  top,
                  formal_param->front(),
                  actual_param->front(),
                  constraints,
                  is_default);
              }
              auto formal_ret = apply_subst(top, sf / Type, shape_to_formal);
              auto actual_ret = apply_subst(top, af / Type, actual_subst);
              extract_constraints(
                top,
                formal_ret->front(),
                actual_ret->front(),
                constraints,
                is_default);
              break;
            }
          }
        }
      }
      return;
    }

    // Union/Isect/TupleType: element-wise.
    if (
      f_inner->type() == a_inner->type() &&
      f_inner->size() == a_inner->size() &&
      f_inner->in({Union, Isect, TupleType}))
    {
      for (size_t i = 0; i < f_inner->size(); i++)
        extract_constraints(
          top, f_inner->at(i), a_inner->at(i), constraints, is_default);
    }
  }

  Node apply_subst(Node top, const Node& type_node, const NodeMap<Node>& subst)
  {
    if (active_infer_profile != nullptr)
      active_infer_profile->apply_subst_calls++;
    InferScopedTimer timer(
      (active_infer_profile != nullptr) ?
        &active_infer_profile->apply_subst_time :
        nullptr);

    if (type_node != Type || subst.empty())
      return clone(type_node);

    auto inner = type_node->front();

    if (inner == TypeName)
    {
      auto def = find_def(top, inner);
      if (def && def == TypeParam)
      {
        auto it = subst.find(def);
        if (it != subst.end())
          return clone(it->second);
      }

      Node new_tn = TypeName;
      for (auto& elem : *inner)
      {
        Node new_ta = TypeArgs;
        for (auto& ta_child : *(elem / TypeArgs))
          new_ta << apply_subst(top, ta_child, subst);
        new_tn << (NameElement << clone(elem / Ident) << new_ta);
      }
      return Type << new_tn;
    }

    if (inner->in({Union, Isect, TupleType}))
    {
      Node new_inner = inner->type();
      for (auto& child : *inner)
        new_inner << apply_subst(top, Type << clone(child), subst)->front();
      return Type << new_inner;
    }

    return clone(type_node);
  }

  // ===== Method resolution =====

  MethodInfo resolve_method(
    Node top,
    const Node& receiver_type,
    const Node& method_ident,
    Token hand,
    size_t arity,
    const Node& method_typeargs)
  {
    if (active_infer_profile != nullptr)
      active_infer_profile->resolve_method_calls++;
    InferScopedTimer timer(
      (active_infer_profile != nullptr) ?
        &active_infer_profile->resolve_method_time :
        nullptr);

    auto owner = resolve_method_owner(top, receiver_type);
    if (owner.class_def)
    {
      auto subst = build_class_subst(owner.class_def, owner.subst_source);
      auto func = resolve_class_method(
        owner, method_ident->location().view(), hand, arity);
      if (func)
      {
        auto func_tps = func / TypeParams;
        if (
          !method_typeargs->empty() &&
          method_typeargs->size() == func_tps->size())
        {
          for (size_t i = 0; i < func_tps->size(); i++)
            subst[func_tps->at(i)] = method_typeargs->at(i);
        }
        return {func, std::move(subst)};
      }
    }

    // Union receiver: resolve on every member, all must have the method.
    if (receiver_type == Type && receiver_type->front() == Union)
    {
      MethodInfo first_info;

      for (auto& member : *(receiver_type->front()))
      {
        Node member_type = Type << clone(member);
        auto info = resolve_method(
          top, member_type, method_ident, hand, arity, method_typeargs);

        if (!info.func)
          return {};

        if (!first_info.func)
          first_info = std::move(info);
      }

      return first_info;
    }

    // Auto-deref ref[T] / cown[T].
    auto ref_inner = extract_ref_inner(receiver_type);
    if (!ref_inner)
      ref_inner = extract_cown_inner(receiver_type);
    if (ref_inner)
      return resolve_method(
        top, ref_inner, method_ident, hand, arity, method_typeargs);
    return {};
  }

  Node resolve_method_return_type(
    Node top,
    const Node& receiver_type,
    const Node& method_ident,
    Token hand,
    size_t arity,
    const Node& method_typeargs)
  {
    auto info = resolve_method(
      top, receiver_type, method_ident, hand, arity, method_typeargs);
    if (!info.func)
      return {};
    auto ret = apply_subst(top, info.func / Type, info.subst);

    // Rhs wrapper auto-gen: return type = lhs unwrapped ref.
    if (ret && ret->front() == TypeVar && hand == Rhs)
    {
      auto lhs_info = resolve_method(
        top, receiver_type, method_ident, Lhs, arity, method_typeargs);
      if (lhs_info.func)
      {
        auto lhs_ret = apply_subst(top, lhs_info.func / Type, lhs_info.subst);
        auto inner = extract_ref_inner(lhs_ret);
        if (inner)
          return inner;
      }
    }
    return ret;
  }

  MethodInfo resolve_callable_method(
    Node top,
    const Node& receiver_type,
    const Node& method_ident,
    Token hand,
    size_t arity,
    const Node& method_typeargs)
  {
    auto info = resolve_method(
      top, receiver_type, method_ident, hand, arity, method_typeargs);
    if (!info.func && hand == Rhs)
      info = resolve_method(
        top, receiver_type, method_ident, Lhs, arity, method_typeargs);
    return info;
  }

  bool propagate_shape_to_lambda(
    Node top, const Node& shape_type, const Node& actual_type)
  {
    if (shape_type != Type || actual_type != Type)
      return false;
    auto shape_inner = shape_type->front();
    auto actual_inner = actual_type->front();
    if (shape_inner != TypeName || actual_inner != TypeName)
      return false;
    auto shape_def = find_def(top, shape_inner);
    auto actual_def = find_def(top, actual_inner);
    if (!shape_def || !actual_def)
      return false;

    // Follow TypeAlias chains to the underlying ClassDef.
    while (shape_def == TypeAlias)
    {
      auto alias_type = shape_def / Type;
      if (alias_type->front() != TypeName)
        break;
      shape_def = find_def(top, alias_type->front());
      if (!shape_def)
        return false;
    }

    if (shape_def != ClassDef || actual_def != ClassDef)
      return false;
    if ((shape_def / Shape) != Shape)
      return false;

    bool changed = false;
    auto shape_subst = build_class_subst(shape_def, shape_inner);
    for (auto& sf : *(shape_def / ClassBody))
    {
      if (sf != Function)
        continue;
      auto mname = (sf / Ident)->location().view();
      auto hand = (sf / Lhs)->type();

      for (auto& af : *(actual_def / ClassBody))
      {
        if (af != Function)
          continue;
        if ((af / Ident)->location().view() != mname)
          continue;
        if ((af / Lhs)->type() != hand)
          continue;
        if ((af / Params)->size() != (sf / Params)->size())
          continue;

        auto shape_params = sf / Params;
        auto actual_params = af / Params;
        for (size_t j = 0; j < shape_params->size(); j++)
        {
          auto ap = actual_params->at(j);
          auto apt = ap / Type;
          if (apt->front() != TypeVar)
            continue;
          auto spt = apply_subst(top, shape_params->at(j) / Type, shape_subst);
          if (!spt || spt->front() == TypeVar || spt->front() == TypeSelf)
            continue;
          changed |= replace_if_changed(ap, apt, clone(spt));
        }

        auto actual_ret = af / Type;
        auto shape_ret = apply_subst(top, sf / Type, shape_subst);
        if (shape_ret && shape_ret->front() != TypeVar)
        {
          if (
            actual_ret->front() == TypeVar ||
            (is_lambda_function(af) && lambda_return_was_omitted(af)))
          {
            // For lambdas without an explicit return annotation, callable
            // context supplies the return obligation. Any concrete return
            // inferred in isolation is provisional until checked against that
            // context.
            changed |= replace_if_changed(af, actual_ret, clone(shape_ret));
          }
        }
        break;
      }
    }

    return changed;
  }

  // ===== Cross-function propagation =====

  struct ScopeInfo
  {
    Node name_elem;
    Node def;
  };

  static bool is_lambda_function(const Node& func);
  static Node
  retarget_numeric_type(const Node& type, const Node& expected_prim);
  static Node extract_backward_primitive(const Node& type);
  static bool recover_local_type_from_def(
    const Location& loc,
    TypeEnv& env,
    const std::map<Location, Node>& def_stmts,
    Node top);
  static bool
  refine_function_return_consts(const Node& func, const Node& expected_prim);

  Node navigate_call(Node call, Node top, std::vector<ScopeInfo>& scopes)
  {
    if (active_infer_profile != nullptr)
      active_infer_profile->navigate_call_calls++;
    InferScopedTimer timer(
      (active_infer_profile != nullptr) ?
        &active_infer_profile->navigate_call_time :
        nullptr);

    auto funcname = call / FuncName;
    auto args = call / Args;
    auto func_def = find_func_def(top, funcname, args->size(), call / Lhs);
    if (!func_def)
      return {};

    Node def = top;
    for (auto it = funcname->begin(); it != funcname->end(); ++it)
    {
      bool is_last = (it + 1 == funcname->end());
      if (is_last)
      {
        scopes.push_back({*it, func_def});
      }
      else
      {
        auto defs = def->look(((*it) / Ident)->location());
        def = defs.front();
        scopes.push_back({*it, def});
      }
    }
    return func_def;
  }

  void backward_refine_call(
    Node prior_call, const Node& expected_type, TypeEnv& env, Node top)
  {
    std::vector<ScopeInfo> scopes;
    auto func_def = navigate_call(prior_call, top, scopes);
    if (!func_def)
      return;

    auto args = prior_call / Args;
    auto params = func_def / Params;
    auto ret_type = func_def / Type;

    // Extract TypeParam constraints from return type vs expected type.
    NodeMap<LocalTypeInfo> constraints;
    extract_constraints(
      top, ret_type->front(), expected_type->front(), constraints, false);

    // Reverse shape matching: concrete return → shape expected.
    if (
      constraints.empty() && ret_type->front() == TypeName &&
      expected_type->front() == TypeName)
    {
      auto ret_def = find_def(top, ret_type->front());
      auto exp_def = find_def(top, expected_type->front());
      if (
        ret_def && exp_def && ret_def == ClassDef &&
        (ret_def / Shape) != Shape && exp_def == ClassDef &&
        (exp_def / Shape) == Shape)
      {
        auto shape_to_concrete =
          build_class_subst(exp_def, expected_type->front());
        if (!shape_to_concrete.empty())
        {
          for (auto& sf : *(exp_def / ClassBody))
          {
            if (sf != Function)
              continue;
            auto mname = (sf / Ident)->location().view();
            auto hand = (sf / Lhs)->type();
            auto arity = (sf / Params)->size();
            for (auto& cf : *(ret_def / ClassBody))
            {
              if (cf != Function)
                continue;
              if ((cf / Ident)->location().view() != mname)
                continue;
              if ((cf / Lhs)->type() != hand)
                continue;
              if ((cf / Params)->size() != arity)
                continue;
              auto concrete_ret =
                apply_subst(top, sf / Type, shape_to_concrete);
              extract_constraints(
                top,
                (cf / Type)->front(),
                concrete_ret->front(),
                constraints,
                false);
              break;
            }
          }
        }
      }
    }

    if (constraints.empty())
      return;

    // Update TypeArgs in the AST.
    for (auto& scope : scopes)
    {
      auto ta = scope.name_elem / TypeArgs;
      auto tps = scope.def / TypeParams;
      if (tps->empty())
        continue;

      bool all_constrained = true;
      Node new_ta = TypeArgs;
      for (auto& tp : *tps)
      {
        auto find = constraints.find(tp);
        if (find == constraints.end())
        {
          all_constrained = false;
          break;
        }
        new_ta << clone(find->second.type);
      }
      if (all_constrained)
        replace_if_changed(scope.name_elem, ta, new_ta);
    }

    // Build substitution from updated TypeArgs.
    NodeMap<Node> subst;
    for (auto& scope : scopes)
    {
      auto ta = scope.name_elem / TypeArgs;
      auto tps = scope.def / TypeParams;
      if (!ta->empty() && ta->size() == tps->size())
        for (size_t i = 0; i < tps->size(); i++)
          subst[tps->at(i)] = ta->at(i);
    }

    std::map<Location, Node> const_defs;
    if (auto label = prior_call->parent(Label))
    {
      for (auto& stmt : *(label / Body))
        if (stmt == Const)
          const_defs[(stmt / LocalId)->location()] = stmt;
    }

    // Refine default literal args using the newly constrained TypeArgs.
    for (size_t i = 0; i < params->size() && i < args->size(); i++)
    {
      auto formal = params->at(i) / Type;
      Node expected_prim;
      auto tp_def = direct_typeparam(top, formal);
      if (tp_def)
      {
        auto find = subst.find(tp_def);
        if (find != subst.end())
          expected_prim = extract_primitive(find->second);
      }
      else
      {
        expected_prim = extract_primitive(formal);
      }

      if (!expected_prim)
        continue;

      auto arg_loc = (args->at(i) / Rhs)->location();
      refine_const_local(
        env, const_defs, arg_loc, primitive_or_ffi_type(expected_prim->type()));
    }

    // Backward: merge expected param types into arg env entries.
    for (size_t i = 0; i < params->size() && i < args->size(); i++)
    {
      auto formal = params->at(i) / Type;
      auto expected = apply_subst(top, formal, subst);
      if (expected && !is_uninformative_backward_type(expected))
      {
        auto arg_loc = (args->at(i) / Rhs)->location();
        merge_env(env, arg_loc, expected, top);
      }
    }

    // Update call result.
    auto result = apply_subst(top, ret_type, subst);
    if (result)
      merge_env(env, (prior_call / LocalId)->location(), result, top);
  }

  void backward_refine_calldyn(
    Node calldyn,
    const Node& expected_prim,
    TypeEnv& env,
    Node top,
    std::map<Location, Node>& lookup_stmts,
    const std::map<Location, Node>* def_stmts = nullptr)
  {
    assert(calldyn->in({CallDyn, TryCallDyn}));
    assert(expected_prim);
    auto dst = calldyn / LocalId;
    auto src = calldyn / Rhs;
    auto args = calldyn / Args;
    auto lookup_it = lookup_stmts.find(src->location());
    if (lookup_it == lookup_stmts.end())
      return;

    auto lookup_node = lookup_it->second;
    std::map<Location, Node> const_defs;
    if (def_stmts != nullptr)
    {
      for (auto& [loc, stmt] : *def_stmts)
      {
        if (stmt == Const)
          const_defs[loc] = stmt;
      }
    }
    auto refine_numeric = [&](const Location& loc) -> bool {
      auto it = env.find(loc);
      if (it == env.end())
        return false;

      bool compatible = (it->second.type->front() == DefaultInt &&
                         expected_prim->in(integer_types)) ||
        (it->second.type->front() == DefaultFloat &&
         expected_prim->in(float_types));
      if (!compatible)
        return false;

      it->second.type = primitive_or_ffi_type(expected_prim->type());

      if (def_stmts != nullptr)
      {
        auto def_it = def_stmts->find(loc);
        if (def_it != def_stmts->end() && def_it->second == Const)
        {
          auto const_stmt = def_it->second;
          if (const_stmt->size() == 3)
          {
            auto old_type = const_stmt->at(1);
            if (old_type->type() != expected_prim->type())
            {
              const_stmt->replace(old_type, expected_prim->type());
              note_infer_transfer_change();
            }
          }
          else
          {
            auto dst = const_stmt->front();
            auto lit = const_stmt->back();
            const_stmt->erase(const_stmt->begin(), const_stmt->end());
            const_stmt << dst << expected_prim->type() << lit;
            note_infer_transfer_change();
          }
        }
      }

      return true;
    };
    auto refine_from_expected =
      [&](const Location& loc, const Node& expected) -> bool {
      bool changed = false;
      if (
        !const_defs.empty() &&
        refine_const_local(env, const_defs, loc, expected))
        changed = true;
      if (merge_env(env, loc, expected, top))
        changed = true;
      return changed;
    };

    bool refined = false;

    auto lookup_src = lookup_node / Rhs;
    refined = refine_numeric(lookup_src->location()) || refined;

    auto recv_it = env.find(lookup_src->location());
    if (recv_it == env.end() && def_stmts != nullptr)
    {
      recover_local_type_from_def(lookup_src->location(), env, *def_stmts, top);
      recv_it = env.find(lookup_src->location());
    }
    if (recv_it != env.end())
    {
      auto info = resolve_callable_method(
        top,
        recv_it->second.type,
        lookup_method_name(lookup_node),
        (lookup_node / Lhs)->type(),
        from_chars_sep_v<size_t>(lookup_node / Int),
        lookup_node / TypeArgs);
      if (info.func && is_lambda_function(info.func))
      {
        auto new_ret = retarget_numeric_type(info.func / Type, expected_prim);
        if (new_ret)
        {
          replace_if_changed(info.func, info.func / Type, new_ret);
          refined = true;
        }
        if (refine_function_return_consts(info.func, expected_prim))
          refined = true;
      }
      if (info.func)
      {
        auto params = info.func / Params;
        for (size_t i = 0; i < params->size() && i < args->size(); i++)
        {
          auto expected = apply_subst(top, params->at(i) / Type, info.subst);
          if (!expected || is_uninformative_backward_type(expected))
            continue;

          auto arg_loc = (args->at(i) / Rhs)->location();
          if (refine_from_expected(arg_loc, expected))
            refined = true;
        }
      }
    }

    if (!refined)
      return;

    if (recv_it != env.end())
    {
      auto hand = (lookup_node / Lhs)->type();
      auto method_ident = lookup_method_name(lookup_node);
      auto method_ta = lookup_node / TypeArgs;
      auto arity = from_chars_sep_v<size_t>(lookup_node / Int);
      auto ret = resolve_method_return_type(
        top, recv_it->second.type, method_ident, hand, arity, method_ta);
      if (ret)
      {
        env[(lookup_node / LocalId)->location()] = {ret, false, {}};
      }
    }

    auto src_it = env.find(src->location());
    if (src_it != env.end())
    {
      env[dst->location()] = {clone(src_it->second.type), false, {}};
      auto lookup_dst_it = env.find((lookup_node / LocalId)->location());
      if (lookup_dst_it != env.end())
      {
        env[src->location()] = {clone(lookup_dst_it->second.type), false, {}};
        env[dst->location()] = {clone(lookup_dst_it->second.type), false, {}};
      }
    }
  }

  // ===== Typetest trace =====

  struct TypetestTrace
  {
    Node src;
    Node type;
    bool negated;
  };

  static std::optional<TypetestTrace>
  trace_typetest(const Node& cond_local, const Node& body)
  {
    auto target_loc = cond_local->location();
    bool negated = false;
    for (auto it = body->rbegin(); it != body->rend(); ++it)
    {
      auto& stmt = *it;
      if (stmt == Not)
      {
        auto dst = stmt / LocalId;
        if (dst->location() == target_loc)
        {
          target_loc = (stmt / Rhs)->location();
          negated = !negated;
        }
      }
      else if (stmt == Typetest)
      {
        auto dst = stmt / LocalId;
        if (dst->location() == target_loc)
          return TypetestTrace{stmt / Rhs, stmt / Type, negated};
      }
    }
    return std::nullopt;
  }

  // ===== Propagate call_node =====

  static void propagate_call_node(
    TypeEnv& env,
    const Location& loc,
    Node top,
    std::map<Location, Node>& lookup_stmts,
    const std::map<Location, Node>* def_stmts = nullptr)
  {
    if (active_infer_profile != nullptr)
      active_infer_profile->propagate_call_node_calls++;
    InferScopedTimer timer(
      (active_infer_profile != nullptr) ?
        &active_infer_profile->propagate_call_node_time :
        nullptr);

    auto it = env.find(loc);
    if (
      it == env.end() || is_default_type(it->second.type) ||
      !it->second.call_node)
      return;

    auto call = it->second.call_node;
    if (call == Call)
      backward_refine_call(call, it->second.type, env, top);
    else if (call->in({CallDyn, TryCallDyn}))
    {
      auto prim = extract_primitive(it->second.type);
      if (prim)
        backward_refine_calldyn(call, prim, env, top, lookup_stmts, def_stmts);
    }
  }

  static void propagate_call_constraint(
    TypeEnv& env,
    const Location& loc,
    const Node& expected,
    Node top,
    std::map<Location, Node>& lookup_stmts,
    const std::map<Location, Node>* def_stmts = nullptr)
  {
    if (
      !expected || is_default_type(expected) || contains_default_type(expected))
      return;

    std::set<std::string> seen;
    auto recurse = [&](auto&& self, const Location& cur_loc) -> void {
      auto [_, inserted] = seen.insert(std::string(cur_loc.view()));
      if (!inserted)
        return;

      auto it = env.find(cur_loc);
      if (it != env.end() && it->second.call_node)
      {
        auto call = it->second.call_node;
        if (call == Call)
        {
          backward_refine_call(call, expected, env, top);
          return;
        }
        if (call->in({CallDyn, TryCallDyn}))
        {
          auto prim = extract_backward_primitive(expected);
          if (prim)
            backward_refine_calldyn(
              call, prim, env, top, lookup_stmts, def_stmts);
          return;
        }
      }

      if (def_stmts == nullptr)
        return;

      auto def_it = def_stmts->find(cur_loc);
      if (def_it == def_stmts->end())
        return;

      auto def = def_it->second;
      if (def == Call)
      {
        backward_refine_call(def, expected, env, top);
      }
      else if (def->in({CallDyn, TryCallDyn}))
      {
        auto prim = extract_backward_primitive(expected);
        if (prim)
          backward_refine_calldyn(def, prim, env, top, lookup_stmts, def_stmts);
      }
      else if (def->in({Copy, Move}))
      {
        self(self, (def / Rhs)->location());
      }
    };

    recurse(recurse, loc);
  }

  // ===== TypeArg inference for Call sites =====

  // Infers TypeArgs for a Call statement. Returns true if all constraints
  // came from default-typed args (enables call_node tracking).
  static bool infer_typeargs(
    Node call,
    Node func_def,
    std::vector<ScopeInfo>& scopes,
    TypeEnv& env,
    Node top)
  {
    if (active_infer_profile != nullptr)
      active_infer_profile->infer_typeargs_calls++;
    InferScopedTimer timer(
      (active_infer_profile != nullptr) ?
        &active_infer_profile->infer_typeargs_time :
        nullptr);

    auto args = call / Args;
    auto params = func_def / Params;

    bool needs_inference = false;
    for (auto& scope : scopes)
    {
      auto ta = scope.name_elem / TypeArgs;
      auto tps = scope.def / TypeParams;
      if (!tps->empty())
      {
        if (ta->empty())
        {
          needs_inference = true;
          break;
        }

        for (auto& t : *ta)
        {
          if (t == Type && t->front() == TypeVar)
          {
            needs_inference = true;
            break;
          }
        }

        if (needs_inference)
          break;
      }
    }

    if (!needs_inference)
      return false;

    // Collect constraints from arg types.
    NodeMap<LocalTypeInfo> constraints;
    for (size_t i = 0; i < params->size() && i < args->size(); i++)
    {
      auto arg_loc = (args->at(i) / Rhs)->location();
      auto arg_it = env.find(arg_loc);
      if (arg_it == env.end())
        continue;
      extract_constraints(
        top,
        (params->at(i) / Type)->front(),
        arg_it->second.type->front(),
        constraints,
        is_default_type(arg_it->second.type));
    }

    bool all_default = !constraints.empty();
    for (auto& [tp, info] : constraints)
      if (!is_default_type(info.type))
        all_default = false;

    // Fill TypeArgs.
    for (auto& scope : scopes)
    {
      auto ta = scope.name_elem / TypeArgs;
      auto tps = scope.def / TypeParams;
      if (tps->empty())
        continue;

      if (!ta->empty())
      {
        bool needs_reinfer = false;
        for (auto& t : *ta)
          if (direct_typeparam(top, t) || (t == Type && t->front() == TypeVar))
          {
            needs_reinfer = true;
            break;
          }
        if (!needs_reinfer)
          continue;
      }

      bool all_constrained = true;
      Node new_ta = TypeArgs;
      for (auto& tp : *tps)
      {
        auto find = constraints.find(tp);
        if (find == constraints.end())
        {
          all_constrained = false;
          break;
        }
        new_ta << clone(find->second.type);
      }
      if (all_constrained)
        replace_if_changed(scope.name_elem, ta, new_ta);
    }

    return all_default;
  }

  // Push concrete arg types into TypeVar formal params (AST mutation).
  static void push_arg_types_to_params(
    Node func_def, const Node& args, TypeEnv& env, Node /*top*/)
  {
    if (active_infer_profile != nullptr)
      active_infer_profile->push_arg_types_to_params_calls++;
    InferScopedTimer timer(
      (active_infer_profile != nullptr) ?
        &active_infer_profile->push_arg_types_to_params_time :
        nullptr);

    auto params = func_def / Params;
    auto parent_cls = func_def->parent(ClassDef);

    for (size_t i = 0; i < params->size() && i < args->size(); i++)
    {
      auto param = params->at(i);
      auto formal_type = param / Type;
      if (!contains_typevar(formal_type))
        continue;

      Node resolved;
      bool allow_default_arg = false;

      // FieldDef takes priority.
      if (parent_cls)
      {
        auto pname = (param / Ident)->location().view();
        for (auto& child : *(parent_cls / ClassBody))
        {
          if (child != FieldDef)
            continue;
          if ((child / Ident)->location().view() != pname)
            continue;
          if (!contains_typevar(child / Type))
            resolved = child / Type;
          else if (is_lambda_function(func_def))
            allow_default_arg = true;
          break;
        }
      }

      if (!resolved)
      {
        auto arg_loc = (args->at(i) / Rhs)->location();
        auto arg_it = env.find(arg_loc);
        if (arg_it == env.end() || contains_typevar(arg_it->second.type))
          continue;
        if (contains_default_type(arg_it->second.type) && !allow_default_arg)
          continue;
        resolved = arg_it->second.type;
      }

      replace_if_changed(param, formal_type, clone(resolved));

      // Keep FieldDef in sync.
      if (parent_cls)
      {
        auto pname = (param / Ident)->location().view();
        for (auto& child : *(parent_cls / ClassBody))
        {
          if (child != FieldDef)
            continue;
          if ((child / Ident)->location().view() != pname)
            continue;
          if (contains_typevar(child / Type))
            replace_if_changed(child, child / Type, clone(resolved));
          break;
        }
      }
    }
  }

  static bool is_lambda_function(const Node& func)
  {
    auto class_def = func->parent(ClassBody);
    if (!class_def)
      return false;
    class_def = class_def->parent(ClassDef);
    if (!class_def)
      return false;

    auto ident = class_def / Ident;
    auto name = ident->location().view();
    return name.rfind("lambda$", 0) == 0;
  }

  static Node retarget_numeric_type(const Node& type, const Node& expected_prim)
  {
    if (!type || type != Type || !expected_prim)
      return {};

    auto inner = type->front();
    if (inner->in({DefaultInt, DefaultFloat}))
      return primitive_or_ffi_type(expected_prim->type());

    if (auto prim = extract_primitive(type))
    {
      bool compatible =
        (prim->in(integer_types) && expected_prim->in(integer_types)) ||
        (prim->in(float_types) && expected_prim->in(float_types));
      if (compatible && prim->type() != expected_prim->type())
        return primitive_or_ffi_type(expected_prim->type());
      return {};
    }

    if (inner != Union)
      return {};

    Node new_union = Union;
    bool changed = false;
    for (auto& member : *inner)
    {
      Node member_type = Type << clone(member);
      auto refined_member = retarget_numeric_type(member_type, expected_prim);
      if (refined_member)
      {
        new_union << clone(refined_member->front());
        changed = true;
      }
      else
      {
        new_union << clone(member);
      }
    }

    if (!changed)
      return {};

    return Type << new_union;
  }

  static Node extract_backward_primitive(const Node& type)
  {
    auto prim = extract_primitive(type);
    if (prim)
      return prim;

    if (!type || type != Type || type->front() != Union)
      return {};

    Node found;
    for (auto& member : *type->front())
    {
      Node member_type = Type << clone(member);
      auto member_prim = extract_primitive(member_type);
      if (!member_prim)
        continue;

      if (!found)
      {
        found = member_prim;
        continue;
      }

      if (found->type() != member_prim->type())
        return {};
    }

    return found;
  }

  static bool recover_local_type_from_def(
    const Location& loc,
    TypeEnv& env,
    const std::map<Location, Node>& def_stmts,
    Node top)
  {
    if (env.find(loc) != env.end())
      return true;

    auto it = def_stmts.find(loc);
    if (it == def_stmts.end())
      return false;

    auto stmt = it->second;
    Node recovered;

    if (stmt == New)
    {
      recovered = clone(stmt / Type);
    }
    else if (stmt == Convert)
    {
      recovered = clone(stmt / Type);
    }
    else if (stmt == FFIStruct)
    {
      recovered = ffi_struct_result_type();
    }
    else if (stmt == FFILoad)
    {
      recovered = clone(stmt / Type);
    }
    else if (stmt == TypedMove)
    {
      recovered = clone(stmt / Type);
    }
    else if (stmt->in({Copy, Move}))
    {
      auto src_it = env.find((stmt / Rhs)->location());
      if (src_it != env.end())
        recovered = clone(src_it->second.type);
    }
    else if (stmt == Call)
    {
      auto func_def =
        find_func_def(top, stmt / FuncName, (stmt / Args)->size(), stmt / Lhs);
      if (func_def)
      {
        auto ret = func_def / Type;
        if (!contains_typevar(ret) && !is_default_type(ret))
          recovered = clone(ret);
      }
    }

    if (!recovered)
      return false;

    env[loc] = {clone(recovered), false, {}};
    return true;
  }

  static bool
  refine_function_return_consts(const Node& func, const Node& expected_prim)
  {
    if (!func || !expected_prim)
      return false;

    bool changed = false;
    for (auto& label : *(func / Labels))
    {
      std::map<Location, Node> const_defs;
      TypeEnv env;
      for (auto& stmt : *(label / Body))
      {
        if (stmt == Const)
        {
          auto loc = (stmt / LocalId)->location();
          const_defs[loc] = stmt;
          auto type_tok = (stmt->size() == 3) ?
            stmt->at(1) :
            default_literal_type(stmt->back());
          auto type = type_tok->in({DefaultInt, DefaultFloat}) ?
            (Type << type_tok->type()) :
            primitive_or_ffi_type(type_tok->type());
          env[loc] = {type, false, {}};
        }
      }

      auto term = label / Return;
      if (term != Return)
        continue;

      auto ret_loc = (term / LocalId)->location();
      bool refined = refine_const_local(
        env, const_defs, ret_loc, primitive_or_ffi_type(expected_prim->type()));
      if (refined)
        changed = true;
    }

    return changed;
  }

  // ===== Transfer functions: process_label_body =====

  // Tuple tracking for NewArrayConst → ArrayRefConst → Store patterns.
  struct TupleTracking
  {
    size_t size;
    bool is_array_lit;
    std::vector<Node> element_types;
    std::vector<Location> element_value_locs;
  };

  static Node infer_tracked_tuple_type(const TupleTracking& tt)
  {
    if (tt.is_array_lit)
    {
      Node common;
      bool uniform = true;
      for (size_t i = 0; i < tt.size && uniform; i++)
      {
        if (tt.element_value_locs[i].view().empty())
        {
          uniform = false;
          break;
        }

        auto& et = tt.element_types[i];
        if (!et || !extract_primitive(et))
        {
          uniform = false;
          break;
        }

        if (!common)
          common = clone(et);
        else if (et->front()->type() != common->front()->type())
          uniform = false;
      }

      if (uniform && common && tt.size > 0)
        return clone(common);
      return {};
    }

    for (size_t i = 0; i < tt.size; i++)
      if (!tt.element_types[i])
        return {};

    if (tt.size == 0)
      return {};
    if (tt.size == 1)
      return Type << clone(tt.element_types[0]->front());

    Node tup = TupleType;
    for (auto& et : tt.element_types)
      tup << clone(et->front());
    return Type << tup;
  }

  struct LabelChanges
  {
    bool forward = false;
    bool backward = false;
  };

  enum class InferStmtFamily
  {
    ConstLike,
    CopyLike,
    RefOps,
    TupleOps,
    NewOps,
    CallOps,
    FFIWhenOps,
    TypetestOps,
  };

  struct InferStmtScope
  {
    InferStmtScope(InferStmtFamily family)
    {
      if (active_infer_profile == nullptr)
        return;

      switch (family)
      {
        case InferStmtFamily::ConstLike:
          active_infer_profile->stmt_const_like++;
          timer.emplace(&active_infer_profile->stmt_const_like_time);
          break;
        case InferStmtFamily::CopyLike:
          active_infer_profile->stmt_copy_like++;
          timer.emplace(&active_infer_profile->stmt_copy_like_time);
          break;
        case InferStmtFamily::RefOps:
          active_infer_profile->stmt_ref_ops++;
          timer.emplace(&active_infer_profile->stmt_ref_ops_time);
          break;
        case InferStmtFamily::TupleOps:
          active_infer_profile->stmt_tuple_ops++;
          timer.emplace(&active_infer_profile->stmt_tuple_ops_time);
          break;
        case InferStmtFamily::NewOps:
          active_infer_profile->stmt_new_ops++;
          timer.emplace(&active_infer_profile->stmt_new_ops_time);
          break;
        case InferStmtFamily::CallOps:
          active_infer_profile->stmt_call_ops++;
          timer.emplace(&active_infer_profile->stmt_call_ops_time);
          break;
        case InferStmtFamily::FFIWhenOps:
          active_infer_profile->stmt_ffi_when_ops++;
          timer.emplace(&active_infer_profile->stmt_ffi_when_ops_time);
          break;
        case InferStmtFamily::TypetestOps:
          active_infer_profile->stmt_typetest_ops++;
          timer.emplace(&active_infer_profile->stmt_typetest_ops_time);
          break;
      }
    }

    std::optional<InferScopedTimer> timer;
  };

  static LabelChanges process_label_body(
    const Node& body,
    TypeEnv& env,
    TypeEnv& bwd,
    Node top,
    std::map<Location, Node>& lookup_stmts,
    const std::map<Location, Node>& all_def_stmts,
    std::set<std::pair<Location, Location>>& typevar_aliases,
    std::map<Location, std::pair<Location, size_t>>& ref_to_tuple,
    std::map<Location, TupleTracking>& tuple_locals)
  {
    LabelChanges changes;
    std::map<Location, Node> const_defs;
    std::map<Location, Node> def_stmts;

    for (auto& stmt : *body)
    {
      if (!stmt->empty() && stmt->front() == LocalId)
        def_stmts[stmt->front()->location()] = stmt;
      if (stmt == Const)
        const_defs[(stmt / LocalId)->location()] = stmt;
    }

    auto merge =
      [&](const Location& loc, const Node& type, Node call_node = {}) -> bool {
      bool c = merge_env(env, loc, type, top, call_node);
      if (c)
        changes.forward = true;
      return c;
    };

    auto refine_local_const =
      [&](const Location& loc, const Node& expected) -> bool {
      bool changed = refine_const_local(env, const_defs, loc, expected);
      if (changed)
        changes.forward = true;
      return changed;
    };

    auto merge_bwd =
      [&](
        const Location& loc, const Node& type, bool is_fixed = false) -> bool {
      bool changed = false;

      if (merge_env(env, loc, type, top))
      {
        changes.forward = true;
        changed = true;
      }

      // Record backward constraints that carry a concrete primitive signal.
      bool record_backward = type && !type->empty() &&
        !type->front()->in({TypeVar, DefaultInt, DefaultFloat}) &&
        (type->front() != Union || extract_backward_primitive(type));
      if (record_backward)
      {
        auto bit = bwd.find(loc);
        if (bit == bwd.end())
        {
          bwd[loc] = {clone(type), is_fixed, {}};
          changes.backward = true;
          changed = true;
        }
        else
        {
          if (is_fixed && !bit->second.is_fixed)
          {
            if (!same_type_tree(bit->second.type, type))
            {
              bit->second.type = clone(type);
              changed = true;
            }
            bit->second.is_fixed = true;
            changes.backward = true;
            return true;
          }

          if (bit->second.is_fixed)
            return changed;

          auto m = merge_type(bit->second.type, type, top);
          if (m)
          {
            bit->second.type = m;
            changes.backward = true;
            changed = true;
          }
        }
      }

      return changed;
    };

    auto pending_when_lookup_error = [&](const Node& when_arg) -> PendingError {
      auto def_it = all_def_stmts.find((when_arg / Rhs)->location());
      if (
        def_it == all_def_stmts.end() ||
        !def_it->second->in({CallDyn, TryCallDyn}))
      {
        return {};
      }

      auto lookup_it = lookup_stmts.find((def_it->second / Rhs)->location());
      if (lookup_it == lookup_stmts.end())
        return {};

      auto lookup_node = lookup_it->second;
      auto recv_it = env.find((lookup_node / Rhs)->location());
      if (
        recv_it == env.end() || contains_typevar(recv_it->second.type) ||
        contains_default_type(recv_it->second.type))
      {
        return {};
      }

      auto method_ident = lookup_method_name(lookup_node);
      auto hand = (lookup_node / Lhs)->type();
      auto method_ta = lookup_node / TypeArgs;
      auto arity = from_chars_sep_v<size_t>(lookup_node / Int);
      auto info = resolve_callable_method(
        top, recv_it->second.type, method_ident, hand, arity, method_ta);
      if (info.func)
        return {};

      return {
        method_ident,
        std::format(
          "lookup: type '{}' does not have method '{}'",
          infer_type_name(recv_it->second.type),
          std::string(method_ident->location().view()))};
    };

    for (auto& stmt : *body)
    {
      // ----- Const -----
      if (stmt == Const)
      {
        InferStmtScope stmt_scope(InferStmtFamily::ConstLike);
        auto dst = stmt->front();
        Node type_tok;
        if (stmt->size() == 3)
          type_tok = stmt->at(1);
        else
          type_tok = default_literal_type(stmt->back());

        auto type = type_tok->in({DefaultInt, DefaultFloat}) ?
          (Type << type_tok->type()) :
          primitive_or_ffi_type(type_tok->type());

        if (is_default_type(type))
        {
          auto it = env.find(dst->location());
          if (it != env.end() && !is_default_type(it->second.type))
          {
            auto prim = extract_primitive(it->second.type);
            if (
              prim &&
              ((type->front() == DefaultInt && prim->in(integer_types)) ||
               (type->front() == DefaultFloat && prim->in(float_types))))
            {
              type = clone(it->second.type);
            }
          }
        }

        merge(dst->location(), type);
      }
      // ----- ConstStr -----
      else if (stmt == ConstStr)
      {
        InferStmtScope stmt_scope(InferStmtFamily::ConstLike);
        merge((stmt / LocalId)->location(), string_type());
      }
      // ----- Convert -----
      else if (stmt == Convert)
      {
        InferStmtScope stmt_scope(InferStmtFamily::ConstLike);
        merge((stmt / LocalId)->location(), clone(stmt / Type));
      }
      // ----- TypedMove -----
      else if (stmt == TypedMove)
      {
        InferStmtScope stmt_scope(InferStmtFamily::CopyLike);
        auto dst_loc = (stmt / LocalId)->location();
        auto src_loc = (stmt / Rhs)->location();
        auto expected = stmt / Type;

        merge(dst_loc, clone(expected));
        auto dst_it = env.find(dst_loc);
        if (dst_it != env.end())
          dst_it->second.is_fixed = true;

        auto tuple_ref = ref_to_tuple.find(src_loc);
        if (tuple_ref != ref_to_tuple.end())
          upsert_ref_to_tuple(ref_to_tuple, dst_loc, tuple_ref->second);

        refine_local_const(src_loc, expected);
        propagate_call_constraint(
          env, src_loc, expected, top, lookup_stmts, &all_def_stmts);
        if (merge_bwd(src_loc, expected, true))
        {
          propagate_call_constraint(
            env, src_loc, expected, top, lookup_stmts, &all_def_stmts);
          propagate_call_node(
            env, src_loc, top, lookup_stmts, &all_def_stmts);
        }
      }
      // ----- Copy / Move -----
      else if (stmt->in({Copy, Move}))
      {
        InferStmtScope stmt_scope(InferStmtFamily::CopyLike);
        auto dst_loc = (stmt / LocalId)->location();
        auto src_loc = (stmt / Rhs)->location();
        auto src_it = env.find(src_loc);
        auto dst_it = env.find(dst_loc);

        // Track TypeVar aliases for post-convergence back-prop.
        if (src_it != env.end() && dst_it != env.end())
        {
          bool dst_tv = dst_it->second.type->front() == TypeVar;
          bool src_tv = src_it->second.type->front() == TypeVar;
          if (dst_tv != src_tv)
            if (typevar_aliases.insert({dst_loc, src_loc}).second)
              note_infer_transfer_change();
        }

        // Forward: dst = merge(dst, src).
        if (src_it != env.end())
          merge(dst_loc, src_it->second.type, src_it->second.call_node);

        auto tuple_ref = ref_to_tuple.find(src_loc);
        if (tuple_ref != ref_to_tuple.end())
          upsert_ref_to_tuple(ref_to_tuple, dst_loc, tuple_ref->second);

        // Backward: src = merge(src, dst).
        dst_it = env.find(dst_loc);
        if (dst_it != env.end())
        {
          Node expected = dst_it->second.type;
          Node call_expected = expected;
          auto bwd_it = bwd.find(dst_loc);
          if (
            bwd_it != bwd.end() && bwd_it->second.type &&
            !bwd_it->second.type->empty() &&
            !bwd_it->second.type->front()->in(
              {TypeVar, DefaultInt, DefaultFloat, Union}))
          {
            expected = bwd_it->second.type;
          }
          else if (
            bwd_it != bwd.end() &&
            extract_backward_primitive(bwd_it->second.type))
          {
            call_expected = bwd_it->second.type;
          }

          if (
            (dst_it->second.is_fixed ||
             (bwd_it != bwd.end() && bwd_it->second.is_fixed)))
            refine_local_const(src_loc, expected);
          if (call_expected)
            propagate_call_constraint(
              env, src_loc, call_expected, top, lookup_stmts, &all_def_stmts);
          bool expected_fixed = dst_it->second.is_fixed ||
            (bwd_it != bwd.end() && bwd_it->second.is_fixed);
          if (is_default_type(dst_it->second.type))
          {
            if (merge_bwd(dst_loc, expected, expected_fixed))
            {
              propagate_call_constraint(
                env, dst_loc, expected, top, lookup_stmts, &all_def_stmts);
              propagate_call_node(
                env, dst_loc, top, lookup_stmts, &all_def_stmts);
            }
          }
          if (merge_bwd(src_loc, expected, expected_fixed))
          {
            propagate_call_constraint(
              env, src_loc, expected, top, lookup_stmts, &all_def_stmts);
            propagate_call_node(
              env, src_loc, top, lookup_stmts, &all_def_stmts);
          }
        }
      }
      // ----- RegisterRef -----
      else if (stmt == RegisterRef)
      {
        InferStmtScope stmt_scope(InferStmtFamily::RefOps);
        auto src_it = env.find((stmt / Rhs)->location());
        if (src_it != env.end())
          merge(
            (stmt / LocalId)->location(), ref_type(clone(src_it->second.type)));
      }
      // ----- FieldRef -----
      else if (stmt == FieldRef)
      {
        InferStmtScope stmt_scope(InferStmtFamily::RefOps);
        auto arg_src = (stmt / Arg) / Rhs;
        auto dst_loc = (stmt / LocalId)->location();
        auto obj_it = env.find(arg_src->location());
        if (obj_it != env.end())
        {
          auto inner = obj_it->second.type->front();
          if (inner == TypeName)
          {
            auto class_def = find_def(top, inner);
            if (class_def && class_def == ClassDef)
            {
              auto subst = build_class_subst(class_def, inner);
              auto fname = (stmt / FieldId)->location().view();
              for (auto& f : *(class_def / ClassBody))
              {
                if (f != FieldDef)
                  continue;
                if ((f / Ident)->location().view() != fname)
                  continue;
                auto ft = apply_subst(top, f / Type, subst);
                if (ft)
                  merge(dst_loc, ref_type(ft));

                auto dst_it = env.find(dst_loc);
                auto field_inner = dst_it != env.end() ?
                  extract_ref_inner(dst_it->second.type) :
                  Node{};
                auto class_ident = class_def / Ident;
                bool lambda_field =
                  class_ident->location().view().rfind("lambda$", 0) == 0;
                if (
                  lambda_field && field_inner &&
                  !contains_typevar(field_inner) &&
                  !contains_default_type(field_inner) &&
                  !is_any_type(field_inner))
                {
                  bool should_refine =
                    is_any_type(f / Type) || contains_typevar(f / Type);

                  if (!should_refine)
                  {
                    auto old_prim = extract_primitive(f / Type);
                    auto new_prim = extract_primitive(field_inner);
                    should_refine = old_prim && new_prim &&
                      (((old_prim->in(integer_types) &&
                         new_prim->in(integer_types)) ||
                        (old_prim->in(float_types) &&
                         new_prim->in(float_types))) &&
                       old_prim->type() != new_prim->type());
                  }

                  if (
                    should_refine &&
                    replace_if_changed(f, f / Type, clone(field_inner)))
                    changes.forward = true;
                }
                break;
              }
            }
          }
        }
      }
      // ----- Load -----
      else if (stmt == Load)
      {
        InferStmtScope stmt_scope(InferStmtFamily::RefOps);
        auto src_loc = (stmt / Rhs)->location();
        auto dst_loc = (stmt / LocalId)->location();
        auto src_it = env.find(src_loc);
        if (src_it != env.end())
        {
          auto inner = extract_ref_inner(src_it->second.type);
          if (inner)
            merge(dst_loc, inner);
        }

        auto dst_it = env.find(dst_loc);
        if (dst_it != env.end() && !is_default_type(dst_it->second.type))
        {
          merge_bwd(src_loc, ref_type(clone(dst_it->second.type)));
        }
      }
      // ----- Store -----
      else if (stmt == Store)
      {
        InferStmtScope stmt_scope(InferStmtFamily::RefOps);
        // WF: wfDst * wfSrc * Arg → LocalId(dst), Rhs(ref), Arg(val)
        auto dst_loc = (stmt / LocalId)->location();
        auto ref_loc = (stmt / Rhs)->location();
        auto val_loc = ((stmt / Arg) / Rhs)->location();

        auto ref_it = env.find(ref_loc);
        if (ref_it != env.end())
        {
          auto inner = extract_ref_inner(ref_it->second.type);
          if (inner)
          {
            auto expected = clone(inner);
            // Forward: dst = inner type.
            merge(dst_loc, expected);
            // Backward: refine stored value.
            if (!is_any_type(inner) && merge_bwd(val_loc, clone(inner)))
            {
              propagate_call_constraint(
                env, val_loc, expected, top, lookup_stmts, &all_def_stmts);
              propagate_call_node(
                env, val_loc, top, lookup_stmts, &all_def_stmts);
            }

            // Track tuple element types.
            auto rtt = ref_to_tuple.find(ref_loc);
            if (rtt != ref_to_tuple.end())
            {
              auto& [tup_loc, idx] = rtt->second;
              auto tt = tuple_locals.find(tup_loc);
              if (tt != tuple_locals.end() && idx < tt->second.size)
              {
                auto val_it = env.find(val_loc);
                if (val_it != env.end())
                {
                  tt->second.element_types[idx] = clone(val_it->second.type);
                  if (tt->second.is_array_lit)
                    tt->second.element_value_locs[idx] = val_loc;

                  auto tracked = infer_tracked_tuple_type(tt->second);
                  if (tracked)
                  {
                    auto tup_it = env.find(tup_loc);
                    if (
                      (tup_it == env.end()) ||
                      !same_type_tree(tup_it->second.type, tracked))
                    {
                      env[tup_loc] = {clone(tracked), false, {}};
                      changes.forward = true;
                    }
                  }
                }
              }
            }
          }
        }
      }
      // ----- ArrayRef / ArrayRefConst -----
      else if (stmt->in({ArrayRef, ArrayRefConst}))
      {
        InferStmtScope stmt_scope(InferStmtFamily::TupleOps);
        auto dst_loc = (stmt / LocalId)->location();
        auto arg_loc = ((stmt / Arg) / Rhs)->location();
        auto src_it = env.find(arg_loc);

        if (stmt == ArrayRefConst)
        {
          auto index = from_chars_sep_v<size_t>(stmt / Rhs);
          // Track ref_to_tuple for Store-based element tracking.
          auto rtt = ref_to_tuple.find(arg_loc);
          if (rtt != ref_to_tuple.end())
            upsert_ref_to_tuple(
              ref_to_tuple, dst_loc, {rtt->second.first, index});
          else
            upsert_ref_to_tuple(ref_to_tuple, dst_loc, {arg_loc, index});

          // Resolve element if source is TupleType.
          if (src_it != env.end())
          {
            auto inner = src_it->second.type->front();
            if (inner == TupleType && index < inner->size())
            {
              merge(dst_loc, ref_type(Type << clone(inner->at(index))));
              continue;
            }
          }
        }

        if (src_it != env.end())
          merge(dst_loc, ref_type(clone(src_it->second.type)));
      }
      // ----- ArrayRefFromEnd -----
      else if (stmt == ArrayRefFromEnd)
      {
        InferStmtScope stmt_scope(InferStmtFamily::TupleOps);
        auto arg_loc = ((stmt / Arg) / Rhs)->location();
        auto src_it = env.find(arg_loc);
        if (src_it != env.end())
        {
          auto inner = src_it->second.type->front();
          if (inner == TupleType)
          {
            auto offset = from_chars_sep_v<size_t>(stmt / Rhs);
            if (offset > 0 && offset <= inner->size())
            {
              auto index = inner->size() - offset;
              merge(
                (stmt / LocalId)->location(),
                ref_type(Type << clone(inner->at(index))));
              continue;
            }
          }

          merge(
            (stmt / LocalId)->location(), ref_type(clone(src_it->second.type)));
        }
      }
      // ----- SplatOp -----
      else if (stmt == SplatOp)
      {
        InferStmtScope stmt_scope(InferStmtFamily::TupleOps);
        auto arg_loc = ((stmt / Arg) / Rhs)->location();
        auto src_it = env.find(arg_loc);
        if (src_it != env.end() && src_it->second.type->front() == TupleType)
        {
          auto inner = src_it->second.type->front();
          auto before = from_chars_sep_v<size_t>(stmt / Lhs);
          auto after = from_chars_sep_v<size_t>(stmt / Rhs);
          if (before + after <= inner->size())
          {
            auto remaining = inner->size() - before - after;
            if (remaining == 0)
            {
              merge((stmt / LocalId)->location(), primitive_type(None));
            }
            else if (remaining == 1)
            {
              merge(
                (stmt / LocalId)->location(), Type << clone(inner->at(before)));
            }
            else
            {
              Node tup = TupleType;
              for (size_t i = before; i < before + remaining; i++)
                tup << clone(inner->at(i));
              merge((stmt / LocalId)->location(), Type << tup);
            }
            continue;
          }
        }

        merge((stmt / LocalId)->location(), make_type());
      }
      // ----- NewArray / NewArrayConst -----
      else if (stmt->in({NewArray, NewArrayConst}))
      {
        InferStmtScope stmt_scope(InferStmtFamily::TupleOps);
        auto dst_loc = (stmt / LocalId)->location();
        auto init_type = stmt / Type;
        auto dst_it = env.find(dst_loc);
        if (!(dst_it != env.end() && is_any_type(init_type) &&
              !is_any_type(dst_it->second.type) &&
              (dst_it->second.type->front() != TypeVar)))
        {
          merge(dst_loc, clone(init_type));
        }

        if (stmt == NewArrayConst)
        {
          auto sz = from_chars_sep_v<size_t>(stmt / Rhs);
          bool is_lit = dst_loc.view().find("array") != std::string_view::npos;
          tuple_locals[dst_loc] = {
            sz, is_lit, std::vector<Node>(sz), std::vector<Location>(sz)};
          upsert_ref_to_tuple(ref_to_tuple, dst_loc, {dst_loc, 0});
        }
      }
      // ----- TypeAssertion -----
      else if (stmt == TypeAssertion)
      {
        InferStmtScope stmt_scope(InferStmtFamily::ConstLike);
        auto loc = (stmt / LocalId)->location();
        merge(loc, clone(stmt / Type));
        auto it = env.find(loc);
        if (it != env.end())
          it->second.is_fixed = true;
      }
      // ----- New / Stack -----
      else if (stmt->in({New, Stack}))
      {
        InferStmtScope stmt_scope(InferStmtFamily::NewOps);
        auto dst_loc = (stmt / LocalId)->location();
        auto new_type = stmt / Type;
        merge(dst_loc, clone(new_type));

        // Backward: constrain args from field types.
        auto inner = new_type->front();
        if (inner == TypeName)
        {
          auto class_def = find_def(top, inner);
          if (class_def && class_def == ClassDef)
          {
            auto subst = build_class_subst(class_def, inner);
            for (auto& na : *(stmt / NewArgs))
            {
              auto arg_loc = (na / Rhs)->location();
              auto fname = (na / Ident)->location().view();
              for (auto& f : *(class_def / ClassBody))
              {
                if (f != FieldDef)
                  continue;
                if ((f / Ident)->location().view() != fname)
                  continue;
                auto ft = apply_subst(top, f / Type, subst);
                if (ft && !contains_typevar(ft))
                {
                  auto expected = clone(ft);
                  refine_local_const(arg_loc, ft);
                  if (merge_bwd(arg_loc, ft))
                  {
                    propagate_call_constraint(
                      env,
                      arg_loc,
                      expected,
                      top,
                      lookup_stmts,
                      &all_def_stmts);
                    propagate_call_node(
                      env, arg_loc, top, lookup_stmts, &all_def_stmts);
                  }
                }
                // Reverse: push concrete arg into TypeVar FieldDef.
                auto arg_it = env.find(arg_loc);
                if (
                  arg_it != env.end() && contains_typevar(f / Type) &&
                  !contains_typevar(arg_it->second.type) &&
                  !contains_default_type(arg_it->second.type))
                  replace_if_changed(f, f / Type, clone(arg_it->second.type));
                break;
              }
            }
          }
        }
      }
      // ----- Binary ops (result = LHS type) -----
      else if (stmt->in(propagate_lhs_ops))
      {
        InferStmtScope stmt_scope(InferStmtFamily::CallOps);
        auto dst_loc = (stmt / LocalId)->location();
        auto lhs_loc = (stmt / Lhs)->location();
        auto rhs_loc = (stmt / Rhs)->location();

        auto lhs_it = env.find(lhs_loc);
        auto rhs_it = env.find(rhs_loc);

        // If the lhs is still a default numeric, use a concrete rhs
        // numeric type to pin the expression to that family.
        if (
          lhs_it != env.end() && rhs_it != env.end() &&
          is_default_type(lhs_it->second.type))
        {
          auto rhs_prim = extract_callable_primitive(rhs_it->second.type);
          bool compatible = rhs_prim &&
            ((lhs_it->second.type->front() == DefaultInt &&
              rhs_prim->in(integer_types)) ||
             (lhs_it->second.type->front() == DefaultFloat &&
              rhs_prim->in(float_types)));
          if (compatible)
          {
            auto refined = primitive_or_ffi_type(rhs_prim->type());
            refine_local_const(lhs_loc, refined);
            if (merge_bwd(lhs_loc, clone(refined)))
              propagate_call_node(
                env, lhs_loc, top, lookup_stmts, &all_def_stmts);
            merge(dst_loc, clone(refined));
            lhs_it = env.find(lhs_loc);
          }
        }

        if (lhs_it != env.end())
        {
          // Forward: result = lhs type.
          merge(dst_loc, clone(lhs_it->second.type));
          // Backward: refine rhs from lhs.
          if (merge_bwd(rhs_loc, clone(lhs_it->second.type)))
            propagate_call_node(
              env, rhs_loc, top, lookup_stmts, &all_def_stmts);
        }

        // Backward from dst: refine lhs and rhs from dst (from prior
        // iteration's backward flow, e.g., Call backward refined dst).
        auto dst_it = env.find(dst_loc);
        if (dst_it != env.end() && !is_default_type(dst_it->second.type))
        {
          if (merge_bwd(lhs_loc, clone(dst_it->second.type)))
            propagate_call_node(
              env, lhs_loc, top, lookup_stmts, &all_def_stmts);
          if (merge_bwd(rhs_loc, clone(dst_it->second.type)))
            propagate_call_node(
              env, rhs_loc, top, lookup_stmts, &all_def_stmts);
        }
      }
      // ----- Unary ops (result = operand type) -----
      else if (stmt->in(propagate_rhs_ops))
      {
        InferStmtScope stmt_scope(InferStmtFamily::CallOps);
        auto dst_loc = (stmt / LocalId)->location();
        auto src_loc = (stmt / Rhs)->location();

        auto src_it = env.find(src_loc);
        if (src_it != env.end())
          merge(dst_loc, clone(src_it->second.type));

        // Backward from dst.
        auto dst_it = env.find(dst_loc);
        if (dst_it != env.end() && !is_default_type(dst_it->second.type))
        {
          if (merge_bwd(src_loc, clone(dst_it->second.type)))
            propagate_call_node(
              env, src_loc, top, lookup_stmts, &all_def_stmts);
        }
      }
      // ----- Fixed result types -----
      else if (
        auto frt = fixed_result_type.find(stmt->type());
        frt != fixed_result_type.end())
      {
        InferStmtScope stmt_scope(InferStmtFamily::ConstLike);
        merge((stmt / LocalId)->location(), primitive_type(frt->second));
      }
      else if (
        auto ffrt = fixed_ffi_result_type.find(stmt->type());
        ffrt != fixed_ffi_result_type.end())
      {
        InferStmtScope stmt_scope(InferStmtFamily::ConstLike);
        merge((stmt / LocalId)->location(), ffi_primitive_type(ffrt->second));
      }
      else if (stmt == FFIStruct)
      {
        InferStmtScope stmt_scope(InferStmtFamily::ConstLike);
        merge((stmt / LocalId)->location(), ffi_struct_result_type());
      }
      else if (stmt == FFILoad)
      {
        InferStmtScope stmt_scope(InferStmtFamily::ConstLike);
        merge((stmt / LocalId)->location(), clone(stmt / Type));
      }
      else if (stmt == FFIStore)
      {
        InferStmtScope stmt_scope(InferStmtFamily::CallOps);
        auto value_loc = (stmt / ValueSrc)->location();
        auto expected = clone(stmt / Type);
        refine_local_const(value_loc, expected);
        if (merge_bwd(value_loc, expected))
          propagate_call_node(
            env, value_loc, top, lookup_stmts, &all_def_stmts);
      }
      // ----- Call -----
      else if (stmt == Call)
      {
        InferStmtScope stmt_scope(InferStmtFamily::CallOps);
        std::vector<ScopeInfo> scopes;
        auto func_def = navigate_call(stmt, top, scopes);
        if (!func_def)
          continue;

        auto args = stmt / Args;
        auto params = func_def / Params;

        // TypeArg inference.
        bool all_default = infer_typeargs(stmt, func_def, scopes, env, top);

        // Build substitution.
        NodeMap<Node> subst;
        for (auto& scope : scopes)
        {
          auto ta = scope.name_elem / TypeArgs;
          auto tps = scope.def / TypeParams;
          if (!ta->empty() && ta->size() == tps->size())
            for (size_t i = 0; i < tps->size(); i++)
              subst[tps->at(i)] = ta->at(i);
        }

        // Forward: return type.
        auto ret = apply_subst(top, func_def / Type, subst);
        if (ret)
          merge((stmt / LocalId)->location(), ret, all_default ? stmt : Node{});

        // Shape-to-lambda propagation.
        for (size_t i = 0; i < params->size() && i < args->size(); i++)
        {
          auto pt = apply_subst(top, params->at(i) / Type, subst);
          if (pt && pt->front() != TypeVar)
          {
            auto arg_it = env.find((args->at(i) / Rhs)->location());
            if (arg_it != env.end())
              changes.forward |=
                propagate_shape_to_lambda(top, pt, arg_it->second.type);
          }
        }

        // Backward: param types into args.
        for (size_t i = 0; i < params->size() && i < args->size(); i++)
        {
          auto expected = apply_subst(top, params->at(i) / Type, subst);
          if (
            expected && expected->front() != TypeVar &&
            !is_uninformative_backward_type(expected))
          {
            auto arg_loc = (args->at(i) / Rhs)->location();
            refine_local_const(arg_loc, expected);
            if (merge_bwd(arg_loc, expected))
            {
              propagate_call_constraint(
                env, arg_loc, expected, top, lookup_stmts, &all_def_stmts);
              propagate_call_node(
                env, arg_loc, top, lookup_stmts, &all_def_stmts);
            }

            auto expected_prim = extract_backward_primitive(expected);
            auto def_it = def_stmts.find(arg_loc);
            if (
              expected_prim && def_it != def_stmts.end() &&
              def_it->second->in({CallDyn, TryCallDyn}))
            {
              backward_refine_calldyn(
                def_it->second,
                expected_prim,
                env,
                top,
                lookup_stmts,
                &all_def_stmts);
              changes.forward = true;
            }
          }
        }

        // Forward into callee: push arg types into TypeVar params.
        push_arg_types_to_params(func_def, args, env, top);
      }
      // ----- Lookup -----
      else if (stmt == Lookup)
      {
        InferStmtScope stmt_scope(InferStmtFamily::CallOps);
        auto dst_loc = (stmt / LocalId)->location();
        auto src_it = env.find((stmt / Rhs)->location());
        if (src_it != env.end())
        {
          // When receiver is default-typed, propagate the default type
          // as the Lookup result. This makes the CallDyn result default,
          // enabling call_node tracking for backward refinement.
          if (is_default_type(src_it->second.type))
          {
            merge(dst_loc, clone(src_it->second.type));
          }
          else
          {
            auto hand = (stmt / Lhs)->type();
            auto method_ident = lookup_method_name(stmt);
            auto method_ta = stmt / TypeArgs;
            auto arity = from_chars_sep_v<size_t>(stmt / Int);
            auto ret = resolve_method_return_type(
              top, src_it->second.type, method_ident, hand, arity, method_ta);
            if (ret)
              merge(dst_loc, ret);
          }
        }
        upsert_lookup_stmt(lookup_stmts, dst_loc, stmt);
      }
      // ----- CallDyn / TryCallDyn -----
      else if (stmt->in({CallDyn, TryCallDyn}))
      {
        InferStmtScope stmt_scope(InferStmtFamily::CallOps);
        auto dst_loc = (stmt / LocalId)->location();
        auto src_loc = (stmt / Rhs)->location();
        auto args = stmt / Args;
        bool refined = false;
        bool resolved_callable = false;

        // Forward: result from Lookup.
        auto src_it = env.find(src_loc);
        if (src_it != env.end())
          merge(dst_loc, clone(src_it->second.type));

        // Preserve the producing dynamic call so later backward constraints
        // can still reach it after the result becomes concrete or a union.
        auto dst_it = env.find(dst_loc);
        if (dst_it != env.end() && !dst_it->second.call_node)
          dst_it->second.call_node = stmt;

        // Resolve method for backward refinement.
        // Skip when receiver is default-typed — the method resolution
        // would use the fallback type (u64/f64), pushing wrong types
        // into args. The correct type will be determined by backward
        // refinement from downstream constraints.
        auto lookup_it = lookup_stmts.find(src_loc);
        if (lookup_it != lookup_stmts.end())
        {
          auto lookup_node = lookup_it->second;
          auto recv_it = env.find((lookup_node / Rhs)->location());
          if (recv_it != env.end() && !is_default_type(recv_it->second.type))
          {
            auto hand = (lookup_node / Lhs)->type();
            auto method_ident = lookup_method_name(lookup_node);
            auto method_ta = lookup_node / TypeArgs;
            auto arity = from_chars_sep_v<size_t>(lookup_node / Int);
            auto info = resolve_callable_method(
              top, recv_it->second.type, method_ident, hand, arity, method_ta);
            if (info.func)
            {
              resolved_callable = true;
              auto params = info.func / Params;

              // Shape-to-lambda propagation.
              for (size_t i = 0; i < params->size() && i < args->size(); i++)
              {
                auto pt = apply_subst(top, params->at(i) / Type, info.subst);
                if (pt && pt->front() != TypeVar)
                {
                  auto arg_it = env.find((args->at(i) / Rhs)->location());
                  if (arg_it != env.end())
                    refined |=
                      propagate_shape_to_lambda(top, pt, arg_it->second.type);
                }
              }

              // Backward: param types into args.
              for (size_t i = 0; i < params->size() && i < args->size(); i++)
              {
                auto expected =
                  apply_subst(top, params->at(i) / Type, info.subst);
                if (
                  expected && expected->front() != TypeVar &&
                  !is_uninformative_backward_type(expected))
                {
                  auto arg_loc = (args->at(i) / Rhs)->location();
                  refine_local_const(arg_loc, expected);
                  if (merge_bwd(arg_loc, expected))
                  {
                    refined = true;
                    propagate_call_constraint(
                      env,
                      arg_loc,
                      expected,
                      top,
                      lookup_stmts,
                      &all_def_stmts);
                    propagate_call_node(
                      env, arg_loc, top, lookup_stmts, &all_def_stmts);
                  }
                }
              }

              // Forward into callee: TypeVar params.
              push_arg_types_to_params(info.func, args, env, top);
            }
          }
        }

        if (!refined && !resolved_callable)
        {
          Node target_prim;
          auto lookup_it = lookup_stmts.find(src_loc);
          bool allow_arg_fallback = false;

          if (lookup_it != lookup_stmts.end())
          {
            auto recv_it = env.find((lookup_it->second / Rhs)->location());
            if (recv_it != env.end() && is_default_type(recv_it->second.type))
            {
              allow_arg_fallback = true;
            }
          }

          if (allow_arg_fallback)
          {
            for (auto& arg_node : *args)
            {
              auto arg_it = env.find((arg_node / Rhs)->location());
              if (
                arg_it == env.end() ||
                contains_default_type(arg_it->second.type))
              {
                continue;
              }

              auto prim = extract_callable_primitive(arg_it->second.type);
              if (!prim)
                prim = extract_wrapper_primitive(arg_it->second.type);
              if (prim)
              {
                target_prim = prim;
                break;
              }
            }
          }

          if (!target_prim)
          {
            if (lookup_it != lookup_stmts.end())
            {
              auto recv_it = env.find((lookup_it->second / Rhs)->location());
              if (
                recv_it != env.end() && !is_default_type(recv_it->second.type))
              {
                auto prim = extract_callable_primitive(recv_it->second.type);
                if (!prim)
                  prim = extract_wrapper_primitive(recv_it->second.type);
                if (prim)
                  target_prim = prim;
              }
            }
          }

          if (target_prim)
          {
            auto target_type = primitive_or_ffi_type(target_prim->type());
            auto lookup_it = lookup_stmts.find(src_loc);
            if (lookup_it != lookup_stmts.end())
            {
              auto recv_loc = (lookup_it->second / Rhs)->location();
              bool local_refined = refine_local_const(recv_loc, target_type);
              bool fwd_refined = merge(recv_loc, target_type);
              bool bwd_refined = merge_bwd(recv_loc, target_type);
              if (bwd_refined)
                propagate_call_node(
                  env, recv_loc, top, lookup_stmts, &all_def_stmts);
              refined = refined || local_refined || fwd_refined || bwd_refined;
            }

            for (auto& arg_node : *args)
            {
              auto arg_loc = (arg_node / Rhs)->location();
              bool local_refined = refine_local_const(arg_loc, target_type);
              bool fwd_refined = merge(arg_loc, target_type);
              bool bwd_refined = merge_bwd(arg_loc, target_type);
              if (bwd_refined)
                propagate_call_node(
                  env, arg_loc, top, lookup_stmts, &all_def_stmts);
              refined = refined || local_refined || fwd_refined || bwd_refined;
            }

            if (lookup_it != lookup_stmts.end())
            {
              auto lookup_node = lookup_it->second;
              auto recv_it = env.find((lookup_node / Rhs)->location());
              if (
                recv_it != env.end() &&
                !contains_default_type(recv_it->second.type))
              {
                auto hand = (lookup_node / Lhs)->type();
                auto method_ident = lookup_method_name(lookup_node);
                auto method_ta = lookup_node / TypeArgs;
                auto arity = from_chars_sep_v<size_t>(lookup_node / Int);
                auto ret = resolve_method_return_type(
                  top,
                  recv_it->second.type,
                  method_ident,
                  hand,
                  arity,
                  method_ta);
                if (ret)
                {
                  env[(lookup_node / LocalId)->location()] = {
                    clone(ret), false, {}};
                  env[dst_loc] = {clone(ret), false, {}};
                }
              }
            }
          }
        }
      }
      // ----- FFI -----
      else if (stmt == FFI)
      {
        InferStmtScope stmt_scope(InferStmtFamily::FFIWhenOps);
        auto dst_loc = (stmt / LocalId)->location();
        auto sym_name = (stmt / SymbolId)->location();
        auto cls = body->parent(Function)->parent(ClassDef);

        while (cls)
        {
          bool found = false;
          for (auto& child : *(cls / ClassBody))
          {
            if (child != Lib)
              continue;
            for (auto& sym : *(child / Symbols))
            {
              if (sym != Symbol)
                continue;
              if ((sym / SymbolId)->location() != sym_name)
                continue;

              // Forward: return type.
              auto ret_type = sym / Type;
              if (!ret_type->empty())
                merge(dst_loc, clone(ret_type));

              // Backward: param types into args.
              auto ffi_params = sym / FFIParams;
              auto ffi_args = stmt / Args;
              auto fp = ffi_params->begin();
              auto fa = ffi_args->begin();
              while (fp != ffi_params->end() && fa != ffi_args->end())
              {
                refine_local_const((*fa)->location(), clone(*fp));
                if (merge_bwd((*fa)->location(), clone(*fp)))
                  propagate_call_node(
                    env, (*fa)->location(), top, lookup_stmts, &all_def_stmts);
                ++fp;
                ++fa;
              }
              found = true;
              break;
            }
            if (found)
              break;
          }
          if (found)
            break;
          cls = cls->parent(ClassDef);
        }
      }
      // ----- When -----
      else if (stmt == When)
      {
        InferStmtScope stmt_scope(InferStmtFamily::FFIWhenOps);
        auto dst_loc = (stmt / LocalId)->location();
        auto src_it = env.find((stmt / Rhs)->location());

        if (src_it != env.end())
        {
          auto apply_ret = src_it->second.type;
          replace_if_changed(stmt, stmt / Type, clone(apply_ret));
          merge(dst_loc, cown_type(apply_ret));
        }

        // Set lambda params from cown types.
        auto lookup_it = lookup_stmts.find((stmt / Rhs)->location());
        if (lookup_it != lookup_stmts.end())
        {
          auto recv_it = env.find((lookup_it->second / Rhs)->location());
          if (recv_it != env.end())
          {
            auto recv_inner = recv_it->second.type->front();
            if (recv_inner == TypeName)
            {
              auto class_def = find_def(top, recv_inner);
              if (class_def && class_def == ClassDef)
              {
                Node apply_func;
                for (auto& child : *(class_def / ClassBody))
                {
                  if (
                    child == Function &&
                    (child / Ident)->location().view() == "apply")
                  {
                    apply_func = child;
                    break;
                  }
                }

                if (apply_func)
                {
                  auto params = apply_func / Params;
                  auto when_args = stmt / Args;
                  for (size_t i = 1;
                       i < when_args->size() && i < params->size();
                       ++i)
                  {
                    auto param = params->at(i);
                    auto arg_it =
                      env.find((when_args->at(i) / Rhs)->location());
                    if (arg_it == env.end())
                    {
                      auto pending =
                        pending_when_lookup_error(when_args->at(i));
                      if (
                        pending.site &&
                        deferred_param_errors.find(
                          (param / Ident)->location()) ==
                          deferred_param_errors.end())
                      {
                        deferred_param_errors[(param / Ident)->location()] =
                          std::move(pending);
                      }
                      continue;
                    }
                    auto ci = extract_cown_inner(arg_it->second.type);
                    if (!ci)
                    {
                      auto pending =
                        pending_when_lookup_error(when_args->at(i));
                      if (
                        pending.site &&
                        deferred_param_errors.find(
                          (param / Ident)->location()) ==
                          deferred_param_errors.end())
                      {
                        deferred_param_errors[(param / Ident)->location()] =
                          std::move(pending);
                      }
                      continue;
                    }
                    auto new_type = ref_type(ci);
                    replace_if_changed(param, param / Type, new_type);
                    env[(param / Ident)->location()] = {
                      clone(new_type), true, {}};
                  }
                }
              }
            }
          }
        }
      }
      // ----- Typetest -----
      else if (stmt == Typetest)
      {
        InferStmtScope stmt_scope(InferStmtFamily::TypetestOps);
        merge((stmt / LocalId)->location(), primitive_type(Bool));
      }
    }

    // Finalize tuple/array-lit types within this label.
    for (auto& [loc, tt] : tuple_locals)
    {
      InferScopedTimer tuple_finalize_timer(
        (active_infer_profile != nullptr) ?
          &active_infer_profile->tuple_finalize_time :
          nullptr);
      if (active_infer_profile != nullptr)
        active_infer_profile->tuple_finalize_ops++;
      if (tt.is_array_lit)
      {
        // Sibling refinement: find dominant non-default type.
        Node dom_prim;
        for (size_t i = 0; i < tt.size; i++)
        {
          if (tt.element_value_locs[i].view().empty())
            continue;
          auto it = env.find(tt.element_value_locs[i]);
          if (it == env.end() || is_default_type(it->second.type))
            continue;
          dom_prim = extract_primitive(it->second.type);
          if (dom_prim)
            break;
        }
        if (dom_prim)
        {
          auto dom_type = primitive_or_ffi_type(dom_prim->type());
          for (size_t i = 0; i < tt.size; i++)
          {
            if (!tt.element_value_locs[i].view().empty())
              merge(tt.element_value_locs[i], dom_type);
          }
        }

        // Determine homogeneous type.
        Node common;
        bool uniform = true;
        for (size_t i = 0; i < tt.size && uniform; i++)
        {
          if (tt.element_value_locs[i].view().empty())
          {
            uniform = false;
            break;
          }
          auto it = env.find(tt.element_value_locs[i]);
          if (it == env.end() || !extract_primitive(it->second.type))
          {
            uniform = false;
            break;
          }
          if (!common)
            common = clone(it->second.type);
          else if (it->second.type->front()->type() != common->front()->type())
            uniform = false;
        }
        if (uniform && common && tt.size > 0)
          env[loc] = {clone(common), false, {}};
      }
      else
      {
        auto tracked = infer_tracked_tuple_type(tt);
        if (tracked)
          env[loc] = {tracked, false, {}};
      }
    }

    return changes;
  }

  // ===== Fixpoint algorithm =====

  static bool process_function(Node node, Node top, bool check_errors)
  {
    auto labels = node / Labels;
    size_t n = labels->size();
    if (n == 0)
      return true;

    InferProfileStats profile;
    profile.function_name = infer_function_name(node);
    profile.labels = n;
    InferProfileStats* prev_profile = active_infer_profile;
    MethodLookupCache method_cache;
    MethodLookupCache* prev_method_cache = active_method_cache;
    size_t transfer_epoch = 0;
    size_t* prev_transfer_epoch = active_infer_transfer_epoch;
    bool profile_enabled = infer_profile_enabled();
    if (profile_enabled)
      active_infer_profile = &profile;
    active_method_cache = &method_cache;
    active_infer_transfer_epoch = &transfer_epoch;
    InferProcessScope process_scope(
      profile_enabled ? &profile : nullptr,
      prev_profile,
      prev_method_cache,
      prev_transfer_epoch);
    InferScopedTimer function_timer(
      (active_infer_profile != nullptr) ?
        &active_infer_profile->process_function_time :
        nullptr);

    // Build label graph.
    std::map<std::string, size_t> label_idx;
    for (size_t i = 0; i < n; i++)
      label_idx[std::string((labels->at(i) / LabelId)->location().view())] = i;

    std::vector<std::vector<size_t>> succ(n), pred(n);
    for (size_t i = 0; i < n; i++)
    {
      auto term = labels->at(i) / Return;
      if (term == Cond)
      {
        auto t = label_idx.find(std::string((term / Lhs)->location().view()));
        auto f = label_idx.find(std::string((term / Rhs)->location().view()));
        if (t != label_idx.end())
          succ[i].push_back(t->second);
        if (f != label_idx.end())
          succ[i].push_back(f->second);
      }
      else if (term == Jump)
      {
        auto t =
          label_idx.find(std::string((term / LabelId)->location().view()));
        if (t != label_idx.end())
          succ[i].push_back(t->second);
      }
    }
    for (size_t i = 0; i < n; i++)
      for (auto s : succ[i])
        pred[s].push_back(i);

    // Initialize exit envs.
    std::vector<TypeEnv> exit_envs(n);
    std::map<std::pair<size_t, size_t>, TypeEnv> branch_exits;
    std::vector<TypeEnv> prior_entry_envs(n);
    std::vector<bool> prior_entry_valid(n, false);
    std::vector<size_t> prior_transfer_epochs(n, 0);

    for (auto& pd : *(node / Params))
    {
      auto type = pd / Type;
      bool fixed = type->front() != TypeVar;
      exit_envs[0][(pd / Ident)->location()] = {clone(type), fixed, {}};
    }

    // Backward envs: carry type expectations from downstream.
    std::vector<TypeEnv> bwd_envs(n);
    std::vector<std::set<Location>> label_defs(n);
    std::vector<std::set<Location>> label_uses(n);
    std::vector<std::set<Location>> label_kills(n);
    std::vector<std::set<Location>> label_live_in(n);
    std::vector<std::set<Location>> label_live_out(n);
    for (size_t j = 0; j < n; j++)
    {
      label_defs[j] = collect_label_defs(labels->at(j) / Body);
      label_uses[j] = collect_label_uses(
        labels->at(j) / Body, labels->at(j) / Return, label_defs[j]);
      label_kills[j] = collect_label_kills(labels->at(j) / Body);
    }

    bool live_changed = true;
    while (live_changed)
    {
      live_changed = false;
      for (size_t j = n; j-- > 0;)
      {
        std::set<Location> next_out;
        for (auto s : succ[j])
          next_out.insert(label_live_in[s].begin(), label_live_in[s].end());

        std::set<Location> next_in = label_uses[j];
        for (auto& loc : next_out)
        {
          if (label_kills[j].count(loc) == 0)
            next_in.insert(loc);
        }

        if (next_out != label_live_out[j] || next_in != label_live_in[j])
        {
          label_live_out[j] = std::move(next_out);
          label_live_in[j] = std::move(next_in);
          live_changed = true;
        }
      }
    }
    // Initialize backward env for Return labels with declared return type.
    {
      auto func_ret = node / Type;
      if (!contains_typevar(func_ret) && !is_default_type(func_ret))
      {
        for (size_t j = 0; j < n; j++)
        {
          auto term = labels->at(j) / Return;
          if (term == Return)
          {
            auto ret_loc = (term / LocalId)->location();
            bwd_envs[j][ret_loc] = {clone(func_ret), true, {}};
          }
        }
      }
    }

    // Shared state.
    std::map<Location, Node> all_def_stmts;
    for (auto& label : *labels)
      for (auto& stmt : *(label / Body))
        if (!stmt->empty() && stmt->front() == LocalId)
          all_def_stmts[stmt->front()->location()] = stmt;

    std::map<Location, Node> lookup_stmts;
    std::set<std::pair<Location, Location>> typevar_aliases;
    std::map<Location, std::pair<Location, size_t>> ref_to_tuple;

    // Worklist algorithm.
    std::set<size_t> worklist;
    for (size_t i = 0; i < n; i++)
      worklist.insert(i);

    size_t wl_iters = 0;
    while (!worklist.empty())
    {
      wl_iters++;
      if (wl_iters > n * 100)
        break;

      auto wit = worklist.begin();
      size_t i = *wit;
      worklist.erase(wit);

      prune_bwd_env(bwd_envs[i], label_live_in[i], label_defs[i]);

      // Build the working env directly from predecessor exit envs.
      TypeEnv env;
      {
        InferScopedTimer entry_timer(
          (active_infer_profile != nullptr) ?
            &active_infer_profile->entry_build_time :
            nullptr);
        if (i == 0)
        {
          env = exit_envs[0];
        }
        else
        {
          InferScopedTimer pred_timer(
            (active_infer_profile != nullptr) ?
              &active_infer_profile->entry_pred_time :
              nullptr);
          for (auto p : pred[i])
          {
            auto bk = std::make_pair(p, i);
            auto bi = branch_exits.find(bk);
            const auto& pe =
              (bi != branch_exits.end()) ? bi->second : exit_envs[p];
            for (auto& [loc, info] : pe)
            {
              if (label_live_in[i].count(loc) == 0)
                continue;
              if (active_infer_profile != nullptr)
                active_infer_profile->entry_pred_merge_count++;
              auto eit = env.find(loc);
              if (eit == env.end())
                env[loc] = {clone(info.type), info.is_fixed, info.call_node};
              else
              {
                if (same_type_tree(eit->second.type, info.type))
                {
                  eit->second.is_fixed = eit->second.is_fixed || info.is_fixed;
                  if (!eit->second.call_node && info.call_node)
                    eit->second.call_node = info.call_node;
                  continue;
                }

                if (eit->second.is_fixed && !info.is_fixed)
                {
                  if (!eit->second.call_node && info.call_node)
                    eit->second.call_node = info.call_node;
                  continue;
                }

                if (!eit->second.is_fixed && info.is_fixed)
                {
                  eit->second.type = clone(info.type);
                  eit->second.is_fixed = true;
                  if (!eit->second.call_node && info.call_node)
                    eit->second.call_node = info.call_node;
                  continue;
                }

                auto m = merge_type(eit->second.type, info.type, top);
                if (m)
                  eit->second.type = m;
                eit->second.is_fixed = eit->second.is_fixed || info.is_fixed;
                if (!eit->second.call_node && info.call_node)
                  eit->second.call_node = info.call_node;
              }
            }
          }
        }

        // Merge backward envs for this label and its successors into entry.
        // Including bwd_envs[i] lets same-label requeueing expose new local
        // backward facts to earlier statements on the next iteration.
        {
          InferScopedTimer self_bwd_timer(
            (active_infer_profile != nullptr) ?
              &active_infer_profile->entry_self_bwd_time :
              nullptr);
          for (auto& [loc, info] : bwd_envs[i])
          {
            if (
              (label_live_in[i].count(loc) == 0) &&
              (label_defs[i].count(loc) == 0))
              continue;
            if (active_infer_profile != nullptr)
              active_infer_profile->entry_self_bwd_merge_count++;
            auto eit = env.find(loc);
            if (eit == env.end())
            {
              if (label_defs[i].count(loc) > 0)
                env[loc] = {clone(info.type), info.is_fixed, {}};
            }
            else
            {
              if (same_type_tree(eit->second.type, info.type))
              {
                eit->second.is_fixed = eit->second.is_fixed || info.is_fixed;
                continue;
              }
              if (eit->second.is_fixed && !info.is_fixed)
                continue;
              // Don't widen: if the forward type is already a concrete
              // (non-default) non-union type and the backward is a union,
              // keep the forward type — it's more precise.
              if (
                eit->second.type && info.type && !eit->second.type->empty() &&
                !info.type->empty() && info.type->front() == Union &&
                eit->second.type->front() != Union &&
                !is_default_type(eit->second.type))
              {
                continue;
              }
              if (!eit->second.is_fixed && info.is_fixed)
              {
                eit->second.type = clone(info.type);
                eit->second.is_fixed = true;
                continue;
              }
              auto m = merge_type(eit->second.type, info.type, top);
              if (m)
                eit->second.type = m;
            }
          }
        }

        // Merge backward envs from successors into entry.
        {
          InferScopedTimer succ_bwd_timer(
            (active_infer_profile != nullptr) ?
              &active_infer_profile->entry_succ_bwd_time :
              nullptr);
          for (auto s : succ[i])
          {
            for (auto& [loc, info] : bwd_envs[s])
            {
              if (
                (label_live_in[i].count(loc) == 0) &&
                (label_defs[i].count(loc) == 0))
                continue;
              if (active_infer_profile != nullptr)
                active_infer_profile->entry_succ_bwd_merge_count++;
              auto eit = env.find(loc);
              if (eit == env.end())
              {
                if (label_defs[i].count(loc) > 0)
                  env[loc] = {clone(info.type), info.is_fixed, {}};
              }
              else
              {
                if (same_type_tree(eit->second.type, info.type))
                {
                  eit->second.is_fixed = eit->second.is_fixed || info.is_fixed;
                  continue;
                }
                if (eit->second.is_fixed && !info.is_fixed)
                  continue;
                if (
                  eit->second.type && info.type && !eit->second.type->empty() &&
                  !info.type->empty() && info.type->front() == Union &&
                  eit->second.type->front() != Union &&
                  !is_default_type(eit->second.type))
                {
                  continue;
                }
                if (!eit->second.is_fixed && info.is_fixed)
                {
                  eit->second.type = clone(info.type);
                  eit->second.is_fixed = true;
                  continue;
                }
                auto m = merge_type(eit->second.type, info.type, top);
                if (m)
                  eit->second.type = m;
              }
            }
          }
        }

        // Restore forward metadata for locals defined in this label when the
        // entry env was reconstructed mostly from backward facts.
        for (auto& loc : label_defs[i])
        {
          auto eit = env.find(loc);
          if (eit == env.end())
            continue;

          auto def_it = all_def_stmts.find(loc);
          if (def_it == all_def_stmts.end())
            continue;

          if (
            !eit->second.call_node &&
            def_it->second->in({Call, CallDyn, TryCallDyn}))
            eit->second.call_node = def_it->second;

          if (
            eit->second.type && !eit->second.type->empty() &&
            eit->second.type->front() != TypeVar &&
            !is_default_type(eit->second.type))
            continue;

          TypeEnv recovered;
          if (recover_local_type_from_def(loc, recovered, all_def_stmts, top))
          {
            auto rit = recovered.find(loc);
            if (rit != recovered.end())
              eit->second.type = clone(rit->second.type);
          }
        }
      }
      bool incoming_bwd_changed = false;
      for (auto s : succ[i])
      {
        for (auto& [loc, info] : bwd_envs[s])
        {
          if (
            (label_live_in[i].count(loc) == 0) &&
            (label_defs[i].count(loc) == 0))
            continue;
          auto bit = bwd_envs[i].find(loc);
          if (bit == bwd_envs[i].end())
          {
            incoming_bwd_changed = true;
            break;
          }
          if (same_type_tree(bit->second.type, info.type))
          {
            if (!bit->second.is_fixed && info.is_fixed)
              incoming_bwd_changed = true;
            continue;
          }
          if (bit->second.is_fixed && !info.is_fixed)
            continue;
          incoming_bwd_changed = true;
          break;
        }
        if (incoming_bwd_changed)
          break;
      }

      if (active_infer_profile != nullptr)
        active_infer_profile->entry_bindings += env.size();
      bool same_entry =
        prior_entry_valid[i] && same_type_env(prior_entry_envs[i], env);
      if (same_entry)
      {
        if (active_infer_profile != nullptr)
          active_infer_profile->entry_env_unchanged++;
        if (!incoming_bwd_changed && prior_transfer_epochs[i] == transfer_epoch)
        {
          if (active_infer_profile != nullptr)
            active_infer_profile->labels_skipped++;
          continue;
        }
      }
      else
      {
        if (active_infer_profile != nullptr)
          active_infer_profile->entry_env_changed++;
      }
      prior_entry_envs[i] = env;
      prior_entry_valid[i] = true;

      // Per-label state.
      std::map<Location, TupleTracking> tuple_locals;

      // Process body.
      if (active_infer_profile != nullptr)
      {
        active_infer_profile->labels_processed++;
        active_infer_profile->process_label_body_calls++;
      }
      LabelChanges label_changes;
      {
        InferScopedTimer body_timer(
          (active_infer_profile != nullptr) ?
            &active_infer_profile->process_label_body_time :
            nullptr);
        label_changes = process_label_body(
          labels->at(i) / Body,
          env,
          bwd_envs[i],
          top,
          lookup_stmts,
          all_def_stmts,
          typevar_aliases,
          ref_to_tuple,
          tuple_locals);
      }
      bool forward_out_changed = false;

      // Return terminator: backward merge.
      auto term = labels->at(i) / Return;
      if (term == Return)
      {
        auto func_ret = node / Type;
        if (func_ret->front() != TypeVar && !is_default_type(func_ret))
        {
          std::map<Location, Node> const_defs;
          for (auto& stmt : *(labels->at(i) / Body))
            if (stmt == Const)
              const_defs[(stmt / LocalId)->location()] = stmt;

          auto ret_loc = (term / LocalId)->location();
          auto ret_prim = extract_backward_primitive(func_ret);
          if (ret_prim)
            refine_const_local(
              env,
              const_defs,
              ret_loc,
              primitive_or_ffi_type(ret_prim->type()));
          propagate_call_constraint(
            env, ret_loc, func_ret, top, lookup_stmts, &all_def_stmts);
          bool changed = merge_env(env, ret_loc, clone(func_ret), top);
          auto ret_it = env.find(ret_loc);
          if (ret_it != env.end())
            ret_it->second.is_fixed = true;
          bwd_envs[i][ret_loc] = {clone(func_ret), true, {}};
          if (changed)
          {
            propagate_call_node(
              env, ret_loc, top, lookup_stmts, &all_def_stmts);
          }
        }
      }
      else if (term == Raise)
      {
        std::map<Location, Node> const_defs;
        for (auto& stmt : *(labels->at(i) / Body))
          if (stmt == Const)
            const_defs[(stmt / LocalId)->location()] = stmt;

        auto raise_ret = term / Type;
        auto ret_loc = (term / LocalId)->location();
        if (refine_const_local(env, const_defs, ret_loc, raise_ret))
          label_changes.forward = true;
      }
      // Cond: branch exits with typetest narrowing.
      if (term == Cond)
      {
        auto trace = trace_typetest(term / LocalId, labels->at(i) / Body);
        if (trace)
        {
          auto t_it =
            label_idx.find(std::string((term / Lhs)->location().view()));
          auto f_it =
            label_idx.find(std::string((term / Rhs)->location().view()));

          auto make_clone = [&]() {
            TypeEnv c;
            for (auto& [loc, info] : env)
              c[loc] = {clone(info.type), info.is_fixed, info.call_node};
            return c;
          };
          auto update_branch_exit = [&](size_t succ_idx, TypeEnv&& next_env) {
            auto key = std::make_pair(i, succ_idx);
            auto it = branch_exits.find(key);
            if (it != branch_exits.end() && same_type_env(it->second, next_env))
              return;
            branch_exits[key] = std::move(next_env);
            forward_out_changed = true;
          };

          if (!trace->negated)
          {
            if (t_it != label_idx.end())
            {
              auto ne = make_clone();
              ne[trace->src->location()] = {clone(trace->type), true, {}};
              update_branch_exit(t_it->second, std::move(ne));
            }
            if (f_it != label_idx.end())
            {
              auto ne = make_clone();
              auto src_it = ne.find(trace->src->location());
              if (src_it != ne.end())
              {
                auto excluded =
                  exclude_tested_type(top, src_it->second.type, trace->type);
                if (excluded)
                  src_it->second = {std::move(excluded), true, {}};
              }
              update_branch_exit(f_it->second, std::move(ne));
            }
          }
          else
          {
            if (f_it != label_idx.end())
            {
              auto ne = make_clone();
              ne[trace->src->location()] = {clone(trace->type), true, {}};
              update_branch_exit(f_it->second, std::move(ne));
            }
            if (t_it != label_idx.end())
            {
              auto ne = make_clone();
              auto src_it = ne.find(trace->src->location());
              if (src_it != ne.end())
              {
                auto excluded =
                  exclude_tested_type(top, src_it->second.type, trace->type);
                if (excluded)
                  src_it->second = {std::move(excluded), true, {}};
              }
              update_branch_exit(t_it->second, std::move(ne));
            }
          }
        }
      }

      // Update forward exit env. Re-queue successors if forward changed.
      bool exit_changed = !same_type_env(exit_envs[i], env);
      if (exit_changed)
      {
        exit_envs[i] = env;
        forward_out_changed = true;
      }
      else
      {
        exit_envs[i] = env;
      }
      if (forward_out_changed)
      {
        if (active_infer_profile != nullptr)
          active_infer_profile->requeue_succ += succ[i].size();
        for (auto s : succ[i])
          worklist.insert(s);
      }

      // Body-local backward refinement can improve earlier statements in the
      // same label (for example, a later comparison refining an earlier
      // CallDyn result), so re-run this label until its local backward facts
      // stabilize.
      if (label_changes.backward)
      {
        if (active_infer_profile != nullptr)
          active_infer_profile->requeue_self_bwd++;
        worklist.insert(i);
      }

      // Propagate backward constraints from successors.
      bool bwd_changed = label_changes.backward;
      for (auto s : succ[i])
      {
        for (auto& [loc, info] : bwd_envs[s])
        {
          if (
            (label_live_in[i].count(loc) == 0) &&
            (label_defs[i].count(loc) == 0))
            continue;
          auto note_succ_bwd_change = [&]() { bwd_changed = true; };
          auto bit = bwd_envs[i].find(loc);
          if (bit == bwd_envs[i].end())
          {
            bwd_envs[i][loc] = {clone(info.type), info.is_fixed, {}};
            note_succ_bwd_change();
          }
          else
          {
            if (same_type_tree(bit->second.type, info.type))
            {
              bool fixed_changed = !bit->second.is_fixed && info.is_fixed;
              bit->second.is_fixed = bit->second.is_fixed || info.is_fixed;
              if (fixed_changed)
                note_succ_bwd_change();
              continue;
            }
            if (bit->second.is_fixed && !info.is_fixed)
              continue;
            if (!bit->second.is_fixed && info.is_fixed)
            {
              bit->second.type = clone(info.type);
              bit->second.is_fixed = true;
              note_succ_bwd_change();
              continue;
            }
            auto m = merge_type(bit->second.type, info.type, top);
            if (m)
            {
              bit->second.type = m;
              note_succ_bwd_change();
            }
          }
        }
      }

      // Re-queue predecessors if backward changed.
      if (bwd_changed)
      {
        if (active_infer_profile != nullptr)
          active_infer_profile->requeue_pred_bwd += pred[i].size();
        for (auto p : pred[i])
          worklist.insert(p);
      }

      prior_transfer_epochs[i] = transfer_epoch;
    }

    profile.worklist_iterations = wl_iters;

    for (size_t i = 0; i < n; i++)
    {
      if (active_infer_profile != nullptr)
        active_infer_profile->dependency_cascade_calls++;
      InferScopedTimer cascade_timer(
        (active_infer_profile != nullptr) ?
          &active_infer_profile->dependency_cascade_time :
          nullptr);
      run_dependency_cascade(
        exit_envs[i], labels->at(i) / Body, top, lookup_stmts);
    }

    // ===== Finalization =====

    // 1. Consts: write inferred types to AST.
    for (size_t i = 0; i < n; i++)
    {
      struct FinalTupleInfo
      {
        size_t size;
        bool is_array_lit;
      };

      auto body = labels->at(i) / Body;
      std::map<Location, FinalTupleInfo> final_tuple_info;
      std::map<Location, std::pair<Location, size_t>> final_tuple_refs;
      std::map<Location, std::vector<Location>> final_tuple_values;
      std::map<Location, Node> final_newarray_types;

      for (auto& stmt : *body)
      {
        if (stmt == NewArrayConst)
        {
          auto loc = (stmt / LocalId)->location();
          auto size = from_chars_sep_v<size_t>(stmt / Rhs);
          bool is_array_lit =
            loc.view().find("array") != std::string_view::npos;
          final_tuple_info.emplace(loc, FinalTupleInfo{size, is_array_lit});
          final_tuple_values.emplace(loc, std::vector<Location>(size));
        }
        else if (stmt == ArrayRefConst)
        {
          auto arg_loc = ((stmt / Arg) / Rhs)->location();
          auto info = final_tuple_info.find(arg_loc);
          if (info == final_tuple_info.end())
            continue;
          auto index = from_chars_sep_v<size_t>(stmt / Rhs);
          if (index < info->second.size)
            final_tuple_refs[(stmt / LocalId)->location()] = {arg_loc, index};
        }
        else if (stmt == Store)
        {
          auto ref_it = final_tuple_refs.find((stmt / Rhs)->location());
          if (ref_it == final_tuple_refs.end())
            continue;
          auto& [tuple_loc, index] = ref_it->second;
          auto values_it = final_tuple_values.find(tuple_loc);
          if (
            values_it != final_tuple_values.end() &&
            index < values_it->second.size())
            values_it->second[index] = ((stmt / Arg) / Rhs)->location();
        }
      }

      for (auto& [tuple_loc, info] : final_tuple_info)
      {
        auto values_it = final_tuple_values.find(tuple_loc);
        if (values_it == final_tuple_values.end())
          continue;

        if (info.is_array_lit)
        {
          Node common;
          bool uniform = true;
          for (auto& value_loc : values_it->second)
          {
            if (value_loc.view().empty())
            {
              uniform = false;
              break;
            }
            auto env_it = exit_envs[i].find(value_loc);
            if (env_it == exit_envs[i].end())
            {
              uniform = false;
              break;
            }
            auto prim = extract_primitive(env_it->second.type);
            if (!prim)
            {
              uniform = false;
              break;
            }
            if (!common)
              common = clone(env_it->second.type);
            else if (
              env_it->second.type->front()->type() != common->front()->type())
            {
              uniform = false;
              break;
            }
          }

          if (uniform && common && info.size > 0)
            final_newarray_types[tuple_loc] = clone(common);
          continue;
        }

        bool complete = true;
        std::vector<Node> elems;
        elems.reserve(info.size);
        for (auto& value_loc : values_it->second)
        {
          if (value_loc.view().empty())
          {
            complete = false;
            break;
          }
          auto env_it = exit_envs[i].find(value_loc);
          if (env_it == exit_envs[i].end())
          {
            complete = false;
            break;
          }
          elems.push_back(clone(env_it->second.type));
        }

        if (!complete || elems.empty())
          continue;

        if (elems.size() == 1)
        {
          final_newarray_types[tuple_loc] = Type
            << clone(elems.front()->front());
        }
        else
        {
          Node tup = TupleType;
          for (auto& elem : elems)
            tup << clone(elem->front());
          final_newarray_types[tuple_loc] = Type << tup;
        }
      }

      auto it = body->begin();
      while (it != body->end())
      {
        // 2. TypeAssertion: remove.
        if (*it == TypeAssertion)
        {
          it = body->erase(it, std::next(it));
          continue;
        }

        if (*it == Const)
        {
          auto dst = (*it)->front();
          auto loc = dst->location();
          auto env_it = exit_envs[i].find(loc);
          Node final_type;
          if (
            env_it != exit_envs[i].end() &&
            !is_default_type(env_it->second.type))
            final_type = env_it->second.type;
          if (!final_type)
          {
            auto bwd_it = bwd_envs[i].find(loc);
            if (
              bwd_it != bwd_envs[i].end() &&
              !is_default_type(bwd_it->second.type))
              final_type = bwd_it->second.type;
          }
          Node final_prim;
          if (final_type)
            final_prim = extract_primitive(final_type);

          if (final_prim)
          {
            if ((*it)->size() == 3)
            {
              auto old_type = (*it)->at(1);
              if (old_type->type() != final_prim->type())
                (*it)->replace(old_type, final_prim->type());
            }
            else if ((*it)->size() == 2)
            {
              auto lit = (*it)->back();
              (*it)->erase((*it)->begin(), (*it)->end());
              *it << dst << final_prim->type() << lit;
            }
          }
          else if ((*it)->size() == 2)
          {
            auto lit = (*it)->back();
            auto type_tok = default_literal_type(lit);
            (*it)->erase((*it)->begin(), (*it)->end());
            *it << dst << type_tok << lit;
          }
        }
        // 3. NewArrayConst: update Type from env.
        else if (*it == NewArrayConst)
        {
          auto nloc = ((*it) / LocalId)->location();
          auto final_it = final_newarray_types.find(nloc);
          if (final_it != final_newarray_types.end())
          {
            (*it)->replace((*it) / Type, clone(final_it->second));
          }

          auto env_it = exit_envs[i].find(nloc);
          if (env_it != exit_envs[i].end())
          {
            auto inner = env_it->second.type->front();
            if (inner == TupleType)
              (*it)->replace((*it) / Type, clone(env_it->second.type));
            else
            {
              auto prim = extract_primitive(env_it->second.type);
              if (
                prim &&
                !Subtype.invariant(top, (*it) / Type, env_it->second.type))
                (*it)->replace((*it) / Type, clone(env_it->second.type));
            }
          }
        }

        ++it;
      }
    }

    // 5. TypeVar back-prop.
    {
      bool tv_changed = true;
      while (tv_changed)
      {
        tv_changed = false;
        for (auto& [dst_loc, src_loc] : typevar_aliases)
        {
          for (auto& ee : exit_envs)
          {
            auto d = ee.find(dst_loc);
            auto s = ee.find(src_loc);
            if (d == ee.end() || s == ee.end())
              continue;
            bool dtv = d->second.type->front() == TypeVar;
            bool stv = s->second.type->front() == TypeVar;
            if (!dtv && stv)
            {
              s->second.type = clone(d->second.type);
              s->second.is_fixed = true;
              tv_changed = true;
            }
            else if (dtv && !stv)
            {
              d->second.type = clone(s->second.type);
              tv_changed = true;
            }
          }
        }
      }
    }

    // 6. Update params and fields.
    auto parent_cls = node->parent(ClassDef);
    for (auto& pd : *(node / Params))
    {
      auto type = pd / Type;
      if (type->front() != TypeVar)
        continue;
      auto ident = pd / Ident;
      for (auto& ee : exit_envs)
      {
        auto it = ee.find(ident->location());
        if (it != ee.end() && it->second.type->front() != TypeVar)
        {
          pd->replace(type, clone(it->second.type));
          if (parent_cls)
          {
            for (auto& child : *(parent_cls / ClassBody))
            {
              if (child != FieldDef)
                continue;
              if (
                (child / Ident)->location().view() != ident->location().view())
                continue;
              if (contains_typevar(child / Type))
                child->replace(child / Type, clone(it->second.type));
              break;
            }
          }
          break;
        }
      }
    }

    if (parent_cls)
    {
      for (auto& child : *(parent_cls / ClassBody))
      {
        if (child != FieldDef || (child / Type)->front() != TypeVar)
          continue;
        auto fname = (child / Ident)->location();
        for (auto& ee : exit_envs)
        {
          auto it = ee.find(fname);
          if (it != ee.end() && it->second.type->front() != TypeVar)
          {
            child->replace(child / Type, clone(it->second.type));
            break;
          }
        }
      }
    }

    // 4. Return type inference.
    bool in_generic = (node / TypeParams)->size() > 0 ||
      (node->parent({ClassDef}) != nullptr &&
       (node->parent({ClassDef}) / TypeParams)->size() > 0);

    auto func_ret = node / Type;
    if (func_ret->front() == TypeVar)
    {
      SequentCtx ctx{top, {}, {}};
      Nodes ret_types;
      bool unresolved = false;

      // If any parameter is still TypeVar, the function hasn't been fully
      // resolved yet. Treat as unresolved so the deferred retry can improve.
      for (auto& pd : *(node / Params))
        if ((pd / Type)->front() == TypeVar)
          unresolved = true;

      if (in_generic)
      {
        for (auto& lbl : *labels)
          for (auto& stmt : *(lbl / Body))
            if (stmt->in({CallDyn, TryCallDyn}))
            {
              auto loc = (stmt / LocalId)->location();
              bool found = false;
              for (auto& ee : exit_envs)
                if (ee.find(loc) != ee.end())
                {
                  found = true;
                  break;
                }
              if (!found)
                unresolved = true;
            }
      }

      for (size_t i = 0; i < n; i++)
      {
        auto term = labels->at(i) / Return;
        if (term != Return)
          continue;
        auto ret_loc = (term / LocalId)->location();
        auto eit = exit_envs[i].find(ret_loc);
        if (eit == exit_envs[i].end() || eit->second.type->front() == TypeVar)
        {
          if (in_generic)
            unresolved = true;
          continue;
        }
        bool covered = false;
        for (auto& rt : ret_types)
          if (Subtype(ctx, eit->second.type, rt))
          {
            covered = true;
            break;
          }
        if (!covered)
          ret_types.push_back(clone(eit->second.type));
      }

      if (!unresolved)
      {
        if (ret_types.size() == 1)
          node->replace(func_ret, ret_types.front());
        else if (ret_types.size() > 1)
        {
          Node u = Union;
          for (auto& rt : ret_types)
            u << clone(rt->front());
          node->replace(func_ret, Type << u);
        }
        else
        {
          bool all_nonlocal = true;
          for (auto& lbl : *labels)
          {
            auto term = lbl / Return;
            if (term->in({Jump, Cond}))
              continue;
            if (term != Raise)
            {
              all_nonlocal = false;
              break;
            }
          }
          if (all_nonlocal)
            node->replace(
              func_ret,
              Type
                << (TypeName
                    << (NameElement << (Ident ^ "_builtin") << TypeArgs)
                    << (NameElement << (Ident ^ "none") << TypeArgs)));
        }
      }
    }

    // Error checking.
    if (check_errors && !in_generic)
    {
      for (auto& pd : *(node / Params))
      {
        if ((pd / Type)->front() == TypeVar)
        {
          auto pending = deferred_param_errors.find((pd / Ident)->location());
          if (pending != deferred_param_errors.end())
            node->parent()->replace(
              node, err(pending->second.site, pending->second.msg));
          else
            node->parent()->replace(
              node, err(pd / Ident, "Cannot infer type of parameter"));
          return false;
        }
      }
      func_ret = node / Type;
      if (func_ret->front() == TypeVar)
      {
        node->parent()->replace(
          node, err(node / Ident, "Cannot infer return type of function"));
        return false;
      }
    }
    return true;
  }

  // ===== has_typevar (simplified) =====

  static bool has_typevar(Node func)
  {
    for (auto& pd : *(func / Params))
      if ((pd / Type)->front() == TypeVar)
        return true;
    auto ret = (func / Type)->front();
    if (ret == TypeVar || ret->in({DefaultInt, DefaultFloat}))
      return true;
    if (ret == Union)
      for (auto& child : *ret)
        if (child->in({DefaultInt, DefaultFloat}))
          return true;
    return false;
  }

  // ===== Pass definition =====

  PassDef infer()
  {
    PassDef p{"infer", wfPassInfer, dir::once, {}};

    p.post([](auto top) {
      Nodes deferred;

      lambda_returns_omitted.clear();
      deferred_param_errors.clear();
      top->traverse([&](auto node) {
        if (
          node == Function && is_lambda_function(node) &&
          (node / Type)->front() == TypeVar)
        {
          lambda_returns_omitted.insert(node.get());
        }
        return true;
      });

      top->traverse([&](auto node) {
        if (node != Function)
          return node == Top || node == ClassDef || node == ClassBody ||
            node == Lib || node == Symbols;
        process_function(node, top, false);
        if (has_typevar(node))
          deferred.push_back(node);
        return false;
      });

      size_t prev = deferred.size();
      for (size_t iter = 0; iter < deferred.size(); iter++)
      {
        for (auto& func : deferred)
        {
          if (has_typevar(func))
            func->replace(func / Type, make_type());
          process_function(func, top, false);
        }
        size_t count = 0;
        for (auto& func : deferred)
          if (has_typevar(func))
            count++;
        if (count == 0 || count >= prev)
          break;
        prev = count;
      }

      top->traverse([&](auto node) {
        if (node != Function)
          return node == Top || node == ClassDef || node == ClassBody ||
            node == Lib || node == Symbols;
        process_function(node, top, false);
        return false;
      });

      for (auto& func : deferred)
        if (has_typevar(func))
          process_function(func, top, true);

      // Sweep: DefaultInt → u64, DefaultFloat → f64.
      top->traverse([](Node& node) {
        if (node->in({DefaultInt, DefaultFloat}))
        {
          auto parent = node->parent();
          bool is_int = (node == DefaultInt);
          if (parent == Const)
            parent->replace(node, is_int ? Node{U64} : Node{F64});
          else
            parent->replace(
              node,
              is_int ? primitive_type(U64)->front() :
                       primitive_type(F64)->front());
          return false;
        }
        return true;
      });

      return 0;
    });

    return p;
  }
}
