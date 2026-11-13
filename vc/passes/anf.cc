#include "../lang.h"

namespace vc
{
  // Sythetic locations.
  inline const auto l_arraylit = Location("array");
  inline const auto l_cond = Location("cond");
  inline const auto l_body = Location("body");
  inline const auto l_join = Location("join");

  size_t field_count(Node classbody)
  {
    assert(classbody == ClassBody);
    size_t count = 0;

    for (auto& child : *classbody)
    {
      if (child == FieldDef)
        ++count;
    }

    return count;
  }

  const auto CallPat = T(Call)[Call] << (T(FuncName)[FuncName] * T(Args)[Args]);

  const auto CallDynPat = T(CallDyn)[CallDyn]
    << (T(LocalId)[LocalId] * T(Ident, SymbolId)[Ident] *
        T(TypeArgs)[TypeArgs] * T(Args)[Args]);

  const auto TryCallDynPat = T(TryCallDyn)[TryCallDyn]
    << (T(LocalId)[LocalId] * T(Ident, SymbolId)[Ident] *
        T(TypeArgs)[TypeArgs] * T(Args)[Args]);

  Node make_call(Match& _, bool lvalue, bool ref)
  {
    // lvalue is true if the call appears on the LHS of an assignment.
    // ref is true if lvalue is true, or if the call is in a Ref node.
    auto id = _.fresh(l_local);
    auto res = lvalue ? (Ref << (LocalId ^ id)) : (LocalId ^ id);
    return Seq << (Lift << Body
                        << (Call << (LocalId ^ id) << (ref ? Lhs : Rhs)
                                 << _(FuncName) << (Args << *_[Args])))
               << res;
  }

  Node make_calldyn(Match& _, bool lvalue, bool ref)
  {
    // lvalue is true if the call appears on the LHS of an assignment.
    // ref is true if lvalue is true, or if the call is in a Ref node.
    auto fn = _.fresh(l_local);
    auto id = _.fresh(l_local);
    auto res = lvalue ? (Ref << (LocalId ^ id)) : (LocalId ^ id);
    auto arity = _(Args) ? _(Args)->size() + 1 : 1;
    return Seq << (Lift << Body
                        << (Lookup << (LocalId ^ fn) << (LocalId ^ _(LocalId))
                                   << (ref ? Lhs : Rhs) << _(Ident)
                                   << _(TypeArgs)
                                   << (Int ^ std::to_string(arity))))
               << (Lift << Body
                        << (CallDyn
                            << (LocalId ^ id) << (LocalId ^ fn)
                            << (Args << (LocalId ^ _(LocalId)) << *_[Args])))
               << res;
  }

  Node make_trycalldyn(Match& _)
  {
    auto fn = _.fresh(l_local);
    auto id = _.fresh(l_local);
    auto arity = _(Args) ? _(Args)->size() + 1 : 1;
    return Seq << (Lift << Body
                        << (Lookup << (LocalId ^ fn) << (LocalId ^ _(LocalId))
                                   << Rhs << _(Ident) << _(TypeArgs)
                                   << (Int ^ std::to_string(arity))))
               << (Lift << Body
                        << (TryCallDyn
                            << (LocalId ^ id) << (LocalId ^ fn)
                            << (Args << (LocalId ^ _(LocalId)) << *_[Args])))
               << (LocalId ^ id);
  }

  PassDef anf()
  {
    PassDef p{
      "anf",
      wfPassANF,
      dir::topdown,
      {
        // Turn an initial function body into a label.
        In(Function) * T(Body)[Body] >>
          [](Match& _) {
            return Labels << (Label << (LabelId ^ "start") << _(Body));
          },

        // New.
        In(Expr) * T(New) << T(NewArgs)[NewArgs] >>
          [](Match& _) {
            auto args = _(NewArgs);
            auto id = _.fresh(l_local);
            return Seq << (Lift << Body
                                << (New << (LocalId ^ id)
                                        << make_selftype(args, true) << args))
                       << (LocalId ^ id);
          },

        // Stack (block allocation).
        In(Expr) * T(Stack) << (T(Type)[Type] * T(NewArgs)[NewArgs]) >>
          [](Match& _) {
            auto args = _(NewArgs);
            auto type = _(Type);
            auto id = _.fresh(l_local);
            return Seq << (Lift << Body
                                << (Stack << (LocalId ^ id) << type << args))
                       << (LocalId ^ id);
          },

        // New array.
        In(Expr) * T(NewArray)[NewArray]
            << (T(Type)[Type] *
                (T(Args) << (T(Arg) << (T(ArgCopy) * T(LocalId)[LocalId])))) >>
          [](Match& _) {
            auto type = _(Type);
            auto size = _(LocalId);
            auto id = _.fresh(l_local);

            return Seq << (Lift << Body
                                << (NewArray << (LocalId ^ id) << type << size))
                       << (LocalId ^ id);
          },

        // Array reference.
        In(Expr) * (T(ArrayRef)[ArrayRef] << T(Args)[Args]) >>
          [](Match& _) {
            auto args = _(Args);
            assert(args->size() == 2);
            auto id = _.fresh(l_local);
            return Seq << (Lift
                           << Body
                           << (ArrayRef << (LocalId ^ id)
                                        << (Arg << ArgCopy << args->front())
                                        << args->back()))
                       << (LocalId ^ id);
          },

        // Tuple creation.
        In(Expr) * T(Tuple)[Tuple] << (T(LocalId)++ * End) >>
          [](Match& _) {
            // Allocate an array[any] of the right size.
            auto tuple = _(Tuple);
            auto id = _.fresh(l_local);
            auto seq = Seq
              << (Lift << Body
                       << (NewArrayConst
                           << (LocalId ^ id) << type_any()
                           << (Int ^ std::to_string(tuple->size()))));

            // Copy the elements into the array.
            size_t idx = 0;
            for (auto& elem : *tuple)
            {
              auto ref = _.fresh(l_local);
              auto prev = _.fresh(l_local);
              seq << (Lift << Body
                           << (ArrayRefConst
                               << (LocalId ^ ref)
                               << (Arg << ArgCopy << (LocalId ^ id))
                               << (Int ^ std::to_string(idx++))))
                  << (Lift << Body
                           << (Store << (LocalId ^ prev) << (LocalId ^ ref)
                                     << (Arg << ArgCopy << elem)));
            }

            return seq << (LocalId ^ id);
          },

        // Array literal creation.
        In(Expr) * T(ArrayLit)[ArrayLit] << (T(LocalId)++ * End) >>
          [](Match& _) {
            // Allocate an array[any] of the right size.
            // Uses l_arraylit prefix so infer can distinguish from tuples.
            auto arr = _(ArrayLit);
            auto id = _.fresh(l_arraylit);
            auto seq = Seq
              << (Lift << Body
                       << (NewArrayConst
                           << (LocalId ^ id) << type_any()
                           << (Int ^ std::to_string(arr->size()))));

            // Copy the elements into the array.
            size_t idx = 0;
            for (auto& elem : *arr)
            {
              auto ref = _.fresh(l_local);
              auto prev = _.fresh(l_local);
              seq << (Lift << Body
                           << (ArrayRefConst
                               << (LocalId ^ ref)
                               << (Arg << ArgCopy << (LocalId ^ id))
                               << (Int ^ std::to_string(idx++))))
                  << (Lift << Body
                           << (Store << (LocalId ^ prev) << (LocalId ^ ref)
                                     << (Arg << ArgCopy << elem)));
            }

            return seq << (LocalId ^ id);
          },

        // Field reference.
        In(Expr) * T(FieldRef) << (T(LocalId)[LocalId] * T(FieldId)[FieldId]) >>
          [](Match& _) {
            auto id = _.fresh(l_local);
            return Seq << (Lift << Body
                                << (FieldRef << (LocalId ^ id)
                                             << (Arg << ArgCopy << _(LocalId))
                                             << _(FieldId)))
                       << (LocalId ^ id);
          },

        // L-values.
        In(Expr) * T(Equals) << (T(Expr)[Lhs] * T(Expr)[Rhs]) >>
          [](Match& _) { return Equals << (Lhs << *_[Lhs]) << _(Rhs); },

        In(Lhs, TupleLHS) * T(Expr)[Expr] >>
          [](Match& _) { return Lhs << *_[Expr]; },

        In(Lhs) * T(Tuple)[Tuple] >>
          [](Match& _) { return TupleLHS << *_[Tuple]; },

        // Discard assignment: _ = expr.
        T(Equals) << (T(DontCare) * T(LocalId)[Rhs]) >>
          [](Match& _) {
            auto id = _.fresh(l_local);
            return Seq << (Lift << Body << (Copy << (LocalId ^ id) << _(Rhs)))
                       << (LocalId ^ id);
          },

        // Assignment.
        T(Equals) << (T(LocalId)[Lhs] * T(LocalId)[Rhs]) >>
          [](Match& _) {
            auto id = _.fresh(l_local);
            return Seq << (Lift
                           << Body
                           << (Copy << (LocalId ^ id) << (LocalId ^ _(Lhs))))
                       << (Lift << Body << (Copy << _(Lhs) << _(Rhs)))
                       << (LocalId ^ id);
          },

        // Store.
        T(Equals) << ((T(Ref) << T(LocalId)[Lhs]) * T(LocalId)[Rhs]) >>
          [](Match& _) {
            auto id = _.fresh(l_local);
            return Seq << (Lift << Body
                                << (Store << (LocalId ^ id) << _(Lhs)
                                          << (Arg << ArgCopy << _(Rhs))))
                       << (LocalId ^ id);
          },

        // Destructuring assignment.
        T(Equals)
            << ((T(TupleLHS)[Lhs] << (T(
                   TupleLHS,
                   LocalId,
                   Let,
                   DontCare,
                   SplatLet,
                   SplatDontCare)++)) *
                T(LocalId)[Rhs]) >>
          [](Match& _) {
            auto lhs = _(Lhs);
            auto rhs_loc = _(Rhs);

            // Find splat position (if any) and validate at most one.
            int splat_pos = -1;
            for (size_t i = 0; i < lhs->size(); i++)
            {
              auto& child = lhs->at(i);
              if (child->type() == SplatLet || child->type() == SplatDontCare)
              {
                if (splat_pos >= 0)
                  return err(child, "Only one '...' allowed in destructuring");
                splat_pos = static_cast<int>(i);
              }
            }

            // Without a splat, require exact arity (checked post-reification
            // when the tuple type is known). Emit the simple sequential code.
            Node seq = Seq;
            Node tuple = Tuple;

            if (splat_pos < 0)
            {
              size_t idx = 0;
              for (auto& l : *lhs)
              {
                auto ref = _.fresh(l_local);
                auto val = _.fresh(l_local);
                seq << (Lift << Body
                             << (ArrayRefConst
                                 << (LocalId ^ ref)
                                 << (Arg << ArgCopy << (LocalId ^ rhs_loc))
                                 << (Int ^ std::to_string(idx++))))
                    << (Lift << Body
                             << (Load << (LocalId ^ val) << (LocalId ^ ref)));

                if (l->type() == DontCare)
                {
                  tuple << (LocalId ^ val);
                }
                else if (l->type() == Let)
                {
                  auto name = (l / Ident)->location();
                  seq
                    << (Lift << Body
                             << (Copy << (LocalId ^ name) << (LocalId ^ val)));
                  tuple << (LocalId ^ val);
                }
                else
                {
                  tuple << (Equals << l << (LocalId ^ val));
                }
              }

              return seq << (Expr << tuple);
            }

            // With a splat: split into before / splat / after.
            size_t before_count = static_cast<size_t>(splat_pos);
            size_t after_count = lhs->size() - before_count - 1;

            // Before-splat elements: ArrayRefConst with concrete indices.
            for (size_t i = 0; i < before_count; i++)
            {
              auto& l = lhs->at(i);
              auto ref = _.fresh(l_local);
              auto val = _.fresh(l_local);
              seq << (Lift << Body
                           << (ArrayRefConst
                               << (LocalId ^ ref)
                               << (Arg << ArgCopy << (LocalId ^ rhs_loc))
                               << (Int ^ std::to_string(i))))
                  << (Lift << Body
                           << (Load << (LocalId ^ val) << (LocalId ^ ref)));

              if (l->type() == DontCare)
              {
                tuple << (LocalId ^ val);
              }
              else if (l->type() == Let)
              {
                auto name = (l / Ident)->location();
                seq
                  << (Lift << Body
                           << (Copy << (LocalId ^ name) << (LocalId ^ val)));
                tuple << (LocalId ^ val);
              }
              else
              {
                tuple << (Equals << l << (LocalId ^ val));
              }
            }

            // Splat element: SplatOp creates the splat value.
            auto& splat_node = lhs->at(splat_pos);
            if (splat_node->type() == SplatLet)
            {
              auto name = (splat_node / Ident)->location();
              seq
                << (Lift << Body
                         << (SplatOp << (LocalId ^ name)
                                     << (Arg << ArgCopy << (LocalId ^ rhs_loc))
                                     << (Int ^ std::to_string(before_count))
                                     << (Int ^ std::to_string(after_count))));
              tuple << (LocalId ^ name);
            }
            else
            {
              // SplatDontCare: still emit SplatOp so type checking
              // can validate arity, but the result is unused.
              auto splat_id = _.fresh(l_local);
              seq
                << (Lift << Body
                         << (SplatOp << (LocalId ^ splat_id)
                                     << (Arg << ArgCopy << (LocalId ^ rhs_loc))
                                     << (Int ^ std::to_string(before_count))
                                     << (Int ^ std::to_string(after_count))));
              tuple << (LocalId ^ splat_id);
            }

            // After-splat elements: ArrayRefFromEnd with 1-based from-end
            // index.
            for (size_t i = 0; i < after_count; i++)
            {
              auto& l = lhs->at(before_count + 1 + i);
              auto ref = _.fresh(l_local);
              auto val = _.fresh(l_local);
              // from-end index: after_count - i (last element = 1)
              size_t from_end = after_count - i;
              seq << (Lift << Body
                           << (ArrayRefFromEnd
                               << (LocalId ^ ref)
                               << (Arg << ArgCopy << (LocalId ^ rhs_loc))
                               << (Int ^ std::to_string(from_end))))
                  << (Lift << Body
                           << (Load << (LocalId ^ val) << (LocalId ^ ref)));

              if (l->type() == DontCare)
              {
                tuple << (LocalId ^ val);
              }
              else if (l->type() == Let)
              {
                auto name = (l / Ident)->location();
                seq
                  << (Lift << Body
                           << (Copy << (LocalId ^ name) << (LocalId ^ val)));
                tuple << (LocalId ^ val);
              }
              else
              {
                tuple << (Equals << l << (LocalId ^ val));
              }
            }

            return seq << (Expr << tuple);
          },

        // Invalid l-values.
        In(Lhs) * T(FuncName, If, Else, While, When)[Lhs] >>
          [](Match& _) { return err(_(Lhs), "Can't assign to this"); },

        // Error on remaining DontCare (not on LHS of assignment).
        !In(Equals, TupleLHS) * T(DontCare)[DontCare] >>
          [](Match& _) {
            return err(
              _(DontCare),
              "'_' can only be used on the left side of an assignment");
          },

        // Error on splat outside destructuring.
        !In(TupleLHS) * T(SplatLet)[Lhs] >>
          [](Match& _) {
            return err(_(Lhs), "'...' can only be used in destructuring");
          },

        !In(TupleLHS) * T(SplatDontCare)[Lhs] >>
          [](Match& _) {
            return err(_(Lhs), "'...' can only be used in destructuring");
          },

        // If expression.
        In(Expr) * T(If)[If] >>
          [](Match& _) {
            auto id = _.fresh(l_local);
            return Seq << (Lift << Body << (_(If) << (LocalId ^ id)))
                       << (Var << (Ident ^ id));
          },

        // If body.
        In(Body) * T(If)
            << (T(Expr)[Cond] * (T(Block) << T(Body)[Body]) *
                T(LocalId)[LocalId]) >>
          [](Match& _) {
            auto body = _.fresh(l_body);
            auto join = _.fresh(l_join);
            return Seq << make_nomatch(_(LocalId))
                       << (Cond << _(Cond) << (LabelId ^ body)
                                << (LabelId ^ join))
                       << (Label << (LabelId ^ body)
                                 << (_(Body) << (Copy << (LocalId ^ _(LocalId)))
                                             << (Jump << (LabelId ^ join))))
                       << (Label << (LabelId ^ join) << Body);
          },

        // While expression.
        In(Expr) * T(While)[While] >>
          [](Match& _) {
            auto id = _.fresh(l_local);
            return Seq << (Lift << Body << (_(While) << (LocalId ^ id)))
                       << (Var << (Ident ^ id));
          },

        // While body.
        In(Body) * T(While)
            << (T(Expr)[Cond] * (T(Block) << T(Body)[Body]) *
                T(LocalId)[LocalId]) >>
          [](Match& _) {
            auto cond = _.fresh(l_cond);
            auto body = _.fresh(l_body);
            auto join = _.fresh(l_join);

            _(Body)->traverse([&](Node& node) {
              if (node->in({While, Error}))
                return false;
              else if (node == Break)
                node << (LocalId ^ _(LocalId)) << (LabelId ^ join);
              else if (node == Continue)
                node << (LocalId ^ _(LocalId)) << (LabelId ^ cond);
              return true;
            });

            return Seq << make_nomatch(_(LocalId)) << (Jump << (LabelId ^ cond))
                       << (Label << (LabelId ^ cond)
                                 << (Body
                                     << (Cond << _(Cond) << (LabelId ^ body)
                                              << (LabelId ^ join))))
                       << (Label << (LabelId ^ body)
                                 << (_(Body) << (Copy << (LocalId ^ _(LocalId)))
                                             << (Jump << (LabelId ^ cond))))
                       << (Label << (LabelId ^ join) << Body);
          },

        // Else expression.
        In(Expr) * T(Else) << (T(LocalId)[Lhs] * (T(Block)[Block])) >>
          [](Match& _) {
            auto id = _.fresh(l_local);
            return Seq << (Lift
                           << Body
                           << (Else << _(Lhs) << _(Block) << (LocalId ^ id)))
                       << (Var << (Ident ^ id));
          },

        // Else body.
        In(Body) * T(Else)
            << (T(LocalId)[Lhs] * (T(Block) << T(Body)[Body]) *
                T(LocalId)[LocalId]) >>
          [](Match& _) {
            auto id = _.fresh(l_local);
            auto ok = _.fresh(l_body);
            auto else_lbl = _.fresh(l_body);
            auto join = _.fresh(l_join);
            return Seq << (Typetest << (LocalId ^ id) << (LocalId ^ _(Lhs))
                                    << type_nomatch())
                       << (Cond << (LocalId ^ id) << (LabelId ^ else_lbl)
                                << (LabelId ^ ok))
                       << (Label << (LabelId ^ ok)
                                 << (Body << (Copy << (LocalId ^ _(LocalId))
                                                   << (LocalId ^ _(Lhs)))
                                          << (Jump << (LabelId ^ join))))
                       << (Label << (LabelId ^ else_lbl)
                                 << (_(Body) << (Copy << (LocalId ^ _(LocalId)))
                                             << (Jump << (LabelId ^ join))))
                       << (Label << (LabelId ^ join) << Body);
          },

        // Break, continue.
        In(Body) * T(Break, Continue)[Break]
            << (T(LocalId)[Rhs] * T(LocalId)[Lhs] * T(LabelId)[LabelId]) >>
          [](Match& _) {
            return Seq << (Copy << _(Lhs) << _(Rhs)) << (Jump << _(LabelId));
          },

        // Continuation label.
        In(Body) * (T(Label) << (T(LabelId)[LabelId] * T(Body)[Body])) *
            (!T(Label) * (!T(Label))++)[Continue] * End >>
          [](Match& _) {
            return Label << (LabelId ^ _(LabelId)) << (_(Body) << _[Continue]);
          },

        // Lift labels without reversing ordering.
        In(Labels) * T(Label)[Label] >> [](Match& _) -> Node {
          auto body = _(Label) / Body;
          Nodes labels;

          auto it = std::remove_if(body->begin(), body->end(), [&](Node& n) {
            if (n == Label)
            {
              labels.push_back(n);
              return true;
            }

            return false;
          });

          if (labels.empty())
            return NoChange;

          body->erase(it, body->end());
          auto seq = Seq << _(Label);

          for (auto& l : labels)
            seq << l;

          return seq;
        },

        // When.
        In(Expr) * T(When)
            << (T(Args)[Args] * T(Type)[Type] * T(LocalId)[Rhs]) >>
          [](Match& _) {
            auto args = _(Args);
            auto lambda = _(Rhs);
            args->push_front(Arg << ArgCopy << clone(lambda));

            auto f = _.fresh(l_local);
            auto id = _.fresh(l_local);
            return Seq << (Lift
                           << Body
                           << (Lookup << (LocalId ^ f) << lambda << Rhs
                                      << (Ident ^ "apply") << TypeArgs
                                      << (Int ^ std::to_string(args->size()))))
                       << (Lift << Body
                                << (When << (LocalId ^ id) << (LocalId ^ f)
                                         << _(Args) << _(Type)))
                       << (LocalId ^ id);
          },

        // Replace Let with LocalId. If the Let has an explicit type
        // annotation (non-TypeVar), emit a TypeAssertion.
        In(Expr, Lhs) * T(Let)[Let] >>
          [](Match& _) {
            auto let = _(Let);
            auto type = let / Type;
            auto local = LocalId ^ (let / Ident);

            if (type->front() != TypeVar)
            {
              return Seq << (Lift
                             << Body
                             << (TypeAssertion << clone(local) << clone(type)))
                         << local;
            }

            return local;
          },

        // Replace a typed binder with its SSA destination.
        In(Body) *
            (T(TypedMove)[TypedMove]
             << (T(Ident)[Ident] * T(LocalId)[Rhs] * T(Type)[Type])) >>
          [](Match& _) {
            return (TypedMove ^ _(TypedMove))
              << (LocalId ^ _(Ident)) << _(Rhs) << _(Type);
          },

        // Lift variable declarations. If the Var has an explicit type
        // annotation (non-TypeVar), emit a TypeAssertion. Strip the
        // type from the Var node (Var <<= Ident after ANF).
        In(Expr, Lhs) * T(Var)[Var] >>
          [](Match& _) {
            auto var = _(Var);
            auto ident = var / Ident;
            Node assertion;

            // Synthetic Vars (from If/While desugaring) have no Type child.
            if (var->size() > 1)
            {
              auto type = var / Type;

              if (type->front() != TypeVar)
                assertion = TypeAssertion << (LocalId ^ ident) << clone(type);

              // Strip the Type child from Var.
              var->erase(std::next(var->begin()), var->end());
            }

            if (assertion)
            {
              return Seq << (Lift << Body << var) << (Lift << Body << assertion)
                         << (LocalId ^ ident);
            }

            return Seq << (Lift << Body << var) << (LocalId ^ ident);
          },

        // Lift literals.
        In(Expr) * T(True, False)[Bool] >>
          [](Match& _) {
            auto id = _.fresh(l_local);
            return Seq << (Lift << Body << (Const << (LocalId ^ id) << _(Bool)))
                       << (LocalId ^ id);
          },

        In(Expr) * T(Bin, Oct, Int, Hex, Char)[Int] >>
          [](Match& _) {
            auto id = _.fresh(l_local);
            return Seq << (Lift << Body << (Const << (LocalId ^ id) << _(Int)))
                       << (LocalId ^ id);
          },

        In(Expr) * T(Float, HexFloat)[Float] >>
          [](Match& _) {
            auto id = _.fresh(l_local);
            return Seq << (Lift << Body
                                << (Const << (LocalId ^ id) << _(Float)))
                       << (LocalId ^ id);
          },

        In(Expr) * T(String, RawString)[String] >>
          [](Match& _) {
            auto arr_id = _.fresh(l_local);
            auto str_id = _.fresh(l_local);
            auto frz_id = _.fresh(l_local);
            Node funcname = FuncName
              << (NameElement << (Ident ^ "_builtin") << TypeArgs)
              << (NameElement << (Ident ^ "string") << TypeArgs)
              << (NameElement << (Ident ^ "create") << TypeArgs);
            return Seq
              << (Lift << Body << (ConstStr << (LocalId ^ arr_id) << _(String)))
              << (Lift << Body
                       << (Call << (LocalId ^ str_id) << Rhs << funcname
                                << (Args << (LocalId ^ arr_id))))
              << (Lift << Body
                       << (Freeze << (LocalId ^ frz_id) << (LocalId ^ str_id)))
              << (LocalId ^ frz_id);
          },

        // Dynamic call.
        In(Expr) * T(Ref) << (T(Expr) << CallDynPat) >>
          [](Match& _) { return make_calldyn(_, false, true); },

        In(Expr) * CallDynPat >>
          [](Match& _) { return make_calldyn(_, false, false); },

        In(Lhs) * CallDynPat >>
          [](Match& _) { return make_calldyn(_, true, true); },

        // Try dynamic call.
        In(Expr) * TryCallDynPat >> [](Match& _) { return make_trycalldyn(_); },

        // Static call.
        In(Expr) * (T(Ref) << (T(Expr) << CallPat)) >>
          [](Match& _) { return make_call(_, false, true); },

        In(Expr) * CallPat >>
          [](Match& _) { return make_call(_, false, false); },

        In(Lhs) * CallPat >> [](Match& _) { return make_call(_, true, true); },

        // Treat all Args as ArgCopy, the IR will optimize to moves.
        In(Args) * T(LocalId)[LocalId] >>
          [](Match& _) { return Arg << ArgCopy << _(LocalId); },

        // RHS reference creation.
        In(Expr) * T(Ref) << T(LocalId)[LocalId] >>
          [](Match& _) {
            auto id = _.fresh(l_local);
            return Seq << (Lift
                           << Body
                           << (RegisterRef << (LocalId ^ id) << _(LocalId)))
                       << (LocalId ^ id);
          },

        // LHS dereference.
        In(Lhs) * T(Ref) << (T(Ref) << T(LocalId)[LocalId]) >>
          [](Match& _) {
            auto id = _.fresh(l_local);
            return Seq << (Lift << Body
                                << (Load << (LocalId ^ id) << _(LocalId)))
                       << (Ref << (LocalId ^ id));
          },

        // Hash.
        In(Expr) * T(Hash) << (T(LocalId)[LocalId]) >>
          [](Match& _) {
            auto id0 = _.fresh(l_local);
            auto id1 = _.fresh(l_local);
            return Seq << (Lift << Body
                                << (Bits << (LocalId ^ id0) << _(LocalId)))
                       << (Lift << Body
                                << (Convert << (LocalId ^ id1) << U64
                                            << (LocalId ^ id0)))
                       << (LocalId ^ id1);
          },

        // Load, from auto-RHS fields.
        In(Expr) * T(Load) << T(LocalId)[LocalId] >>
          [](Match& _) {
            auto id = _.fresh(l_local);
            return Seq << (Lift << Body
                                << (Load << (LocalId ^ id) << _(LocalId)))
                       << (LocalId ^ id);
          },

        // GetRaise.
        In(Expr) * T(GetRaise) >>
          [](Match& _) {
            auto id = _.fresh(l_local);
            return Seq << (Lift << Body << (GetRaise << (LocalId ^ id)))
                       << (LocalId ^ id);
          },

        // SetRaise.
        In(Expr) * T(SetRaise) << T(LocalId)[LocalId] >>
          [](Match& _) {
            auto id = _.fresh(l_local);
            return Seq << (Lift << Body
                                << (SetRaise << (LocalId ^ id) << _(LocalId)))
                       << (LocalId ^ id);
          },

        // Typetest expression.
        // Typetest << (LocalId * Type) produces a bool result.
        In(Expr) * T(Typetest) << (T(LocalId)[LocalId] * T(Type)[Type]) >>
          [](Match& _) {
            auto id = _.fresh(l_local);
            return Seq << (Lift << Body
                                << (Typetest << (LocalId ^ id) << _(LocalId)
                                             << _(Type)))
                       << (LocalId ^ id);
          },

        // Convert.
        In(Expr, Lhs) * T(Convert)
            << (Any[Type] *
                (T(Args) << ((
                   T(Arg) << (T(ArgCopy) * T(LocalId)[LocalId]))))) >>
          [](Match& _) {
            auto id = _.fresh(l_local);
            return Seq << (Lift << Body
                                << (Convert << (LocalId ^ id) << _(Type)
                                            << _(LocalId)))
                       << (LocalId ^ id);
          },

        // Binop.
        In(Expr, Lhs) * T(Binop)
            << (Any[MethodId] *
                (T(Args)
                 << ((T(Arg) << (T(ArgCopy) * T(LocalId)[Lhs])) *
                     (T(Arg) << (T(ArgCopy) * T(LocalId)[Rhs]))))) >>
          [](Match& _) {
            auto id = _.fresh(l_local);
            return Seq << (Lift << Body
                                << (_(MethodId)
                                    << (LocalId ^ id) << _(Lhs) << _(Rhs)))
                       << (LocalId ^ id);
          },

        // Unop.
        In(Expr, Lhs) * T(Unop)
            << (Any[MethodId] *
                (T(Args) << (T(Arg) << (T(ArgCopy) * T(LocalId)[Lhs])))) >>
          [](Match& _) {
            auto id = _.fresh(l_local);
            return Seq << (Lift << Body
                                << (_(MethodId) << (LocalId ^ id) << _(Lhs)))
                       << (LocalId ^ id);
          },

        // Nulop.
        In(Expr, Lhs) * T(Nulop) << (T(None) * T(Args)) >>
          [](Match& _) {
            auto id = _.fresh(l_local);
            return Seq << (Lift << Body << (Const << (LocalId ^ id) << None))
                       << (LocalId ^ id);
          },

        In(Expr, Lhs) * T(Nulop) << ((!T(None))[MethodId] * T(Args)) >>
          [](Match& _) {
            auto id = _.fresh(l_local);
            return Seq << (Lift << Body << (_(MethodId) << (LocalId ^ id)))
                       << (LocalId ^ id);
          },

        // Freeze preserves its source-level argument. The low-level operation
        // consumes an explicit copy.
        In(Expr, Lhs) * T(Freeze)[Lhs]
            << (T(Args) << (T(Arg) << (T(ArgCopy) * T(LocalId)[Rhs]))) >>
          [](Match& _) {
            auto copy_id = _.fresh(l_local);
            auto frozen_id = _.fresh(l_local);
            return Seq << (Lift << Body
                                << (Copy << (LocalId ^ copy_id) << _(Rhs)))
                       << (Lift << Body
                                << (Freeze << (LocalId ^ frozen_id)
                                           << (LocalId ^ copy_id)))
                       << (LocalId ^ frozen_id);
          },

        // Other single-arg builtin operations (already a LocalId after earlier
        // rules extract expressions to locals).
        In(Expr, Lhs) *
              T(MakeCallback, CodePtrCallback, FreeCallback, Pin, Unpin)[Lhs]
            << (T(Args) << (T(Arg) << (T(ArgCopy) * T(LocalId)[Rhs]))) >>
          [](Match& _) {
            auto id = _.fresh(l_local);
            auto rhs = _(Rhs);
            Node op = _(Lhs)->type();
            return Seq << (Lift << Body << (op << (LocalId ^ id) << rhs))
                       << (LocalId ^ id);
          },

        // Two-arg builtin operations (Merge).
        In(Expr, Lhs) * T(Merge)
            << (T(Args) << (T(Arg) << (T(ArgCopy) * T(LocalId)[Lhs])) *
                  (T(Arg) << (T(ArgCopy) * T(LocalId)[Rhs]))) >>
          [](Match& _) {
            auto id = _.fresh(l_local);
            return Seq << (Lift
                           << Body
                           << (Merge << (LocalId ^ id) << _(Lhs) << _(Rhs)))
                       << (LocalId ^ id);
          },

        // FFI struct layout builtin.
        In(Expr, Lhs) * T(FFIStruct) << (T(Type)[Type] * (T(Args) << End)) >>
          [](Match& _) {
            auto id = _.fresh(l_local);
            return Seq << (Lift << Body
                                << (FFIStruct << (LocalId ^ id)
                                              << clone((_(Type))->front())))
                       << (LocalId ^ id);
          },

        // FFI raw load builtin.
        In(Expr, Lhs) * T(FFILoad)
            << (T(Type)[Type] *
                (T(Args) << (T(Arg) << (T(ArgCopy) * T(LocalId)[Lhs])) *
                   (T(Arg) << (T(ArgCopy) * T(LocalId)[Rhs])) *
                   (T(Arg) << (T(ArgCopy) * T(LocalId)[MethodId])))) >>
          [](Match& _) {
            auto id = _.fresh(l_local);
            return Seq << (Lift << Body
                                << (FFILoad << (LocalId ^ id) << _(Lhs)
                                            << _(Rhs) << _(MethodId)
                                            << clone((_(Type))->front())))
                       << (LocalId ^ id);
          },

        // FFI raw store builtin.
        In(Expr, Lhs) * T(FFIStore)
            << (T(Type)[Type] *
                (T(Args) << (T(Arg) << (T(ArgCopy) * T(LocalId)[Lhs])) *
                   (T(Arg) << (T(ArgCopy) * T(LocalId)[Rhs])) *
                   (T(Arg) << (T(ArgCopy) * T(LocalId)[MethodId])) *
                   (T(Arg) << (T(ArgCopy) * T(LocalId)[SymbolId])))) >>
          [](Match& _) {
            auto id = _.fresh(l_local);
            return Seq << (Lift
                           << Body
                           << (FFIStore << (LocalId ^ id) << _(Lhs) << _(Rhs)
                                        << _(MethodId) << _(SymbolId)
                                        << clone((_(Type))->front())))
                       << (LocalId ^ id);
          },

        // FFI call.
        In(Expr, Lhs) * T(FFI) << (T(SymbolId)[SymbolId] * T(Args)[Args]) >>
          [](Match& _) {
            auto id = _.fresh(l_local);
            return Seq << (Lift
                           << Body
                           << (FFI << (LocalId ^ id) << _(SymbolId) << _(Args)))
                       << (LocalId ^ id);
          },

        // Bulk array operations.
        In(Expr, Lhs) * T(ArrayCopy, ArrayFill, ArrayCompare)[Lhs]
            << T(Args)[Args] >>
          [](Match& _) {
            auto id = _.fresh(l_local);
            auto op = _(Lhs)->type();
            return Seq << (Lift << Body
                                << (NodeDef::create(op)
                                    << (LocalId ^ id) << _(Args)))
                       << (LocalId ^ id);
          },

        // Compact LocalId.
        T(Expr, Lhs) << (T(LocalId)[LocalId] * End) >>
          [](Match& _) { return _(LocalId); },

        // Compact LHS Ref LocalId.
        T(Lhs) << (T(Ref) << T(LocalId)[LocalId] * End) >>
          [](Match& _) { return Ref << _(LocalId); },

        // Compact TupleLHS.
        T(Lhs) << (T(TupleLHS)[TupleLHS] * End) >>
          [](Match& _) { return _(TupleLHS); },

        // Compact LHS DontCare.
        T(Lhs) << (T(DontCare) * End) >>
          [](Match&) -> Node { return DontCare; },

        // Compact TupleLHS DontCare.
        In(TupleLHS) * T(Lhs) << (T(DontCare) * End) >>
          [](Match&) -> Node { return DontCare; },

        // Compact LHS Let inside TupleLHS. This must fire before the Let
        // rule converts Let to LocalId, so that Let survives into the
        // destructuring rule where it receives special handling (no swap).
        In(TupleLHS) * T(Lhs) << (T(Let)[Let] * End) >>
          [](Match& _) { return _(Let); },

        // Compact SplatLet inside TupleLHS.
        In(TupleLHS) * T(Lhs) << (T(SplatLet)[SplatLet] * End) >>
          [](Match& _) { return _(SplatLet); },

        // Compact SplatDontCare inside TupleLHS.
        In(TupleLHS) * T(Lhs) << (T(SplatDontCare) * End) >>
          [](Match&) -> Node { return SplatDontCare; },

        // Combine non-terminal LocalId with an incomplete copy.
        // This is for `if` and `else`.
        In(Body) * T(LocalId)[Rhs] * (T(Copy) << (T(LocalId)[Lhs] * End)) >>
          [](Match& _) { return Copy << _(Lhs) << _(Rhs); },

        // If there's a terminator, elide the incomplete Copy and the Jump.
        In(Body) * (T(Jump, Return, Raise))[Return] *
            (T(Copy) << (T(LocalId) * End)) * T(Jump) * End >>
          [](Match& _) -> Node { return _(Return); },

        // Discard non-terminal LocalId.
        In(Body) * T(LocalId) * ++Any >> [](Match&) -> Node { return {}; },

        // Decompose expression sequences.
        T(ExprSeq)[ExprSeq] >> [](Match& _) -> Node {
          auto p = _(ExprSeq);
          assert(!p->empty());
          Node seq = Seq;

          for (size_t i = 0; i < p->size() - 1; ++i)
            seq << (Lift << Body << p->at(i));

          seq << p->back();
          return seq;
        },
      }};

    p.post([](auto top) {
      top->traverse([&](auto node) {
        bool ok = true;

        if (node == Error)
        {
          ok = false;
        }
        else if (node->get_contains_error())
        {
          // Do nothing.
        }
        else if (node == Label)
        {
          // Move the terminator.
          auto body = node / Body;
          auto term = body->pop_back();

          if (term == LocalId)
            node << (Return << term);
          else if (term->in({Return, Raise, Jump, Cond}))
            node << term;
          else
          {
            // If the terminator is not a control flow node, it is an error.
            node << err(term, "Invalid terminator");
            ok = false;
          }
        }
        else if (node == Body)
        {
          for (auto& child : *node)
          {
            if (child->in({Return, Raise, Jump, Cond}))
            {
              node->replace(child, err(child, "Terminators must come last"));
              ok = false;
            }
          }
        }
        else if (node->in({Break, Continue}))
        {
          node->parent()->replace(
            node, err(node, "Break and continue must be inside a loop"));
          ok = false;
        }

        return ok;
      });

      return 0;
    });

    return p;
  }
}
