#include "../infix.h"

namespace infix {

PassDef get_function_parse_pass() {
  return PassDef{
      "function_parse",
      wf_function_parse,
      dir::topdown,
      {

          In(Function) * Start * --T(TypeParams) * (~T(Square))[Square] >>
              [](auto &_) -> Node {
            return Seq << (TypeParams << *_[Square]) << Lhs;
          },

          In(TypeParams) * T(Group)[Group] >>
              [](auto &_) -> Node { return TypeParam << *_[Group]; },

          In(Function) * T(Lhs, Rhs)[Lhs] *
                  (T(Group, Paren) << (T(Name)[Name] * T(Colon) * ~T(Hat)[Hat] *
                                       Any++[Type])) >>
              [](auto &_) {
                return _(Lhs)
                       << (Param << _(Name) << (Mode << (_(Hat) ? CBN : CBV))
                                 << (Type << _[Type]));
              },

          In(Function) * T(Lhs, Rhs, Type, Where)[Lhs] *
                  (T(Group, Indent))[Group] >>
              [](auto &_) { return Seq << _(Lhs) << *_(Group); },

          In(Function) * T(Lhs)[Lhs] * T(Name)[Name] * --T(Rhs) >>
              [](auto &_) { return Seq << _(Lhs) << _(Name) << Rhs; },

          In(Function) * T(Rhs)[Rhs] * T(Colon) >>
              [](auto &_) { return Seq << _(Rhs) << Type; },

          In(Function) * T(Type)[Type] * T(Eq) * Any++[Body] >>
              [](auto &_) {
                return Seq << _(Type) << Where
                           << (Body << ExprStack << _[Body]);
              },

          In(Function) * T(Where)[Where] * T(Eq) * Any++[Body] >>
              [](auto &_) {
                return Seq << _(Where) << (Body << ExprStack << _[Body]);
              },

          In(Function) * T(Type)[Type] * (!T(Eq, Body, Where))[Rhs] >>
              [](auto &_) { return _(Type) << _[Rhs]; },

          In(Function) * T(Where)[Where] * (!T(Eq, Body))[Rhs] >>
              [](auto &_) { return _(Where) << _[Rhs]; },
      }};
}

} // namespace infix