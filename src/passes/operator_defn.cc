#include "../infix.h"

namespace infix {

PassDef get_operator_defn_pass() {
  auto Decls = T(Struct, TypeAlias, Let, Module, Function);

  return PassDef{
      "structure",
      wf::empty,
      dir::topdown,
      {T(Group)[Group] << (Decls[Lhs] * Any++[Rhs]) >> [](auto &_) -> Node {
         auto result = (_(Lhs)->type() ^ _(Group)) << _[Rhs];
         return result;
       },
       T(Paren)[Paren] << (T(Group)[Group] * End) >>
           [](auto &_) -> Node { return (Paren ^ _(Paren)) << *_(Group); },

       T(Group) << (T(Paren)[Paren] * End) >>
           [](auto &_) -> Node { return _(Paren); },

       T(Group) << (T(Square)[Square] * End) >>
           [](auto &_) -> Node { return _(Square); },

       // Detect offside indentation in the body of a function,
       // struct or module.
       (T(Indent) * T(Indent)[Indent]) * In(Body, Struct, Module) >>
           [](auto &_) {
             return Error << (ErrorMsg ^ "Offside indentation detected!")
                          << (ErrorAst << _[Indent]);
           },

       In(Struct, Module, TypeAlias) * T(Indent)[Indent] * End >>
           [](auto &_) { return Seq << *_(Indent); },

       T(Struct)[Struct] << (T(Name)[Name] * (~T(Square))[Square] *
                             T(Group, Paren)++[Group] * End) >>
           [](auto &_) -> Node {
         return (Struct ^ _(Struct)) << (_[Name]) << (TypeParams << *_[Square])
                                     << (Fields << _[Group]);
       },

       Any[Lhs] * T(Dot) * T(Name)[Name] >>
           [](auto &_) { return Lookup << _(Name) << (Args << _(Lhs)); },

       In(Fields) * T(Group, Paren)
               << (T(Name)[Name] * T(Colon) * Any++[Type]) >>
           [](auto &_) { return Field << _(Name) << (Type << _[Type]); },

       T(TypeAlias)[TypeAlias] << (T(Name)[Name] * (~T(Square))[Square] *
                                   T(Eq) * Any++[Body] * End) >>
           [](auto &_) -> Node {
         return (TypeAlias ^ _(TypeAlias))
                << _[Name] << (TypeParams << *_[Square]) << (Type << _[Body]);
       },

       T(Module)[Module] << (T(Name)[Name] * ~T(Square)[TypeParams] *
                             (--T(Body, TypeParams) * Any++)[Body] * End) >>
           [](auto &_) -> Node {
         auto result = (Module ^ _(Module))
                       << _(Name) << (TypeParams << *_[TypeParams])
                       << (Body << _[Body]);
         return result;
       },

       T(Group)[Group] << (T(Use) * Any++[Rhs]) >> [](auto &_) -> Node {
         return (Use ^ _(Group)) << (Type << _[Rhs]);
       }}};
}
} // namespace infix