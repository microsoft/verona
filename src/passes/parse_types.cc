#include "../infix.h"

namespace infix {

PassDef get_parse_types_pass() {
  // Normalise type lookups into an explicit chain of `Lookup` nodes, ready for
  // later resolution. This pass rewrites a leading name (with optional type
  // arguments) into a `Lookup`, then repeatedly consumes `::`-separated
  // segments to extend the chain.
  PassDef pass{
      "parse_types",
      wf_function_parse,
      dir::topdown,
      {
          In(Type)++ * T(Name, "\\|")[Name] >>
                [](auto &_) { return TypeOr ^ _(Name); },

          // Start of a type: Name [Square]? -> Lookup(Name, Args).
            (--T(DoubleColon)) * T(Name)[Name] * ~T(Square)[Args] * (--(In(Reference))) >>
              [](auto &_) {
                auto reference = Reference << _(Name)
                                               << (+(TypeArgs << +(*_[Args])));
                auto result = Lookup << +reference;
                return Seq << _[TypeOr] << result;
              },

          // Extend lookup chain: Lookup :: Name [Square]? ->
          // Lookup Lookup.
          T(Lookup)[Lhs] * T(DoubleColon) * T(Name)[Name] *
                  ~T(Square)[Args] >>
              [](auto &_) {
                auto extension = Reference << (+_(Name))
                                               << (+(TypeArgs << +(*_[Args])));
                auto result = _(Lhs) << (+extension);
                return result;
              },

          In(TypeArgs) * T(Group)[Group] >>
              [](auto &_) { return Type << (*_(Group)); },

      }};
  return pass;
}

} // namespace infix
