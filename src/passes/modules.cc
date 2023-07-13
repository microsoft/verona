// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#include "lang.h"

namespace verona
{
  PassDef modules()
  {
    return {
      // Files at the top level and directories are modules.
      ((In(Top) * T(File)[Class]) / T(Directory)[Class]) >>
        [](Match& _) {
          return Group << (Class ^ _(Class)) << (Ident ^ _(Class)->location())
                       << (Brace << *_[Class]);
        },

      // Files in a directory aren't semantically meaningful.
      In(Brace) * T(File)[File] >> [](Match& _) { return Seq << *_[File]; },

      In(Type)++ * T(Colon)[Colon] >>
        [](Match& _) {
          return err(_[Colon], "can't put a type assertion inside a type");
        },

      In(Type)++ * T(Where)[Where] >>
        [](Match& _) {
          return err(_[Where], "can't put a type predicate inside a type");
        },

      // Type assertion. Treat an empty assertion as DontCare. Accept a brace if
      // it comes immediately after the colon or after a symbol or dot.
      // Otherwise end the type at a Where, Brace, or TripleColon.
      T(Colon) *
          (~T(Brace) *
           (((T(Symbol) / T(Dot)) * T(Brace)) /
            (!(T(Where) / T(Brace) / T(TripleColon))))++)[Type] >>
        [](Match& _) { return Type << (_[Type] || DontCare); },

      // Type predicate.
      T(Where) *
          (~T(Brace) *
           (((T(Symbol) / T(Dot)) * T(Brace)) /
            (!(T(Where) / T(Brace) / T(TripleColon))))++)[Type] >>
        [](Match& _) { return TypePred << (Type << (_[Type] || TypeTrue)); },

      T(TripleColon) *
          (T(Paren)
           << ((T(List) << (T(Group) << (T(Ident) / T(LLVM)))++[Args]) /
               ~(T(Group) << (T(Ident) / T(LLVM)))[Args])) *
          T(Symbol, "->") * (T(Ident) / T(LLVM))[Return] >>
        [](Match& _) {
          return LLVMFuncType << (LLVMList << *_[Args]) << _(Return);
        },

      T(TripleColon)[TripleColon] >>
        [](Match& _) { return err(_[TripleColon], "malformed LLVM type"); },
    };
  }
}
