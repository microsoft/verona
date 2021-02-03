// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

// Test with: verona-ffi test.h Foo TFoo EFoo

#define SOME_MACRO

int FOO = 0;

class Foo { };

enum EFoo { };

template<class T = int>
struct TFoo {
  T innerFoo;
};

int foo() {
  TFoo<int> TF;
  return TF.innerFoo;
}
