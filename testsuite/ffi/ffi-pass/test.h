// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

// Test with: verona-ffi test.h Foo TFoo EFoo

#define SOME_MACRO

int FOO = 0;

class Foo
{};

enum EFoo
{
};

template<class T = int>
struct TFoo
{
  T innerFoo;
  T add(T arg)
  {
    return arg + innerFoo;
  }
};

int foo()
{
  TFoo<int> TF;
  TF.innerFoo = 3;
  return TF.add(4);
}
