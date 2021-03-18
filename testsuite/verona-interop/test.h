// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

// Non-template struct
struct NoTemp
{
  char a;
  short b;
  float c;
};

// Simple test, with default argument
template<class T, class U = int>
struct TFoo
{
  T innerFoo;
  T add(U arg)
  {
    return arg + innerFoo;
  }
};

// Simple test, with single default argument
template<class T = int>
struct SFoo
{
  T innerFoo;
  static T id(T arg)
  {
    return arg;
  }
};

int call_to_TFoo_add(TFoo<int>& obj, int arg)
{
  return obj.add(arg);
}

int call_to_SFoo_id(int arg)
{
  return SFoo<int>::id(arg);
}

/* These test namespaces and will be dealt with later
namespace One
{
  // Inheritance test, with required + default argument
  template<class A, class B = int>
  struct TestDefaultArgsBase
  {
    A a;
    B b;
    virtual A cast(B from) = 0;
    A convert(B from)
    {
      return cast(from);
    }
    virtual ~TestDefaultArgsBase() {}
  };

  namespace InnerOne
  {
    // Test using the default argument
    template<class T>
    struct TestDefaultArgsInt : public TestDefaultArgsBase<T>
    {
      T cast(int from)
      {
        return (T)from;
      }
    };
  }
}

namespace Two
{
  // Test not using the default argument
  template<class T>
  struct TestDefaultArgsFloat : public One::TestDefaultArgsBase<T, float>
  {
    T cast(float from)
    {
      return (T)from;
    }
  };
}
*/

/*
// The code below isn't part of the test, but it helps check the
// generated LLVM IR for comparison.
int foo()
{
  int isum = 0;
  float fsum = 0.0;

  // with default argument
  TFoo<int, long> TFdef;
  TFdef.innerFoo = 3;
  isum += TFdef.add(4);

  // without default argument
  TFoo<long> TFarg;
  TFarg.innerFoo = 3;
  isum += TFarg.add(4);

  // inheritance with default argument
  One::TestDefaultArgsBase<float> *T1 = new
One::InnerOne::TestDefaultArgsInt<float>(); fsum += T1->convert(42);

  // inheritance without default argument
  One::TestDefaultArgsBase<float, float> *T2 = new
Two::TestDefaultArgsFloat<float>(); fsum += T2->convert(42);

  return isum + (int)fsum;
}
// */
