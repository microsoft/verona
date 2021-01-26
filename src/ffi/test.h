
#define SOME_MACRO

int FOO = 0;

class Foo { };

enum EFoo { };

template<class T>
struct TFoo {
  T innerFoo;
};

int foo() {
  TFoo<int> TF;
  return TF.innerFoo;
}
