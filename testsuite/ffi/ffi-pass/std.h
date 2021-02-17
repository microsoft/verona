#include <array>

int main()
{
  std::array<double, 10> a;
  a[3] = 3.1415;
  return sizeof(a);
}
