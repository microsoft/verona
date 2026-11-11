check[T](a: T, b: T): bool
{
  let sum: T = a + b;
  (a < b) & (a <= a) & (b > a) & (b >= b) & ((sum - b) == a)
}

main(): none
{
  var result = 0;
  if !narrow_compare::check[i8](1, 2)
  {
    result = result + 1
  }
  if !narrow_compare::check[u8](1, 2)
  {
    result = result + 2
  }
  if !narrow_compare::check[i16](1, 2)
  {
    result = result + 4
  }
  if !narrow_compare::check[u16](1, 2)
  {
    result = result + 8
  }
  ffi::exit_code(result)
}
