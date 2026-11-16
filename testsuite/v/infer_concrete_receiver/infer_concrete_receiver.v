widen(value: i16): i64
{
  let widened = i64(value.i64);
  if value != 7
  {
    return -1
  }
  widened
}

main(): none
{
  if infer_concrete_receiver::widen(7) != 7
  {
    ffi::exit_code(1)
  }
}
