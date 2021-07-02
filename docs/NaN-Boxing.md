# NaN Boxing

## Packing into a double

Doubles are 64 bits, on both 32-bit and 64-bit platforms.

1. 1 sign bit
2. 11 exponent bits
3. 52 mantissa bits

Encodable types:
* Pointer
* F64
* F32
* U32
* U16
* U8
* I32
* I16
* I8
* Bool
* ISize (on a 32 bit platform)
* USize (on a 32 bit platform)

```
[*][***********][52 bit mantissa] = F64
[1][11111111111][52 data bits] = Pointer
[0][11111111111][0000][47*][1 data bit] = Bool
[0][11111111111][0001][40*][8 data bits] = F8
[0][11111111111][0010][32*][16 data bits] = F16
[0][11111111111][0011][16*][32 data bits] = F32
[0][11111111111][0100][40*][8 data bits] = I8
[0][11111111111][0101][32*][16 data bits] = I16
[0][11111111111][0110][16*][32 data bits] = I32
[0][11111111111][0111][16*][32 data bits] = ISize (32)
[0][11111111111][1000][40*][8 data bits] = U8
[0][11111111111][1001][32*][16 data bits] = U16
[0][11111111111][1010][16*][32 data bits] = U32
[0][11111111111][1011][16*][32 data bits] = USize (32)
```

Unencodable types:
* I64
* U64
* I128
* U128
* F128
* ISize (on a 64 bit platform)
* USize (on a 64 bit platform)

## Wide packing

Use an 8-bit discriminator in an LLVM struct.

%packed64 allows encoding everything that can be NaN boxed, plus I64 and U64.
%packed128 allows encoding everything.

```
%packed64 = type { i8, i64 }
%packed128 = type { i8, i128 }
```
