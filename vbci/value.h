#pragma once

#include "ident.h"
#include "logging.h"
#include "platform.h"
#include "value_type.h"

#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <numbers>
#include <ostream>
#include <source_location>
#include <type_traits>

namespace vbci
{
  struct Register;
  struct ValueBorrow;
  struct ValueImmortal;
  struct ValueTransfer;

  template<bool is_move>
  using Reg = std::conditional_t<is_move, Register, const Register&>;

  struct ValueBits
  {
    uint64_t hi;
    uint64_t lo;
  };

  struct Value
  {
  private:
    struct Err
    {
      uintptr_t func : 56;
      Error error : 8;
    };

    union
    {
      bool b;
      int8_t i8;
      int16_t i16;
      int32_t i32;
      int64_t i64;
      uint8_t u8;
      uint16_t u16;
      uint32_t u32;
      uint64_t u64;
      long ilong;
      unsigned long ulong;
      ssize_t isize;
      size_t usize;
      float f32;
      double f64;
      void* ptr;
      Object* obj;
      Array* arr;
      Cown* cown;
      Err err;
      Function* func;
    };

    uint64_t idx : 56;
    ValueType tag : 7;
    uint8_t readonly : 1 = 0;

    Value(ValueType t) : tag(t), readonly(0) {}

  public:
    Value();
    explicit Value(bool b);
    explicit Value(uint8_t u8);
    explicit Value(uint16_t u16);
    explicit Value(uint32_t u32);
    explicit Value(uint64_t u64);
    explicit Value(int8_t i8);
    explicit Value(int16_t i16);
    explicit Value(int32_t i32);
    explicit Value(int64_t i64);
#ifdef PLATFORM_IS_MACOSX
    explicit Value(long ilong);
    explicit Value(unsigned long ulong);
#endif
    explicit Value(float f32);
    explicit Value(double f64);
    explicit Value(void* ptr);
    explicit Value(Object* obj);
    explicit Value(Object* obj, bool ro);
    explicit Value(Array* arr);
    explicit Value(Cown* cown);
    explicit Value(uint64_t idx, Location frame);
    explicit Value(Object* obj, size_t f, bool ro);
    explicit Value(Array* arr, size_t idx, bool ro);
    explicit Value(Cown* cown, bool ro);
    explicit Value(Error error);
    explicit Value(Function* func);

    template<typename T>
    explicit Value(ValueType t, T v) : tag(t)
    {
      set<T>(v);
    }

    operator ValueBits() const noexcept
    {
      static_assert(
        sizeof(Value) == sizeof(ValueBits), "ValueBits must match Value size");
      ValueBits bits;
      std::memcpy(&bits, this, sizeof(ValueBits));
      return bits;
    }

    static ValueImmortal none();
    static ValueImmortal null();
    static ValueImmortal from_ffi(ValueType t, uint64_t v);
    const void* to_ffi() const;
    static ValueBorrow from_addr(ValueType t, void* v);

    [[noreturn]] static void error(
      Error error,
      const std::source_location& location = std::source_location::current());

    void to_addr(ValueType t, void* v) const;

    ValueType type() const;
    uint32_t type_id() const;

    bool is_invalid() const;
    bool is_readonly() const;
    bool is_header() const;
    bool is_object() const;
    bool is_array() const;
    bool is_function() const;
    bool is_sendable() const;
    bool is_cown() const;
    bool is_error() const;
    bool get_bool() const;
    int32_t get_i32() const;
    uint8_t get_u8() const;
    uint64_t get_u64() const;
    void* get_ptr() const;
    Header* get_header() const;
    // Base allocation for direct and interior managed values, or nullptr.
    Header* get_base_header() const;
    Object* get_object() const;
    Cown* get_cown() const;
    Array* get_array() const;
    Function* function() const;
    Function* error_function() const;
    PC error_pc() const;
    size_t get_size() const;
    Register& get_register() const;

    Location location() const;
    Region* region() const;
    void immortalize();

    template<bool is_move>
    ValueTransfer exchange(Reg<is_move> v) const;
    ValueBorrow load_reference() const;

    Function* method(size_t w) const;
    Value convert(ValueType to) const;

    std::string to_string() const;

    ValueType get_value_type() const;

#define make_unop(name, func) \
  struct name \
  { \
    template<typename T> \
    constexpr auto operator()(T&& arg) const -> decltype(func) \
    { \
      return func; \
    } \
  }

#define make_binop(name, func) \
  struct name \
  { \
    template<typename T, typename U> \
    constexpr auto operator()(T&& lhs, U&& rhs) const -> decltype(func) \
    { \
      return func; \
    } \
  }

    Value op_add(const Value& v) const
    {
      return binop<nobinop, std::plus<>>(v);
    }

    Value op_sub(const Value& v) const
    {
      return binop<nobinop, std::minus<>>(v);
    }

    Value op_mul(const Value& v) const
    {
      return binop<nobinop, std::multiplies<>>(v);
    }

    Value op_div(const Value& v) const
    {
      return binop<nobinop, std::divides<>>(v);
    }

    make_binop(fmod, std::fmod(lhs, rhs));
    Value op_mod(const Value& v) const
    {
      return binop<nobinop, std::modulus<>, std::modulus<>, fmod>(v);
    }

    make_binop(fpow, std::pow(lhs, rhs));
    Value op_pow(const Value& v) const
    {
      return binop<nobinop, nobinop, nobinop, fpow>(v);
    }

    Value op_and(const Value& v) const
    {
      return binop<std::logical_and<>, std::bit_and<>, std::bit_and<>, nobinop>(
        v);
    }

    Value op_or(const Value& v) const
    {
      return binop<std::logical_or<>, std::bit_or<>, std::bit_or<>, nobinop>(v);
    }

    Value op_xor(const Value& v) const
    {
      return binop<
        std::not_equal_to<>,
        std::bit_xor<>,
        std::bit_xor<>,
        nobinop>(v);
    }

    make_binop(bit_left_shift, lhs << rhs);
    Value op_shl(const Value& v) const
    {
      return binop<nobinop, bit_left_shift, bit_left_shift, nobinop>(v);
    }

    make_binop(bit_right_shift, lhs >> rhs);
    Value op_shr(const Value& v) const
    {
      return binop<nobinop, bit_right_shift, bit_right_shift, nobinop>(v);
    }

    Value op_eq(const Value& v) const
    {
      if (tag != v.tag)
        return Value(false);

      switch (tag)
      {
        case ValueType::None:
          return Value(true);

        case ValueType::Bool:
          return Value(b == v.b);

        case ValueType::I8:
          return Value(i8 == v.i8);

        case ValueType::I16:
          return Value(i16 == v.i16);

        case ValueType::I32:
          return Value(i32 == v.i32);

        case ValueType::I64:
          return Value(i64 == v.i64);

        case ValueType::U8:
          return Value(u8 == v.u8);

        case ValueType::U16:
          return Value(u16 == v.u16);

        case ValueType::U32:
          return Value(u32 == v.u32);

        case ValueType::U64:
          return Value(u64 == v.u64);

        case ValueType::ILong:
          return Value(ilong == v.ilong);

        case ValueType::ULong:
          return Value(ulong == v.ulong);

        case ValueType::ISize:
          return Value(isize == v.isize);

        case ValueType::USize:
          return Value(usize == v.usize);

        case ValueType::F32:
          return Value(f32 == v.f32);

        case ValueType::F64:
          return Value(f64 == v.f64);

        case ValueType::Ptr:
          return Value(ptr == v.ptr);

        case ValueType::Object:
          return Value(obj == v.obj);

        case ValueType::Array:
          return Value(arr == v.arr);

        case ValueType::Cown:
          return Value(cown == v.cown);

        case ValueType::RegisterRef:
          return Value(u64 == v.u64);

        case ValueType::FieldRef:
          return Value((obj == v.obj) && (idx == v.idx));

        case ValueType::ArrayRef:
          return Value((arr == v.arr) && (idx == v.idx));

        case ValueType::CownRef:
          return Value(cown == v.cown);

        case ValueType::Function:
          return Value(func == v.func);

        case ValueType::Error:
          return Value(err.error == v.err.error);

        case ValueType::Invalid:
          return Value(true);

        default:
          // Unreachable.
          assert(false);
          return Value(false);
      }
    }

    Value op_ne(const Value& v) const
    {
      auto r = op_eq(v);
      r.b = !r.b;
      return r;
    }

    Value op_lt(const Value& v) const
    {
      return binop<std::less<>>(v);
    }

    Value op_le(const Value& v) const
    {
      return binop<std::less_equal<>>(v);
    }

    Value op_gt(const Value& v) const
    {
      return binop<std::greater<>>(v);
    }

    Value op_ge(const Value& v) const
    {
      return binop<std::greater_equal<>>(v);
    }

    make_binop(min, std::min(lhs, rhs));
    Value op_min(const Value& v) const
    {
      return binop<min>(v);
    }

    make_binop(max, std::max(lhs, rhs));
    Value op_max(const Value& v) const
    {
      return binop<max>(v);
    }

    make_binop(logbase, std::log(lhs) / std::log(rhs));
    Value op_logbase(const Value& v) const
    {
      return binop<nobinop, nobinop, nobinop, logbase>(v);
    }

    make_binop(atan2, std::atan2(lhs, rhs));
    Value op_atan2(const Value& v) const
    {
      return binop<nobinop, nobinop, nobinop, atan2>(v);
    }

    Value op_neg() const
    {
      return unop<nounop, std::negate<>>();
    }

    Value op_not() const
    {
      return unop<std::logical_not<>, std::bit_not<>, std::bit_not<>, nounop>();
    }

    make_unop(abs, std::abs(arg));
    Value op_abs() const
    {
      return unop<nounop, abs, nounop, abs>();
    }

    make_unop(ceil, std::ceil(arg));
    Value op_ceil() const
    {
      return unop<nounop, nounop, nounop, ceil>();
    }

    make_unop(floor, std::floor(arg));
    Value op_floor() const
    {
      return unop<nounop, nounop, nounop, floor>();
    }

    make_unop(exp, std::exp(arg));
    Value op_exp() const
    {
      return unop<nounop, nounop, nounop, exp>();
    }

    make_unop(log, std::log(arg));
    Value op_log() const
    {
      return unop<nounop, nounop, nounop, log>();
    }

    make_unop(sqrt, std::sqrt(arg));
    Value op_sqrt() const
    {
      return unop<nounop, nounop, nounop, sqrt>();
    }

    make_unop(cbrt, std::cbrt(arg));
    Value op_cbrt() const
    {
      return unop<nounop, nounop, nounop, cbrt>();
    }

    make_unop(isinf, std::isinf(arg));
    Value op_isinf() const
    {
      return unop<nounop, nounop, nounop, isinf>();
    }

    make_unop(isnan, std::isnan(arg));
    Value op_isnan() const
    {
      return unop<nounop, nounop, nounop, isnan>();
    }

    make_unop(sin, std::sin(arg));
    Value op_sin() const
    {
      return unop<nounop, nounop, nounop, sin>();
    }

    make_unop(cos, std::cos(arg));
    Value op_cos() const
    {
      return unop<nounop, nounop, nounop, cos>();
    }

    make_unop(tan, std::tan(arg));
    Value op_tan() const
    {
      return unop<nounop, nounop, nounop, tan>();
    }

    make_unop(asin, std::asin(arg));
    Value op_asin() const
    {
      return unop<nounop, nounop, nounop, asin>();
    }

    make_unop(acos, std::acos(arg));
    Value op_acos() const
    {
      return unop<nounop, nounop, nounop, acos>();
    }

    make_unop(atan, std::atan(arg));
    Value op_atan() const
    {
      return unop<nounop, nounop, nounop, atan>();
    }

    make_unop(sinh, std::sinh(arg));
    Value op_sinh() const
    {
      return unop<nounop, nounop, nounop, sinh>();
    }

    make_unop(cosh, std::cosh(arg));
    Value op_cosh() const
    {
      return unop<nounop, nounop, nounop, cosh>();
    }

    make_unop(tanh, std::tanh(arg));
    Value op_tanh() const
    {
      return unop<nounop, nounop, nounop, tanh>();
    }

    make_unop(asinh, std::asinh(arg));
    Value op_asinh() const
    {
      return unop<nounop, nounop, nounop, asinh>();
    }

    make_unop(acosh, std::acosh(arg));
    Value op_acosh() const
    {
      return unop<nounop, nounop, nounop, acosh>();
    }

    make_unop(atanh, std::atanh(arg));
    Value op_atanh() const
    {
      return unop<nounop, nounop, nounop, atanh>();
    }

    Value op_bits() const;
    Value op_len() const;
    Value op_ptr();
    ValueBorrow op_read() const;

    static Value e()
    {
      return Value(std::numbers::e);
    }

    static Value pi()
    {
      return Value(std::numbers::pi);
    }

    static Value inf()
    {
      return Value(std::numeric_limits<double>::infinity());
    }

    static Value nan()
    {
      return Value(std::numeric_limits<double>::quiet_NaN());
    }

    // Increment RC for a reference held in a register context.
    // Adjusts both object RC and region stack_rc.
    void reg_inc() const;

    // Decrement RC for a reference held in a register context.
    // Adjusts both object RC and region stack_rc. Collects if RC hits 0.
    void reg_dec() const;

    // Increment RC for a reference held in a field context.
    // Adjusts object RC only — no stack_rc change.
    void field_inc() const;

    // Decrement RC for a reference held in a field context.
    // Adjusts object RC only — no stack_rc change. Collects if RC hits 0.
    void field_dec() const;

    // Pin or unpin a refcounted value for external use.
    void pin() const;
    void unpin() const;

  public:
    struct nounop
    {
      template<typename T>
      constexpr T operator()(T&&) const
      {
        Value::error(Error::BadOperand);
      }
    };

    struct nobinop
    {
      template<typename T, typename U>
      constexpr T operator()(T&&, U&&) const
      {
        Value::error(Error::BadOperand);
      }
    };

    static Value platform_tag(ValueType t, Value&& v)
    {
      switch (v.tag)
      {
        case ValueType::I32:
        case ValueType::I64:
        case ValueType::U32:
        case ValueType::U64:
#ifdef PLATFORM_IS_MACOSX
        case ValueType::ILong:
        case ValueType::ULong:
#endif
          v.tag = t;
          break;

        default:
          break;
      }

      return v;
    }

    template<
      typename OpB,
      typename OpI = OpB,
      typename OpU = OpI,
      typename OpF = OpU>
    Value unop() const
    {
      switch (tag)
      {
        case ValueType::Bool:
          return Value(OpB{}(b));

        case ValueType::I8:
          return Value(OpI{}(i8));

        case ValueType::I16:
          return Value(OpI{}(i16));

        case ValueType::I32:
          return Value(OpI{}(i32));

        case ValueType::I64:
          return Value(OpI{}(i64));

        case ValueType::U8:
          return Value(OpU{}(u8));

        case ValueType::U16:
          return Value(OpU{}(u16));

        case ValueType::U32:
          return Value(OpU{}(u32));

        case ValueType::U64:
          return Value(OpU{}(u64));

        case ValueType::ILong:
          return platform_tag(this->tag, Value(OpI{}(ilong)));

        case ValueType::ULong:
          return platform_tag(this->tag, Value(OpU{}(ulong)));

        case ValueType::ISize:
          return platform_tag(this->tag, Value(OpI{}(isize)));

        case ValueType::USize:
          return platform_tag(this->tag, Value(OpU{}(usize)));

        case ValueType::F32:
          return Value(OpF{}(f32));

        case ValueType::F64:
          return Value(OpF{}(f64));

        default:
          Value::error(Error::BadOperand);
      }
    }

    template<typename Op, typename T>
    static Value narrow_binop(T lhs, T rhs)
    {
      auto result = Op{}(lhs, rhs);
      if constexpr (std::is_same_v<decltype(result), bool>)
        return Value(result);
      else
        return Value(static_cast<T>(result));
    }

    template<
      typename OpB,
      typename OpI = OpB,
      typename OpU = OpI,
      typename OpF = OpU>
    Value binop(const Value& v) const
    {
      if (this->tag != v.tag)
        Value::error(Error::MismatchedTypes);

      switch (this->tag)
      {
        case ValueType::Bool:
          return Value(OpB{}(b, v.b));

        case ValueType::I8:
          return narrow_binop<OpI>(i8, v.i8);

        case ValueType::I16:
          return narrow_binop<OpI>(i16, v.i16);

        case ValueType::I32:
          return Value(OpI{}(i32, v.i32));

        case ValueType::I64:
          return Value(OpI{}(i64, v.i64));

        case ValueType::U8:
          return narrow_binop<OpU>(u8, v.u8);

        case ValueType::U16:
          return narrow_binop<OpU>(u16, v.u16);

        case ValueType::U32:
          return Value(OpU{}(u32, v.u32));

        case ValueType::U64:
          return Value(OpU{}(u64, v.u64));

        case ValueType::ILong:
          return platform_tag(this->tag, Value(OpI{}(ilong, v.ilong)));

        case ValueType::ULong:
          return platform_tag(this->tag, Value(OpU{}(ulong, v.ulong)));

        case ValueType::ISize:
          return platform_tag(this->tag, Value(OpI{}(isize, v.isize)));

        case ValueType::USize:
          return platform_tag(this->tag, Value(OpU{}(usize, v.usize)));

        case ValueType::F32:
          return Value(OpF{}(f32, v.f32));

        case ValueType::F64:
          return Value(OpF{}(f64, v.f64));

        default:
          Value::error(Error::BadOperand);
      }
    }

    template<typename T>
    T get() const
    {
      switch (tag)
      {
        case ValueType::None:
          return 0;

        case ValueType::Bool:
          return b;

        case ValueType::I8:
          return i8;

        case ValueType::I16:
          return i16;

        case ValueType::I32:
          return i32;

        case ValueType::I64:
          return i64;

        case ValueType::U8:
          return u8;

        case ValueType::U16:
          return u16;

        case ValueType::U32:
          return u32;

        case ValueType::U64:
          return u64;

        case ValueType::ILong:
          return ilong;

        case ValueType::ULong:
          return ulong;

        case ValueType::ISize:
          return isize;

        case ValueType::USize:
          return usize;

        case ValueType::F32:
          return f32;

        case ValueType::F64:
          return f64;

        case ValueType::Ptr:
          return reinterpret_cast<size_t>(ptr);

        case ValueType::Object:
          return reinterpret_cast<size_t>(obj);

        case ValueType::Array:
          return reinterpret_cast<size_t>(arr);

        case ValueType::Cown:
          return reinterpret_cast<size_t>(cown);

        case ValueType::RegisterRef:
          return u64;

        case ValueType::FieldRef:
          return reinterpret_cast<size_t>(obj) + idx + 1;

        case ValueType::ArrayRef:
          return reinterpret_cast<size_t>(arr) + idx + 1;

        case ValueType::CownRef:
          return reinterpret_cast<size_t>(cown) + 1;

        case ValueType::Function:
          return reinterpret_cast<size_t>(func);

        default:
          Value::error(Error::BadOperand);
      }
    }

    template<typename T>
    void set(T value)
    {
      switch (tag)
      {
        case ValueType::None:
          break;

        case ValueType::Bool:
          b = !!value;
          break;

        case ValueType::I8:
          i8 = value;
          break;

        case ValueType::I16:
          i16 = value;
          break;

        case ValueType::I32:
          i32 = value;
          break;

        case ValueType::I64:
          i64 = value;
          break;

        case ValueType::U8:
          u8 = value;
          break;

        case ValueType::U16:
          u16 = value;
          break;

        case ValueType::U32:
          u32 = value;
          break;

        case ValueType::U64:
          u64 = value;
          break;

        case ValueType::ILong:
          ilong = value;
          break;

        case ValueType::ULong:
          ulong = value;
          break;

        case ValueType::ISize:
          isize = value;
          break;

        case ValueType::USize:
          usize = value;
          break;

        case ValueType::F32:
          f32 = value;
          break;

        case ValueType::F64:
          f64 = value;
          break;

        default:
          Value::error(Error::BadOperand);
      }
    }
  };

  inline std::ostream& operator<<(std::ostream& os, ValueType vt)
  {
    switch (vt)
    {
      case ValueType::None:
        return os << "None";
      case ValueType::Bool:
        return os << "Bool";
      case ValueType::I8:
        return os << "I8";
      case ValueType::I16:
        return os << "I16";
      case ValueType::I32:
        return os << "I32";
      case ValueType::I64:
        return os << "I64";
      case ValueType::U8:
        return os << "U8";
      case ValueType::U16:
        return os << "U16";
      case ValueType::U32:
        return os << "U32";
      case ValueType::U64:
        return os << "U64";
      case ValueType::ILong:
        return os << "ILong";
      case ValueType::ULong:
        return os << "ULong";
      case ValueType::ISize:
        return os << "ISize";
      case ValueType::USize:
        return os << "USize";
      case ValueType::F32:
        return os << "F32";
      case ValueType::F64:
        return os << "F64";
      case ValueType::Ptr:
        return os << "Ptr";
      case ValueType::Object:
        return os << "Object";
      case ValueType::Array:
        return os << "Array";
      case ValueType::Cown:
        return os << "Cown";
      case ValueType::RegisterRef:
        return os << "RegisterRef";
      case ValueType::FieldRef:
        return os << "FieldRef";
      case ValueType::ArrayRef:
        return os << "ArrayRef";
      case ValueType::CownRef:
        return os << "CownRef";
      case ValueType::Function:
        return os << "Function";
      case ValueType::Error:
        return os << "Error";
      case ValueType::Invalid:
        return os << "Invalid";
      default:
        return os << "Unknown";
    }
  }

  inline std::ostream& operator<<(std::ostream& os, const Value& v)
  {
    os << v.to_string();
    if (v.is_readonly())
    {
      os << " Read-only ";
    }
    return os;
  }
}
