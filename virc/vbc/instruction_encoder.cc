#include "instruction_encoder.h"

#include "../lang.h"
#include "type_encoding.h"

#include <vbc/format.h>

namespace virc::vbc_backend
{
  using namespace trieste;
  using namespace vir;
  using namespace ::vbc;

  namespace
  {
    class InstructionEncoder
    {
    private:
      Compilation& compilation;
      FuncState& function;
      const MemoSlots& memo_slots;
      ByteBuffer& output;

      uleb<size_t> dst(Node statement)
      {
        return uleb(*function.get_register_id(statement / LocalId));
      }

      uleb<size_t> lhs(Node statement)
      {
        return uleb(*function.get_register_id(statement / Lhs));
      }

      uleb<size_t> rhs(Node statement)
      {
        return uleb(*function.get_register_id(statement / Rhs));
      }

      uleb<size_t> src(Node statement)
      {
        return rhs(statement);
      }

      uleb<size_t> cls(Node statement)
      {
        return uleb(compilation.type_id(statement / ClassId));
      }

      uleb<size_t> fld(Node statement)
      {
        return uleb(*compilation.get_field_id(statement / FieldId));
      }

      uleb<size_t> mth(Node statement)
      {
        return uleb(*compilation.get_method_id(statement / MethodId));
      }

      uleb<size_t> fn(Node statement)
      {
        return uleb(*compilation.get_func_id(statement / FunctionId));
      }

      void argument(Node argument)
      {
        if ((argument / Type) == ArgMove)
          output << uleb(+Op::ArgMove) << uleb(src(argument));
        else
          output << uleb(+Op::ArgCopy) << uleb(src(argument));
      }

      void arguments(Node arguments)
      {
        for (auto argument_node : *arguments)
          argument(argument_node);
      }

      void binary(Node statement, Op op)
      {
        output << uleb(+op) << dst(statement) << lhs(statement)
               << rhs(statement);
      }

      void unary(Node statement, Op op)
      {
        output << uleb(+op) << dst(statement) << src(statement);
      }

    public:
      InstructionEncoder(
        Compilation& compilation,
        FuncState& function,
        const MemoSlots& memo_slots,
        ByteBuffer& output)
      : compilation(compilation),
        function(function),
        memo_slots(memo_slots),
        output(output)
      {}

      void statement(Node statement)
      {
        if (statement == Const)
        {
          auto type = statement / Type;
          auto value = statement / Rhs;

          if (type == None)
          {
            output << uleb(+Op::Const) << dst(statement) << uleb(+val(type));
          }
          else if (type == Bool)
          {
            output << uleb(+Op::Const) << dst(statement) << uleb(+val(type));
            output << uleb(((statement / Rhs) == True) ? 1 : 0);
          }
          else if (type == I8)
          {
            output << uleb(+Op::Const) << dst(statement) << uleb(+val(type))
                   << sleb(from_chars_sep_v<int8_t>(value));
          }
          else if (type == U8)
          {
            output << uleb(+Op::Const) << dst(statement) << uleb(+val(type))
                   << uleb(from_chars_sep_v<uint8_t>(value));
          }
          else if (type == I16)
          {
            output << uleb(+Op::Const) << dst(statement) << uleb(+val(type))
                   << sleb(from_chars_sep_v<int16_t>(value));
          }
          else if (type == U16)
          {
            output << uleb(+Op::Const) << dst(statement) << uleb(+val(type))
                   << uleb(from_chars_sep_v<uint16_t>(value));
          }
          else if (type == I32)
          {
            output << uleb(+Op::Const) << dst(statement) << uleb(+val(type))
                   << sleb(from_chars_sep_v<int32_t>(value));
          }
          else if (type == U32)
          {
            output << uleb(+Op::Const) << dst(statement) << uleb(+val(type))
                   << uleb(from_chars_sep_v<uint32_t>(value));
          }
          else if (type->in({I64, ILong, ISize}))
          {
            output << uleb(+Op::Const) << dst(statement) << uleb(+val(type))
                   << sleb(from_chars_sep_v<int64_t>(value));
          }
          else if (type->in({U64, ULong, USize, Ptr}))
          {
            output << uleb(+Op::Const) << dst(statement) << uleb(+val(type))
                   << uleb(from_chars_sep_v<uint64_t>(value));
          }
          else if (type == F32)
          {
            output << uleb(+Op::Const) << dst(statement) << uleb(+val(type))
                   << sleb(from_chars_sep_v<float>(value));
          }
          else if (type == F64)
          {
            output << uleb(+Op::Const) << dst(statement) << uleb(+val(type))
                   << sleb(from_chars_sep_v<double>(value));
          }
        }
        else if (statement == ConstStr)
        {
          output << uleb(+Op::String) << dst(statement)
                 << uleb(ST::exec().string(statement / String));
        }
        else if (statement == Convert)
        {
          output << uleb(+Op::Convert) << dst(statement)
                 << uleb(+val(statement / Type)) << rhs(statement);
        }
        else if (statement == New)
        {
          arguments(statement / Args);
          output << uleb(+Op::New) << dst(statement) << cls(statement);
        }
        else if (statement == Stack)
        {
          arguments(statement / Args);
          output << uleb(+Op::Stack) << dst(statement) << cls(statement);
        }
        else if (statement == Heap)
        {
          arguments(statement / Args);
          output << uleb(+Op::Heap) << dst(statement) << rhs(statement)
                 << cls(statement);
        }
        else if (statement == Region)
        {
          arguments(statement / Args);
          output << uleb(+Op::Region) << dst(statement)
                 << encode_region(statement) << cls(statement);
        }
        else if (statement == NewArray)
        {
          output << uleb(+Op::NewArray) << dst(statement) << rhs(statement)
                 << uleb(compilation.type_id(statement / Type));
        }
        else if (statement == NewArrayConst)
        {
          output << uleb(+Op::NewArrayConst) << dst(statement)
                 << uleb(compilation.type_id(statement / Type))
                 << uleb(from_chars_sep_v<uint64_t>(statement / Rhs));
        }
        else if (statement == StackArray)
        {
          output << uleb(+Op::StackArray) << dst(statement) << rhs(statement)
                 << uleb(compilation.type_id(statement / Type));
        }
        else if (statement == StackArrayConst)
        {
          output << uleb(+Op::StackArrayConst) << dst(statement)
                 << uleb(compilation.type_id(statement / Type))
                 << uleb(from_chars_sep_v<uint64_t>(statement / Rhs));
        }
        else if (statement == HeapArray)
        {
          output << uleb(+Op::HeapArray) << dst(statement) << lhs(statement)
                 << rhs(statement)
                 << uleb(compilation.type_id(statement / Type));
        }
        else if (statement == HeapArrayConst)
        {
          output << uleb(+Op::HeapArrayConst) << dst(statement)
                 << lhs(statement)
                 << uleb(compilation.type_id(statement / Type))
                 << uleb(from_chars_sep_v<uint64_t>(statement / Rhs));
        }
        else if (statement == RegionArray)
        {
          output << uleb(+Op::RegionArray) << dst(statement)
                 << encode_region(statement) << rhs(statement)
                 << uleb(compilation.type_id(statement / Type));
        }
        else if (statement == RegionArrayConst)
        {
          output << uleb(+Op::RegionArrayConst) << dst(statement)
                 << encode_region(statement)
                 << uleb(compilation.type_id(statement / Type))
                 << uleb(from_chars_sep_v<uint64_t>(statement / Rhs));
        }
        else if (statement == Copy)
        {
          output << uleb(+Op::Copy) << dst(statement) << src(statement);
        }
        else if (statement == Move)
        {
          output << uleb(+Op::Move) << dst(statement) << src(statement);
        }
        else if (statement == Drop)
        {
          output << uleb(+Op::Drop) << dst(statement);
        }
        else if (statement == AtTeardown)
        {
          output << uleb(+Op::AtTeardown) << dst(statement);
        }
        else if (statement == Freeze)
        {
          output << uleb(+Op::Freeze) << dst(statement) << src(statement);
        }
        else if (statement == RegisterRef)
        {
          output << uleb(+Op::RegisterRef) << dst(statement) << src(statement);
        }
        else if (statement == FieldRef)
        {
          auto argument_node = statement / Arg;
          output << uleb(
            +(((argument_node / Type) == ArgMove) ? Op::FieldRefMove :
                                                    Op::FieldRefCopy));
          output << dst(statement) << src(argument_node) << fld(statement);
        }
        else if (statement == ArrayRef)
        {
          auto argument_node = statement / Arg;
          output << uleb(
            +(((argument_node / Type) == ArgMove) ? Op::ArrayRefMove :
                                                    Op::ArrayRefCopy));
          output << dst(statement) << src(argument_node) << rhs(statement);
        }
        else if (statement == ArrayRefConst)
        {
          auto argument_node = statement / Arg;
          output << uleb(
            +(((argument_node / Type) == ArgMove) ? Op::ArrayRefMoveConst :
                                                    Op::ArrayRefCopyConst));
          output << dst(statement) << src(argument_node)
                 << uleb(from_chars_sep_v<uint64_t>(statement / Rhs));
        }
        else if (statement == Load)
        {
          output << uleb(+Op::Load) << dst(statement) << rhs(statement);
        }
        else if (statement == Store)
        {
          auto argument_node = statement / Arg;
          output << uleb(
            +(((argument_node / Type) == ArgMove) ? Op::StoreMove :
                                                    Op::StoreCopy));
          output << dst(statement) << src(statement) << src(argument_node);
        }
        else if (statement == Lookup)
        {
          output << uleb(+Op::LookupDynamic) << dst(statement) << src(statement)
                 << mth(statement);
        }
        else if (statement == Call)
        {
          arguments(statement / Args);
          output << uleb(+Op::CallStatic) << dst(statement) << fn(statement);
        }
        else if (statement == MemoSlot)
        {
          auto memo_id = std::string((statement / MemoId)->location().view());
          auto slot = memo_slots.find(memo_id);
          assert(slot != memo_slots.end());
          output << uleb(+Op::MemoLoad) << dst(statement) << uleb(slot->second);
        }
        else if (statement == CallDyn)
        {
          arguments(statement / Args);
          output << uleb(+Op::CallDynamic) << dst(statement) << src(statement);
        }
        else if (statement == TryCallDyn)
        {
          arguments(statement / Args);
          output << uleb(+Op::TryCallDynamic) << dst(statement)
                 << src(statement);
        }
        else if (statement == FFI)
        {
          arguments(statement / Args);
          output << uleb(+Op::FFI) << dst(statement)
                 << uleb(*compilation.get_symbol_id(statement / SymbolId));
        }
        else if (statement == FFIStruct)
        {
          output << uleb(+Op::FFIStruct) << dst(statement)
                 << uleb(compilation.type_id(statement / Type));
        }
        else if (statement == FFILoad)
        {
          output << uleb(+Op::FFILoad) << dst(statement) << lhs(statement)
                 << rhs(statement)
                 << uleb(*function.get_register_id(statement / Kind))
                 << uleb(compilation.type_id(statement / Type));
        }
        else if (statement == FFIStore)
        {
          output << uleb(+Op::FFIStore) << dst(statement) << lhs(statement)
                 << rhs(statement)
                 << uleb(*function.get_register_id(statement / Kind))
                 << uleb(*function.get_register_id(statement / ValueSrc))
                 << uleb(compilation.type_id(statement / Type));
        }
        else if (statement == ArrayCopy)
        {
          arguments(statement / Args);
          output << uleb(+Op::ArrayCopy) << dst(statement);
        }
        else if (statement == ArrayFill)
        {
          arguments(statement / Args);
          output << uleb(+Op::ArrayFill) << dst(statement);
        }
        else if (statement == ArrayCompare)
        {
          arguments(statement / Args);
          output << uleb(+Op::ArrayCompare) << dst(statement);
        }
        else if (statement == When)
        {
          arguments(statement / Args);
          output << uleb(+Op::WhenStatic) << dst(statement)
                 << uleb(compilation.type_id(statement / Cown))
                 << fn(statement);
        }
        else if (statement == WhenDyn)
        {
          arguments(statement / Args);
          output << uleb(+Op::WhenDynamic) << dst(statement)
                 << uleb(compilation.type_id(statement / Cown))
                 << src(statement);
        }
        else if (statement == Add)
          binary(statement, Op::Add);
        else if (statement == Sub)
          binary(statement, Op::Sub);
        else if (statement == Mul)
          binary(statement, Op::Mul);
        else if (statement == Div)
          binary(statement, Op::Div);
        else if (statement == Mod)
          binary(statement, Op::Mod);
        else if (statement == Pow)
          binary(statement, Op::Pow);
        else if (statement == And)
          binary(statement, Op::And);
        else if (statement == Or)
          binary(statement, Op::Or);
        else if (statement == Xor)
          binary(statement, Op::Xor);
        else if (statement == Shl)
          binary(statement, Op::Shl);
        else if (statement == Shr)
          binary(statement, Op::Shr);
        else if (statement == Eq)
          binary(statement, Op::Eq);
        else if (statement == Ne)
          binary(statement, Op::Ne);
        else if (statement == Lt)
          binary(statement, Op::Lt);
        else if (statement == Le)
          binary(statement, Op::Le);
        else if (statement == Gt)
          binary(statement, Op::Gt);
        else if (statement == Ge)
          binary(statement, Op::Ge);
        else if (statement == Min)
          binary(statement, Op::Min);
        else if (statement == Max)
          binary(statement, Op::Max);
        else if (statement == LogBase)
          binary(statement, Op::LogBase);
        else if (statement == Atan2)
          binary(statement, Op::Atan2);
        else if (statement == Neg)
          unary(statement, Op::Neg);
        else if (statement == Not)
          unary(statement, Op::Not);
        else if (statement == Abs)
          unary(statement, Op::Abs);
        else if (statement == Ceil)
          unary(statement, Op::Ceil);
        else if (statement == Floor)
          unary(statement, Op::Floor);
        else if (statement == Exp)
          unary(statement, Op::Exp);
        else if (statement == Log)
          unary(statement, Op::Log);
        else if (statement == Sqrt)
          unary(statement, Op::Sqrt);
        else if (statement == Cbrt)
          unary(statement, Op::Cbrt);
        else if (statement == IsInf)
          unary(statement, Op::IsInf);
        else if (statement == IsNaN)
          unary(statement, Op::IsNaN);
        else if (statement == Sin)
          unary(statement, Op::Sin);
        else if (statement == Cos)
          unary(statement, Op::Cos);
        else if (statement == Tan)
          unary(statement, Op::Tan);
        else if (statement == Asin)
          unary(statement, Op::Asin);
        else if (statement == Acos)
          unary(statement, Op::Acos);
        else if (statement == Atan)
          unary(statement, Op::Atan);
        else if (statement == Sinh)
          unary(statement, Op::Sinh);
        else if (statement == Cosh)
          unary(statement, Op::Cosh);
        else if (statement == Tanh)
          unary(statement, Op::Tanh);
        else if (statement == Asinh)
          unary(statement, Op::Asinh);
        else if (statement == Acosh)
          unary(statement, Op::Acosh);
        else if (statement == Atanh)
          unary(statement, Op::Atanh);
        else if (statement == Bits)
          unary(statement, Op::Bits);
        else if (statement == Len)
          unary(statement, Op::Len);
        else if (statement == MakePtr)
          unary(statement, Op::Ptr);
        else if (statement == Read)
          unary(statement, Op::Read);
        else if (statement == Const_E)
          output << uleb(+Op::Const_E) << dst(statement);
        else if (statement == Const_Pi)
          output << uleb(+Op::Const_Pi) << dst(statement);
        else if (statement == Const_Inf)
          output << uleb(+Op::Const_Inf) << dst(statement);
        else if (statement == Const_NaN)
          output << uleb(+Op::Const_NaN) << dst(statement);
        else if (statement == MakeCallback)
          unary(statement, Op::MakeCallback);
        else if (statement == CodePtrCallback)
          unary(statement, Op::CodePtrCallback);
        else if (statement == FreeCallback)
          unary(statement, Op::FreeCallback);
        else if (statement == Pin)
          unary(statement, Op::Pin);
        else if (statement == Unpin)
          unary(statement, Op::Unpin);
        else if (statement == Merge)
          binary(statement, Op::Merge);
        else if (statement == AddExternal)
          output << uleb(+Op::AddExternal) << dst(statement);
        else if (statement == RemoveExternal)
          output << uleb(+Op::RemoveExternal) << dst(statement);
        else if (statement == Typetest)
        {
          output << uleb(+Op::Typetest) << dst(statement) << src(statement)
                 << uleb(compilation.type_id(statement / Type));
        }
        else if (statement == GetRaise)
          output << uleb(+Op::GetRaise) << dst(statement);
        else if (statement == SetRaise)
          unary(statement, Op::SetRaise);
      }

      void terminator(Node terminator)
      {
        if (terminator == Tailcall)
        {
          arguments(terminator / MoveArgs);
          output << uleb(+Op::TailcallStatic) << fn(terminator);
        }
        else if (terminator == TailcallDyn)
        {
          arguments(terminator / MoveArgs);
          output << uleb(+Op::TailcallDynamic) << dst(terminator);
        }
        else if (terminator == Return)
        {
          output << uleb(+Op::Return) << dst(terminator);
        }
        else if (terminator == Raise)
        {
          output << uleb(+Op::Raise) << dst(terminator);
        }
        else if (terminator == Cond)
        {
          auto true_label = *function.get_label_id(terminator / Lhs);
          auto false_label = *function.get_label_id(terminator / Rhs);
          output << uleb(+Op::Cond) << dst(terminator) << uleb(true_label)
                 << uleb(false_label);
        }
        else if (terminator == Jump)
        {
          output << uleb(+Op::Jump)
                 << uleb(*function.get_label_id(terminator / LabelId));
        }
      }
    };
  }

  void encode_statement(
    Compilation& compilation,
    FuncState& function,
    const MemoSlots& memo_slots,
    ByteBuffer& output,
    Node statement)
  {
    InstructionEncoder{compilation, function, memo_slots, output}.statement(
      statement);
  }

  void encode_terminator(
    Compilation& compilation,
    FuncState& function,
    ByteBuffer& output,
    Node terminator)
  {
    const MemoSlots no_memo_slots;
    InstructionEncoder{compilation, function, no_memo_slots, output}.terminator(
      terminator);
  }
}