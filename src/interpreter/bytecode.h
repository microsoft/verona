// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>
#include <cstdlib>
#include <fmt/format.h>
#include <limits>
#include <ostream>
#include <string_view>
#include <vector>

/**
 * # Program Layout
 *
 * Bytecode has the following layout:
 *
 * Program header:
 * - 32-bit verona magic number, equal to the MAGIC_NUMBER constant
 * - 32-bit number of descriptors, followed by that many descriptors (see below)
 * - 32-bit descriptor index of Main class
 * - 32-bit selector index of main method
 * - 32-bit descriptor index of U64 class (optional, ~0 when absent)
 * - 32-bit descriptor index of String class (optional, ~0 when absent)
 *
 * Descriptor:
 * - 16-bit name length, followed by the name bytes
 * - 32-bit size of the fields vtable
 * - 32-bit number of fields
 * - 32-bit size of the methods vtables
 * - 32-bit number of methods
 * - 32-bit number of subtypes
 * - 32-bit offset to finaliser
 * - For each field, 32-bit selector index
 * - For each method, 32-bit selector index and 32-bit absolute offset
 * - For each subtype, 32-bit descriptor index.
 *
 * Methods:
 * - 16-bit name length, followed by the name bytes
 * - 8-bit number of arguments
 * - 8-bit number of return values (currently always 1)
 * - 8-bit number of local variables (must be greater than both the number of
 *   arguments and return values)
 * - 32-bit size of the method body, followed by that number of bytes of
 *   instruction data
 *
 * All integers are little-endian format. There is no padding or alignment
 * anywhere.
 *
 * # Instruction encoding
 *
 * Instructions are encoded using a variable length format. Each instruction
 * starts with a one byte opcode number, followed by zero or more operands.
 *
 * Some instructions will therefore be just one byte long (e.g. Return) whereas
 * others may occupy many bytes (e.g. Store is 8 bytes).
 *
 * # Register machine
 *
 * Instructions operate over registers. These are encoded as a single byte,
 * allowing up to 256 accessible registers. The registers however are organized
 * in a larger register file, with each call frame having a different, but
 * possibly overlapping, view of the file.
 *
 * In the diagram below, the register file contains 6 registers. The call stack
 * has two frames, A and B. Frame A's view of the register file ranges spans
 * from registers 0 to 3, while Frame B's view spans from registers 2 to 5.
 *
 *    -------  <----- Frame B's top
 * 5: |     |
 *    -------
 * 4: |     |
 *    -------  <----- Frame A's top
 * 3: |     |
 *    -------
 * 2: |     |
 *    -------  <----- Frame B's base
 * 1: |     |
 *    -------
 * 0: |     |
 *    -------  <----- Frame A's base
 *
 * Registers indices specified in encoded instructions are relative to the
 * current frame. Thus in the example above, as long as the VM is executing the
 * current method, which corresponds to call frame B, a register index of 1
 * actually refers to register 3 in the register file. Once the method returns,
 * a register index of 1 would now refer to register 1 in the file.
 *
 * # Calling convention
 *
 * The overlap between views of the register file is used to pass arguments and
 * return values during method calls. The values are placed starting at the
 * callee's register 0.
 *
 * The Call opcode takes a "call space" immediate operand, which describes the
 * size of the overlap. This is typically the maximum of the number of arguments
 * and return values of the method being called. The callee's frame's view of
 * the file begins at `call-space` registers lower than the current frame's top.
 * The size of the view depends on the number of locals speicifed in the
 * method's header.
 *
 * The register file described earlier would have been the result of calling a 2
 * argument method. The caller would have copied the argument values into it's
 * registers 2 and 3, then invoked the Call opcode with a callspace of 2.
 *
 * To dispatch the actual call, the VM uses the first argument to find a vtable,
 * which is indexed using the selector index passed to the opcode as an
 * immediate operand.
 *
 * # When opcode
 *
 * Using the When opcode requires setting up the frame in the exact same way as
 * for a method call. The very first argument is a dummy receiver. The next n
 * arguments are the cown, and the subsequent m arguments are the captured
 * variables. Rather than immediately creating a new frame with an overlapping
 * view, the VM saves the arguments into the multi-message sent to the cowns.
 *
 * When the message eventually gets executed, a frame is created in a new empty
 * call stack and the arguments are restored onto that VM instance's register
 * file.
 *
 */
namespace verona::bytecode
{
  struct FunctionHeader
  {
    std::string_view name;
    uint8_t argc;
    uint8_t retc;
    uint8_t locals;
    uint32_t size;
  };

  /**
   * Magic number which occurs at the beginning of every Verona bytecode file.
   * It allows the interpreter to bail out when the user obviously tried to run
   * the wrong file (eg. a Verona source code).
   */
  constexpr static uint32_t MAGIC_NUMBER = 0xF38932C3;

  template<typename U>
  struct Wrapper;

  /**
   * This special instantiation of Wrapper is used as a base class for all
   * others. It's purpose is to allow checking whether a type is any
   * instantiation of Wrapper, through the is_wrapper_v trait.
   */
  template<>
  struct Wrapper<void>
  {};

  template<typename T>
  inline constexpr bool is_wrapper_v = std::is_base_of_v<Wrapper<void>, T>;

  /**
   * Type safe wrapper around integers. Used for values that get encoded into
   * the bytecode. This helps avoid implicit conversion from/to integers, as
   * well as confusion between different kinds of integer values (eg. absolute
   * vs relative offsets).
   */
  template<typename U>
  struct Wrapper : public Wrapper<void>
  {
    using underlying_type = U;
    explicit Wrapper(U value) : value(value) {}
    U value;
  };

  struct Register : public Wrapper<uint8_t>
  {
    using Wrapper<uint8_t>::Wrapper;
  };

  /**
   * Index used to identify a method or field. These indices are used as offset
   * into object vtables.
   */
  struct SelectorIdx : public Wrapper<uint32_t>
  {
    using Wrapper<uint32_t>::Wrapper;
  };

  /**
   * Index used to identify a class/interface/primitive type. The index refers
   * to the position of that type in the program's list of descriptor.
   */
  struct DescriptorIdx : public Wrapper<uint32_t>
  {
    using Wrapper<uint32_t>::Wrapper;

    /**
     * Placeholder descriptor value used in places where a descriptor is
     * optional, eg. the descriptor index of the U64 class, in the program
     * header.
     */
    static DescriptorIdx invalid()
    {
      uint32_t value = std::numeric_limits<uint32_t>::max();
      return DescriptorIdx(value);
    }
  };

  /**
   * Absolute offset into the program, in bytes.
   */
  struct AbsoluteOffset : public Wrapper<uint32_t>
  {
    using Wrapper<uint32_t>::Wrapper;
  };

  /**
   * Offset into the program that is relative to the start of the current
   * instruction, in bytes.
   */
  struct RelativeOffset : public Wrapper<int16_t>
  {
    using Wrapper<int16_t>::Wrapper;
  };

  /**
   * Contiguous sequence of Register values. It is used to encode and decode
   * opcodes that accept a variable number of operands.
   *
   * This is a restricted subset of a C++20 `std::span<const Register>`
   *
   * When emitted into the bytecode, this is represented as 8-bit length
   * followed by one byte per register in the list.
   */
  class RegisterSpan
  {
  public:
    explicit RegisterSpan(const Register* begin, const Register* end)
    : begin_(begin), end_(end)
    {}

    explicit RegisterSpan(const Register* begin, size_t size)
    : RegisterSpan(begin, begin + size)
    {}

    RegisterSpan(const std::vector<Register>& regs)
    : RegisterSpan(regs.data(), regs.size())
    {}

    RegisterSpan(const std::initializer_list<Register>& regs)
    : RegisterSpan(regs.begin(), regs.end())
    {}

    size_t size() const
    {
      return end_ - begin_;
    }

    const Register* begin() const
    {
      return begin_;
    }

    const Register* end() const
    {
      return end_;
    }

  private:
    const Register* begin_;
    const Register* end_;
  };

  constexpr static size_t REGISTER_COUNT = 256;

  enum class Opcode : uint8_t
  {
    BinOp, // op(u8), src1(u8), src2(u8)
    Call, // selector(u32), callspace(u8)
    Clear, // dst(u8)
    ClearList, // argc(u8), dst(u8)...
    Copy, // dst(u8), src(u8)
    FulfillSleepingCown, // cown(u8), val(u8)
    Freeze, // dst(u8), src(u8)
    Int64, // dst(u8), immediate(u64)
    String, // dst(u8), immediate(str)
    Jump, // target(u16)
    JumpIf, // src(u8), target(u16)
    Load, // dst(u8), base(u8), selector(u32)
    LoadDescriptor, // dst(u8), descriptor_id(u32)
    MatchCapability, // dst(u8), src(u8), cap(u8)
    MatchDescriptor, // dst(u8), src(u8), descriptor(u8)
    Merge, // into(u8), src(u8)
    Move, // dst(u8), src(u8)
    MutView, // dst(u8), src(u8)
    NewObject, // dst(u8), region(u8), descriptor(u8)
    NewCown, // dst(u8), descriptor(u8), src(u8)
    NewRegion, // dst(u8), descriptor(u8)
    NewSleepingCown, // dst(u8), descriptor(u8)
    Print, // format(u8), argc(u8), args(u8)...
    Protect, // argc(u8), args(u8)...
    Return,
    Store, // dst(u8), base(u8), selector(u32), src(u8)
    TraceRegion, // region(u8)
    Unprotect, // argc(u8), args(u8)...
    Unreachable,
    When, // codepointer(u32), cown count(u8), capture count(u8)

    maximum_value = When,
  };

  enum class BinaryOperator : uint8_t
  {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Shl,
    Shr,
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
    And,
    Or,

    maximum_value = Or,
  };

  enum class Capability : uint8_t
  {
    Iso,
    Mut,
    Imm,

    maximum_value = Imm,
  };

  template<typename... Args>
  struct OpcodeOperands
  {};

  /**
   * Operand specification.
   *
   * A specification is defined through specialization for each Opcode value.
   *
   * Each specialization provides a `Operands` type alias and a format string
   * used to print the decoded instruction. The `Operands` definition drives
   * both the `emit` DSL used by the code generator as well as the dispatch code
   * in the VM.
   */
  template<Opcode opcode>
  struct OpcodeSpec;

  template<>
  struct OpcodeSpec<Opcode::BinOp>
  {
    using Operands =
      OpcodeOperands<Register, BinaryOperator, Register, Register>;
    constexpr static std::string_view format = "{1} {0}, {2}, {3}";
  };

  template<>
  struct OpcodeSpec<Opcode::Call>
  {
    using Operands = OpcodeOperands<SelectorIdx, uint8_t>;
    constexpr static std::string_view format = "CALL {}, {:#x}";
  };

  template<>
  struct OpcodeSpec<Opcode::Clear>
  {
    using Operands = OpcodeOperands<Register>;
    constexpr static std::string_view format = "CLEAR {}";
  };

  template<>
  struct OpcodeSpec<Opcode::ClearList>
  {
    using Operands = OpcodeOperands<RegisterSpan>;
    constexpr static std::string_view format = "CLEAR_LIST {}";
  };

  template<>
  struct OpcodeSpec<Opcode::Copy>
  {
    using Operands = OpcodeOperands<Register, Register>;
    constexpr static std::string_view format = "COPY {}, {}";
  };

  template<>
  struct OpcodeSpec<Opcode::FulfillSleepingCown>
  {
    using Operands = OpcodeOperands<Register, Register>;
    constexpr static std::string_view format = "FULFILL {}, {}";
  };

  template<>
  struct OpcodeSpec<Opcode::Freeze>
  {
    using Operands = OpcodeOperands<Register, Register>;
    constexpr static std::string_view format = "FREEZE {}, {}";
  };

  template<>
  struct OpcodeSpec<Opcode::Int64>
  {
    using Operands = OpcodeOperands<Register, uint64_t>;
    constexpr static std::string_view format = "INT64 {}, {:#x}";
  };

  template<>
  struct OpcodeSpec<Opcode::Jump>
  {
    using Operands = OpcodeOperands<RelativeOffset>;
    constexpr static std::string_view format = "JUMP {:+#x}";
  };

  template<>
  struct OpcodeSpec<Opcode::JumpIf>
  {
    using Operands = OpcodeOperands<Register, RelativeOffset>;
    constexpr static std::string_view format = "JUMP_IF {}, {:+#x}";
  };

  template<>
  struct OpcodeSpec<Opcode::Load>
  {
    using Operands = OpcodeOperands<Register, Register, SelectorIdx>;
    constexpr static std::string_view format = "LOAD {}, {}[{:#x}]";
  };

  template<>
  struct OpcodeSpec<Opcode::LoadDescriptor>
  {
    using Operands = OpcodeOperands<Register, DescriptorIdx>;
    constexpr static std::string_view format = "LOAD_DESCRIPTOR {}, {:#x}";
  };

  template<>
  struct OpcodeSpec<Opcode::MatchCapability>
  {
    using Operands = OpcodeOperands<Register, Register, Capability>;
    constexpr static std::string_view format = "MATCH_CAPABILITY {}, {}, {}";
  };

  template<>
  struct OpcodeSpec<Opcode::MatchDescriptor>
  {
    using Operands = OpcodeOperands<Register, Register, Register>;
    constexpr static std::string_view format = "MATCH_DESCRIPTOR {}, {}, {}";
  };

  template<>
  struct OpcodeSpec<Opcode::Move>
  {
    using Operands = OpcodeOperands<Register, Register>;
    constexpr static std::string_view format = "MOVE {}, {}";
  };

  template<>
  struct OpcodeSpec<Opcode::MutView>
  {
    using Operands = OpcodeOperands<Register, Register>;
    constexpr static std::string_view format = "MUT-VIEW {}, {}";
  };

  template<>
  struct OpcodeSpec<Opcode::NewObject>
  {
    using Operands = OpcodeOperands<Register, Register, Register>;
    constexpr static std::string_view format = "NEW_OBJECT {}, {}, {}";
  };

  template<>
  struct OpcodeSpec<Opcode::NewRegion>
  {
    using Operands = OpcodeOperands<Register, Register>;
    constexpr static std::string_view format = "NEW_REGION {}, {}";
  };

  template<>
  struct OpcodeSpec<Opcode::NewCown>
  {
    using Operands = OpcodeOperands<Register, Register, Register>;
    constexpr static std::string_view format = "NEW_COWN {}, {}, {}";
  };

  template<>
  struct OpcodeSpec<Opcode::NewSleepingCown>
  {
    using Operands = OpcodeOperands<Register, Register>;
    constexpr static std::string_view format = "NEW_SLEEPING_COWN {} {}";
  };

  template<>
  struct OpcodeSpec<Opcode::Print>
  {
    using Operands = OpcodeOperands<Register, RegisterSpan>;
    constexpr static std::string_view format = "PRINT {}, {}";
  };

  template<>
  struct OpcodeSpec<Opcode::Protect>
  {
    using Operands = OpcodeOperands<RegisterSpan>;
    constexpr static std::string_view format = "PROTECT {}";
  };

  template<>
  struct OpcodeSpec<Opcode::Store>
  {
    using Operands = OpcodeOperands<Register, Register, SelectorIdx, Register>;
    constexpr static std::string_view format = "STORE {}, {}[{:#x}], {}";
  };

  template<>
  struct OpcodeSpec<Opcode::Return>
  {
    using Operands = OpcodeOperands<>;
    constexpr static std::string_view format = "RETURN";
  };

  template<>
  struct OpcodeSpec<Opcode::String>
  {
    using Operands = OpcodeOperands<Register, std::string_view>;
    constexpr static std::string_view format = "STRING {}, \"{}\"";
  };

  template<>
  struct OpcodeSpec<Opcode::TraceRegion>
  {
    using Operands = OpcodeOperands<Register>;
    constexpr static std::string_view format = "TRACE REGION {}";
  };

  template<>
  struct OpcodeSpec<Opcode::When>
  {
    using Operands = OpcodeOperands<AbsoluteOffset, uint8_t, uint8_t>;
    constexpr static std::string_view format = "WHEN {}, {:#x}, {:#x}";
  };

  template<>
  struct OpcodeSpec<Opcode::Unprotect>
  {
    using Operands = OpcodeOperands<RegisterSpan>;
    constexpr static std::string_view format = "UNPROTECT {}";
  };

  template<>
  struct OpcodeSpec<Opcode::Unreachable>
  {
    using Operands = OpcodeOperands<>;
    constexpr static std::string_view format = "UNREACHABLE";
  };

  std::ostream& operator<<(std::ostream& out, const BinaryOperator& self);
  std::ostream& operator<<(std::ostream& out, const Capability& self);

  template<typename T>
  std::enable_if_t<bytecode::is_wrapper_v<T>, bool>
  operator==(const T& lhs, const T& rhs)
  {
    return lhs.value == rhs.value;
  }
}

namespace std
{
  // Allow DescriptorIdx to be hashed, using the underlying value. Ideally we
  // would do this for all wrapper types at once, but std::hash cannot be
  // conditionally specialized with an enable_if.
  template<>
  struct hash<verona::bytecode::DescriptorIdx>
  {
    size_t operator()(const verona::bytecode::DescriptorIdx& idx) const
    {
      using underlying_type = verona::bytecode::DescriptorIdx::underlying_type;
      return std::hash<underlying_type>()(idx.value);
    }
  };
}

namespace fmt
{
  // Allow wrapper types to be formatted, by forwarding to the formatter for
  // the underlying type. Doing this, rather than overloading operator<<,
  // enables fmtlib format specifiers to work.
  template<typename T>
  struct formatter<T, char, std::enable_if_t<verona::bytecode::is_wrapper_v<T>>>
  {
    constexpr auto parse(format_parse_context& ctx)
    {
      return underlying.parse(ctx);
    }

    auto format(const T& wrapper, format_context& ctx)
    {
      return underlying.format(wrapper.value, ctx);
    }

    formatter<typename T::underlying_type> underlying;
  };

  // Override the above formatter for the Register type: as a convention, we
  // always print them as rX, where X is the register's index.
  template<>
  struct formatter<verona::bytecode::Register>
  {
    constexpr auto parse(format_parse_context& ctx)
    {
      return ctx.begin();
    }

    auto format(const verona::bytecode::Register& reg, format_context& ctx)
    {
      return format_to(ctx.out(), "r{}", reg.value);
    }
  };
}
