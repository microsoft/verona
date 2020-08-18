// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#include "dialect/VeronaTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir::verona::detail
{
  class TypePrinter
  {
  public:
    TypePrinter(llvm::raw_ostream& os) : os(os) {}

    void printTypeList(ArrayRef<Type> types)
    {
      os << "<";
      llvm::interleaveComma(
        types, os, [&](auto element) { printVeronaType(element); });
      os << ">";
    }

    void printClassType(ClassType type)
    {
      os << "class<\"" << type.getName() << "\"";

      if (!class_stack.insert(type.getName()))
      {
        // If the class is already in the stack, then we're already in the
        // process of printing that class. We should skip the body to avoid
        // infinite recursion.
        os << ">";
        return;
      }

      for (auto [field_name, field_type] : type.getFields())
      {
        os << ", \"" << field_name << "\" : ";
        printVeronaType(field_type);
      }
      os << ">";

      class_stack.pop_back();
    }

    void printVeronaType(Type type)
    {
      TypeSwitch<Type>(type)
        .Case<IntegerType>([&](IntegerType type) {
          if (type.getSign())
          {
            os << "S";
          }
          else
          {
            os << "U";
          }
          os << type.getWidth();
        })
        .Case<FloatType>([&](FloatType type) { os << "F" << type.getWidth(); })
        .Case<BoolType>([&](BoolType type) { os << "bool"; })
        .Case<MeetType>([&](MeetType type) {
          if (type.getElements().empty())
          {
            os << "top";
          }
          else
          {
            os << "meet";
            printTypeList(type.getElements());
          }
        })
        .Case<JoinType>([&](JoinType type) {
          if (type.getElements().empty())
          {
            os << "bottom";
          }
          else
          {
            os << "join";
            printTypeList(type.getElements());
          }
        })
        .Case<CapabilityType>([&](CapabilityType type) {
          switch (type.getCapability())
          {
            case Capability::Isolated:
              os << "iso";
              break;
            case Capability::Mutable:
              os << "mut";
              break;
            case Capability::Immutable:
              os << "imm";
              break;
          }
        })
        .Case<ClassType>([&](ClassType type) { printClassType(type); })
        .Case<ViewpointType>([&](ViewpointType type) {
          os << "viewpoint<" << type.getLeftType() << ", "
             << type.getRightType() << ">";
        });
    }

  private:
    llvm::raw_ostream& os;
    llvm::SetVector<StringRef> class_stack;
  };

  class TypeParser
  {
  public:
    TypeParser(DialectAsmParser& parser)
    : context(parser.getBuilder().getContext()), parser(parser)
    {}

    /// Parse a list of types, surrounded by angle brackets and separated by
    /// commas. The types inside the list must be Verona types and should not
    /// use the `!verona.` prefix.
    ///
    /// Empty lists are allowed, but must still use angle brackets, i.e. `< >`.
    /// Lists of one elements are also allowed.
    ParseResult parseTypeList(llvm::SmallVectorImpl<Type>& result)
    {
      if (parser.parseLess())
        return failure();

      if (succeeded(parser.parseOptionalGreater()))
        return success();

      do
      {
        mlir::Type element = parseVeronaType();
        if (!element)
          return failure();

        result.push_back(element);
      } while (succeeded(parser.parseOptionalComma()));

      if (parser.parseGreater())
        return failure();

      return success();
    }

    Type parseMeetType()
    {
      SmallVector<mlir::Type, 2> elements;
      if (parseTypeList(elements))
        return Type();
      return MeetType::get(context, elements);
    }

    Type parseJoinType()
    {
      SmallVector<mlir::Type, 2> elements;
      if (parseTypeList(elements))
        return Type();
      return JoinType::get(context, elements);
    }

    Type parseIntegerType(StringRef keyword)
    {
      size_t width = 0;
      if (keyword.substr(1).getAsInteger(10, width))
      {
        parser.emitError(parser.getNameLoc(), "unknown verona type: ")
          << keyword;
        return Type();
      }
      bool sign = keyword.startswith("S");
      return IntegerType::get(context, width, sign);
    }

    Type parseFloatType(StringRef keyword)
    {
      size_t width = 0;
      if (keyword.substr(1).getAsInteger(10, width))
      {
        parser.emitError(parser.getNameLoc(), "unknown verona type: ")
          << keyword;
        return Type();
      }
      return FloatType::get(context, width);
    }

    // Annoyingly, DialectAsmParser only exposes `parseOptionalString`, no
    // `parseString`. This method implements the latter based on the former.
    ParseResult parseString(StringRef* value)
    {
      auto loc = parser.getCurrentLocation();
      if (failed(parser.parseOptionalString(value)))
        return parser.emitError(loc, "expected string literal");
      else
        return success();
    }

    Type parseClassType()
    {
      auto loc = parser.getNameLoc();

      StringRef name;
      SmallVector<std::pair<StringRef, Type>, 4> fields;

      if (parser.parseLess() || parseString(&name))
        return Type();

      // We try to insert this class into the stack of pending classes, such
      // that recursive occurences may omit the body.
      if (!class_stack.insert(name))
      {
        // If the class is already in the stack, then we're already in the
        // process of parsing that class. We should not parse the body and
        // instead return a (maybe) incomplete type.
        //
        // The type's body will be completed when the outer parsing is done.
        if (parser.parseGreater())
          return Type();

        return ClassType::get(context, name);
      }

      while (succeeded(parser.parseOptionalComma()))
      {
        StringRef field_name;
        Type field_type;
        if (
          parseString(&field_name) || parser.parseColon() ||
          !(field_type = parseVeronaType()))
          return Type();

        fields.push_back({field_name, field_type});
      }

      if (parser.parseGreater())
        return Type();

      class_stack.pop_back();

      ClassType pending = ClassType::get(context, name);
      if (failed(pending.setFields(fields)))
      {
        InFlightDiagnostic diag = parser.emitError(loc)
          << "class type \"" << pending.getName()
          << "\" already used with different definition";
        return Type();
      }

      return pending;
    }

    Type parseViewpointType()
    {
      Type left;
      Type right;
      if (
        parser.parseLess() || !(left = parseVeronaType()) ||
        parser.parseComma() || !(right = parseVeronaType()) ||
        parser.parseGreater())
        return Type();

      return ViewpointType::get(context, left, right);
    }

    Type parseVeronaType()
    {
      StringRef keyword;
      if (parser.parseKeyword(&keyword))
        return Type();

      if (keyword == "class")
        return parseClassType();
      else if (keyword == "meet")
        return parseMeetType();
      else if (keyword == "join")
        return parseJoinType();
      else if (keyword == "viewpoint")
        return parseViewpointType();
      else if (keyword == "top")
        return MeetType::get(context, {});
      else if (keyword == "bottom")
        return JoinType::get(context, {});
      else if (keyword == "iso")
        return CapabilityType::get(context, Capability::Isolated);
      else if (keyword == "mut")
        return CapabilityType::get(context, Capability::Mutable);
      else if (keyword == "imm")
        return CapabilityType::get(context, Capability::Immutable);
      else if (keyword == "bool")
        return BoolType::get(context);
      else if (keyword.startswith("U") || keyword.startswith("S"))
        return parseIntegerType(keyword);
      else if (keyword.startswith("F"))
        return parseFloatType(keyword);

      parser.emitError(parser.getNameLoc(), "unknown verona type: ") << keyword;
      return Type();
    }

  private:
    MLIRContext* context;
    DialectAsmParser& parser;
    llvm::SetVector<StringRef> class_stack;
  };
}

namespace mlir::verona
{
  void printVeronaType(Type type, DialectAsmPrinter& os)
  {
    return detail::TypePrinter(os.getStream()).printVeronaType(type);
  }

  Type parseVeronaType(DialectAsmParser& parser)
  {
    return detail::TypeParser(parser).parseVeronaType();
  }
}
