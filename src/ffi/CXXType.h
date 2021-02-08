// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include <clang/Sema/Sema.h>

using namespace clang;
namespace verona::ffi
{
  /**
   * C++ types that can be queried from the AST matchers.
   *
   * These are not all types and may not even be the right way to represent it
   * but they'll do for the time being.
   */
  struct CXXType
  {
    /// Match kinds.
    enum class Kind
    {
      Invalid,
      TemplateClass,
      SpecializedTemplateClass,
      Class,
      Enum,
      Builtin
    } kind = Kind::Invalid;

    /// C++ builtin types.
    enum class BuiltinTypeKinds
    {
      Bool,
      SChar,
      Char,
      UChar,
      Short,
      UShort,
      Int,
      UInt,
      Long,
      ULong,
      LongLong,
      ULongLong,
      Float,
      Double
    }; // builtTypeKind

    /// Converts kind name to string.
    const char* kindName()
    {
      switch (kind)
      {
        case Kind::Invalid:
          return "Invalid";
        case Kind::SpecializedTemplateClass:
          return "Specialized Class Template";
        case Kind::TemplateClass:
          return "Class Template";
        case Kind::Class:
          return "Class";
        case Kind::Enum:
          return "Enum";
        case Kind::Builtin:
          return "Builtin";
      }
      return nullptr;
    }

    /// Converts builtin kind name to string.
    const char* builtinKindName()
    {
      assert(kind == Kind::Builtin);
      switch (builtTypeKind)
      {
        case BuiltinTypeKinds::Bool:
          return "bool";
        case BuiltinTypeKinds::SChar:
          return "signed char";
        case BuiltinTypeKinds::Char:
          return "char";
        case BuiltinTypeKinds::UChar:
          return "unsigned char";
        case BuiltinTypeKinds::Short:
          return "short";
        case BuiltinTypeKinds::UShort:
          return "unsigned short";
        case BuiltinTypeKinds::Int:
          return "int";
        case BuiltinTypeKinds::UInt:
          return "unsigned int";
        case BuiltinTypeKinds::Long:
          return "long";
        case BuiltinTypeKinds::ULong:
          return "unsigned long";
        case BuiltinTypeKinds::LongLong:
          return "long long";
        case BuiltinTypeKinds::ULongLong:
          return "unsigned long long";
        case BuiltinTypeKinds::Float:
          return "float";
        case BuiltinTypeKinds::Double:
          return "double";
      }
      return nullptr;
    }

    /// Returns true if the type has a name declaration
    bool hasNameDecl()
    {
      return kind != Kind::Invalid && kind != Kind::Builtin;
    }

    /// Returns true if the type is templated.
    bool isTemplate()
    {
      return kind == Kind::TemplateClass;
    }

    /// Returns true if type is integral
    static bool isIntegral(BuiltinTypeKinds ty)
    {
      return ty != BuiltinTypeKinds::Float && ty != BuiltinTypeKinds::Double;
    }

    /// CXXType builtin c-tor
    CXXType(BuiltinTypeKinds t) : kind(Kind::Builtin), builtTypeKind(t) {}
    /// CXXType class c-tor
    CXXType(const CXXRecordDecl* d) : kind(Kind::Class), decl(d) {}
    /// CXXType template class c-tor
    CXXType(const ClassTemplateDecl* d) : kind(Kind::TemplateClass), decl(d) {}
    /// CXXType template specialisation class c-tor
    CXXType(const ClassTemplateSpecializationDecl* d)
    : kind(Kind::SpecializedTemplateClass), decl(d)
    {}
    /// CXXType enum c-tor
    CXXType(const EnumDecl* d) : kind(Kind::Enum), decl(d) {}
    /// CXXType empty c-tor (kind = Invalid)
    CXXType() = default;

    /// Returns the number of template parameter, if class is a template.
    int numberOfTemplateParameters()
    {
      if (!isTemplate())
      {
        return 0;
      }
      return getAs<ClassTemplateDecl>()->getTemplateParameters()->size();
    }

    /**
     * Access the underlying decl as the specified type.  This removes the
     * `const` qualification, allowing the AST to be modified, and should be
     * used only by the Clang AST interface classes.
     */
    template<class T>
    T* getAs()
    {
      assert(hasNameDecl());
      return dyn_cast<T>(const_cast<NamedDecl*>(decl));
    }

    /**
     * The size and alignment of this type.  Note that this is not valid for
     * templated types that are not fully specified.
     */
    clang::TypeInfo sizeAndAlign;

    /**
     * Only one is ever present at a time.
     */
    union
    {
      /**
       * The declaration that corresponds to this type.
       */
      const NamedDecl* decl = nullptr;
      /**
       * The kind if this is a builtin.
       */
      BuiltinTypeKinds builtTypeKind;
    };
  };
} // namespace verona::ffi
