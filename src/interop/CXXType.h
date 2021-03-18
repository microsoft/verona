// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include <clang/Sema/Sema.h>

namespace verona::interop
{
  /**
   * C++ types that can be queried from the AST matchers.
   *
   * This is basically a structure to hold type information rather than a type
   * on its own right. Some type logic (ex. type size & alignment) need the
   * compiler to know the ABI (sizes, layout, packing). That logic is currently
   * in CXXInterface, which creates the compiler calls and has other type logic.
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

    /// Return false if the type is invalid
    bool valid() const
    {
      return kind != Kind::Invalid;
    }

    /// Returns true if the type has a name declaration
    bool hasNameDecl() const
    {
      assert(decl);
      return kind != Kind::Invalid && kind != Kind::Builtin;
    }

    /// Returns true if the type is a class.
    bool isClass() const
    {
      return kind == Kind::Class || isTemplate();
    }

    /// Returns true if the type is templated.
    bool isTemplate() const
    {
      return kind == Kind::TemplateClass ||
        kind == Kind::SpecializedTemplateClass;
    }

    /// Returns true if type is integral
    /// TODO: Should we make all these helpers static?
    static bool isIntegral(BuiltinTypeKinds ty)
    {
      return ty != BuiltinTypeKinds::Float && ty != BuiltinTypeKinds::Double;
    }

    /// CXXType builtin c-tor
    CXXType(BuiltinTypeKinds t) : kind(Kind::Builtin), builtTypeKind(t) {}
    /// CXXType class c-tor
    CXXType(const clang::CXXRecordDecl* d) : kind(Kind::Class), decl(d) {}
    /// CXXType template class c-tor
    CXXType(const clang::ClassTemplateDecl* d)
    : kind(Kind::TemplateClass), decl(d)
    {}
    /// CXXType template specialisation class c-tor
    CXXType(const clang::ClassTemplateSpecializationDecl* d)
    : kind(Kind::SpecializedTemplateClass), decl(d)
    {}
    /// CXXType enum c-tor
    CXXType(const clang::EnumDecl* d) : kind(Kind::Enum), decl(d) {}
    /// CXXType empty c-tor (kind = Invalid)
    CXXType() = default;

    /// Get a reference to the type name
    llvm::StringRef getName() const
    {
      assert(hasNameDecl());
      return decl->getName();
    }

    /// Return the template parameters, if class is a template.
    const clang::TemplateParameterList* getTemplateParameters()
    {
      assert(isTemplate());
      if (auto ty = getAs<clang::ClassTemplateDecl>())
        return ty->getTemplateParameters();
      if (auto ty = getAs<clang::ClassTemplateSpecializationDecl>())
        return ty->getDescribedTemplateParams();
      return nullptr;
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
      return clang::dyn_cast<T>(const_cast<clang::NamedDecl*>(decl));
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
      const clang::NamedDecl* decl = nullptr;
      /**
       * The kind if this is a builtin.
       */
      BuiltinTypeKinds builtTypeKind;
    };

    // ================================ Builtin helpers
    /// Return a boolean type
    static CXXType getBoolean()
    {
      return CXXType{BuiltinTypeKinds::Bool};
    }
    /// Return an unsigned char type
    static CXXType getUnsignedChar()
    {
      return CXXType{BuiltinTypeKinds::UChar};
    }
    /// Return a char type
    static CXXType getChar()
    {
      return CXXType{BuiltinTypeKinds::Char};
    }
    /// Return a signed char type
    static CXXType getSignedChar()
    {
      return CXXType{BuiltinTypeKinds::SChar};
    }
    /// Return a signed short type
    static CXXType getShort()
    {
      return CXXType{BuiltinTypeKinds::Short};
    }
    /// Return an unsigned short type
    static CXXType getUnsignedShort()
    {
      return CXXType{BuiltinTypeKinds::UShort};
    }
    /// Return a signed int type
    static CXXType getInt()
    {
      return CXXType{BuiltinTypeKinds::Int};
    }
    /// Return an unsigned int type
    static CXXType getUnsignedInt()
    {
      return CXXType{BuiltinTypeKinds::UInt};
    }
    /// Return a signed long type
    static CXXType getLong()
    {
      return CXXType{BuiltinTypeKinds::Long};
    }
    /// Return an unsigned long type
    static CXXType getUnsignedLong()
    {
      return CXXType{BuiltinTypeKinds::ULong};
    }
    /// Return a signed long long type
    static CXXType getLongLong()
    {
      return CXXType{BuiltinTypeKinds::LongLong};
    }
    /// Return an unsigned long long type
    static CXXType getUnsignedLongLong()
    {
      return CXXType{BuiltinTypeKinds::ULongLong};
    }
    /// Return a float type
    static CXXType getFloat()
    {
      return CXXType{BuiltinTypeKinds::Float};
    }
    /// Return a double type
    static CXXType getDouble()
    {
      return CXXType{BuiltinTypeKinds::Double};
    }
  };
} // namespace verona::interop
