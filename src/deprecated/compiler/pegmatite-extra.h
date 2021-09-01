// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once
#include <ast.hh>
#include <pegmatite.hh>

namespace pegmatite
{
  struct ASTInteger : public ASTNode
  {
    bool
    construct(const InputRange& r, ASTStack&, const ErrorReporter&) override
    {
      std::stringstream stream;
      std::for_each(r.begin(), r.end(), [&](char32_t c) {
        stream << static_cast<char>(c);
      });
      std::string s = stream.str();
      size_t end = 0;
      value = std::stoi(s, &end);

      return true;
    }

    uint64_t value;
  };

  template<typename T>
  struct ASTConstant : public ASTNode
  {
    bool
    construct(const InputRange& r, ASTStack&, const ErrorReporter&) override
    {
      return true;
    }
    virtual T value() = 0;
  };

  template<typename T, T v>
  struct ASTConstantImpl : public ASTConstant<T>
  {
    T value() final
    {
      return v;
    }
  };

  template<typename T, T v>
  using BindConstant = BindAST<ASTConstantImpl<T, v>>;
}
