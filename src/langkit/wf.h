// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#include "ast.h"

#include <set>
#include <tuple>

namespace langkit
{
  namespace detail
  {
    struct Undef
    {};

    template<size_t N>
    struct Field
    {
      Token name;
      std::array<Token, N> types;

      constexpr Field(const Token& name, const std::array<Token, N>& types)
      : name(name), types(types)
      {}
    };

    template<size_t N>
    struct Sequence : Undef
    {
      std::array<Token, N> tokens;

      constexpr Sequence(const std::array<Token, N>& tokens) : tokens(tokens) {}
    };

    template<typename... Ts>
    struct Shape
    {
      Token type;
      std::tuple<Ts...> fields;

      constexpr Shape(const Token& type, const std::tuple<Ts...>& fields)
      : type(type), fields(fields)
      {}
    };

    template<typename... Ts>
    struct Wellformed
    {
      std::tuple<Ts...> shapes;

      constexpr Wellformed(const std::tuple<Ts...>& shapes) : shapes(shapes) {}
    };

    template<size_t I = 0, typename... Ts>
    inline constexpr Index index(const Shape<Ts...>& shape, const Token& name)
    {
      if constexpr (I < sizeof...(Ts))
      {
        auto field = std::get<I>(shape.fields);

        if constexpr (std::is_base_of_v<Undef, decltype(field)>)
        {
          return {};
        }
        else if (name == field.name)
        {
          return Index(shape.type, I);
        }
        else
        {
          return index<I + 1>(shape, name);
        }
      }

      return {};
    }

    template<size_t I = 0, typename... Ts>
    inline constexpr Index
    index(const Wellformed<Ts...>& wf, const Token& type, const Token& name)
    {
      if constexpr (I < sizeof...(Ts))
      {
        auto shape = std::get<I>(wf.shapes);

        if (type == shape.type)
          return index<0>(shape, name);

        return index<I + 1>(wf, type, name);
      }

      return {};
    }

    enum class field
    {
      undef,
      field,
      sequence,
    };

    struct FieldCheck
    {
      field field;
      std::set<Token> set;
    };

    using ShapeCheck = std::vector<FieldCheck>;
    using WFCheck = std::map<Token, ShapeCheck>;

    inline FieldCheck field_set(const Undef& undef)
    {
      return {field::undef, {}};
    }

    template<size_t N>
    inline FieldCheck field_set(const Field<N>& field)
    {
      return {field::field, {field.types.begin(), field.types.end()}};
    }

    template<size_t N>
    inline FieldCheck field_set(const Sequence<N>& seq)
    {
      return {field::sequence, {seq.tokens.begin(), seq.tokens.end()}};
    }

    template<size_t I, typename... Ts>
    inline void shape_vector_i(const Shape<Ts...>& shape, ShapeCheck& defs)
    {
      if constexpr (I == sizeof...(Ts))
        return;

      defs.push_back(field_set(std::get<I>(shape.fields)));
      shape_vector_i<I + 1>(shape, defs);
    }

    template<typename... Ts>
    inline ShapeCheck shape_vector(const Shape<Ts...>& shape)
    {
      ShapeCheck defs;
      shape_vector_i<0>(shape, defs);
      return defs;
    }

    template<size_t I, typename... Ts>
    inline void wellformed_map_i(const Wellformed<Ts...>& wf, WFCheck& defs)
    {
      if constexpr (I == sizeof...(Ts))
        return;

      auto shape = std::get<I>(wf.shapes);
      defs[shape.type] = shape_vector(shape);
      wellformed_map_i<I + 1>(wf, defs);
    }

    template<typename... Ts>
    inline WFCheck wellformed_map(const Wellformed<Ts...>& wf)
    {
      WFCheck defs;
      wellformed_map_i<0>(wf, defs);
      return defs;
    }

    inline bool check(WFCheck& map, Node node)
    {
      auto find = map.find(node->type());

      if (find == map.end())
        return false;

      auto& defs = find->second;
      auto it = defs.begin();

      for (auto& child : *node)
      {
        if (it == defs.end())
          return false;

        if (it->field == field::undef)
          return true;

        auto find = it->set.find(child->type());
        if (find == it->set.end())
          return false;

        if (!check(map, child))
          return false;

        if (it->field == field::field)
          ++it;
      }

      if (it != defs.end())
      {
        if (((it + 1) != defs.end()) || (it->field != field::sequence))
          return false;
      }

      return true;
    }

    template<typename... Ts>
    inline bool check(const Wellformed<Ts...>& wf, Node node)
    {
      auto map = wellformed_map(wf);
      return check(map, node);
    }
  }

  inline constexpr auto undef()
  {
    return detail::Undef{};
  }

  template<typename... Ts>
  inline constexpr auto field(const Token& name, const Ts&... types)
  {
    if constexpr (sizeof...(Ts) == 0)
    {
      std::array<Token, 1> arr = {name};
      return detail::Field{name, arr};
    }
    else
    {
      std::array<Token, sizeof...(Ts)> arr = {Token(types)...};
      return detail::Field{name, arr};
    }
  }

  template<typename... Ts>
  inline constexpr auto seq(const Ts&... types)
  {
    if constexpr (sizeof...(Ts) == 0)
    {
      std::array<Token, 0> arr;
      return detail::Sequence{arr};
    }
    else
    {
      std::array<Token, sizeof...(Ts)> arr = {Token(types)...};
      return detail::Sequence{arr};
    }
  }

  template<typename... Ts>
  inline constexpr auto shape(const Token& type, const Ts&... fields)
  {
    return detail::Shape{type, std::make_tuple(fields...)};
  }

  template<typename... Ts>
  inline constexpr auto wellformed(const Ts&... shapes)
  {
    return detail::Wellformed{std::make_tuple(shapes...)};
  }

  template<typename... Ts>
  inline constexpr auto
  operator/(const detail::Wellformed<Ts...>& wf, const Token& type)
  {
    return std::make_pair(wf, type);
  }

  template<typename... Ts>
  inline constexpr Index operator/(
    const std::pair<detail::Wellformed<Ts...>, Token>& pair, const Token& name)
  {
    return detail::index(pair.first, pair.second, name);
  }
}
