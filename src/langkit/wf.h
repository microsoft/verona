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
    template<size_t N>
    struct Field
    {
      Token name;
      std::array<Token, N> types;

      constexpr Field(const Token& name, const std::array<Token, N>& types)
      : name(name), types(types)
      {}
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

        if (name == field.name)
          return Index(shape.type, I);

        return index<I + 1>(shape, name);
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

    template<size_t N>
    inline std::set<Token> field_set(const Field<N>& field)
    {
      return {field.types.begin(), field.types.end()};
    }

    template<size_t I, typename... Ts>
    inline void shape_vector_i(
      const Shape<Ts...>& shape, std::vector<std::set<Token>>& defs)
    {
      if constexpr (I == sizeof...(Ts))
        return;

      defs.push_back(field_set(std::get<I>(shape.fields)));
      shape_vector_i<I + 1>(shape, defs);
    }

    template<typename... Ts>
    inline std::vector<std::set<Token>> shape_vector(const Shape<Ts...>& shape)
    {
      std::vector<std::set<Token>> defs;
      shape_vector_i<0>(shape, defs);
      return defs;
    }

    template<size_t I, typename... Ts>
    inline void wellformed_map_i(
      const Wellformed<Ts...>& wf,
      std::map<Token, std::vector<std::set<Token>>>& defs)
    {
      if constexpr (I == sizeof...(Ts))
        return;

      auto shape = std::get<I>(wf.shapes);
      defs[shape.type] = shape_vector(shape);
      wellformed_map_i<I + 1>(wf, defs);
    }

    template<typename... Ts>
    inline std::map<Token, std::vector<std::set<Token>>>
    wellformed_map(const Wellformed<Ts...>& wf)
    {
      std::map<Token, std::vector<std::set<Token>>> defs;
      wellformed_map_i<0>(wf, defs);
      return defs;
    }

    inline bool
    check(std::map<Token, std::vector<std::set<Token>>>& map, Node node)
    {
      auto find = map.find(node->type());

      if (find == map.end())
        return false;

      auto& defs = find->second;
      if (defs.size() != node->size())
        return false;

      auto it = defs.begin();

      for (auto& child : *node)
      {
        auto find = it->find(child->type());
        if (find == it->end())
          return false;

        if (!check(map, child))
          return false;

        ++it;
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
