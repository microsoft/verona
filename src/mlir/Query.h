// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#pragma once

#include "llvm/ADT/ArrayRef.h"

#include <optional>
#include <tuple>

namespace mlir::verona
{
  /// Index provides a comparator that orders tuples in lexicographical order,
  /// but following a given field sequence. For example, when applied to pairs,
  /// `Index<_, 1, 0>` will first compare the second elements, and only if they
  /// match will it compare the first ones. This is used to provide custom
  /// orderings for tuples stored in Relations.
  ///
  /// The first type parameter, `Compare`, is used as the underlying comparator
  /// for elements of the tuples. The same comparator type must be compatible
  /// with all constituent types.
  ///
  /// Index supports both `std::tuple` and any class which defines a
  /// `data() const` which returns a tuple. This allows wrappers around tuples
  /// to be used with Relation.
  template<typename Compare, size_t... Keys>
  struct Index;

  /// This is the base case, for when no key is provided. Since there is nothing
  /// to compare, all values are considered equal, ie. (left < right) == false.
  template<typename Compare>
  struct Index<Compare>
  {
    template<typename T>
    bool operator()(const T& left, const T& right) const
    {
      return false;
    }
  };

  template<typename Compare, size_t Head, size_t... Tail>
  struct Index<Compare, Head, Tail...>
  {
    template<typename... Ts>
    bool
    operator()(const std::tuple<Ts...>& left, const std::tuple<Ts...>& right)
    {
      const auto& left_value = std::get<Head>(left);
      const auto& right_value = std::get<Head>(right);
      if (Compare()(left_value, right_value))
        return true;
      else if (Compare()(right_value, left_value))
        return false;
      else
        return Index<Compare, Tail...>()(left, right);
    }

    /// Retrieve the "primary" element of the tuple, that is the one that is
    /// used first during comparisons.
    template<typename... Ts>
    static const auto& get_primary(const std::tuple<Ts...>& value)
    {
      return std::get<Head>(value);
    }

    template<typename T>
    bool operator()(const T& left, const T& right)
    {
      return (*this)(left.data(), right.data());
    }

    template<typename T>
    static const auto& get_primary(const T& value)
    {
      return get_primary(value.data());
    }

    /// Underlying comparator used on the contents of the tuple.
    using value_compare = Compare;
  };

  namespace detail
  {
    /// Type trait used to identify instantiations of `Index`.
    template<typename T>
    struct is_index : std::false_type
    {};

    template<typename Compare, size_t... Keys>
    struct is_index<Index<Compare, Keys...>> : std::true_type
    {};

    template<typename T>
    static constexpr bool is_index_v = is_index<T>::value;
  }

  struct QueryEngine;

  /// Type-erased version of a Relation, used to be stored in a QueryEngine.
  struct AbstractRelation
  {
    AbstractRelation(QueryEngine& engine);
    virtual bool iterate() = 0;
    virtual ~AbstractRelation(){};

    // We don't allow copying nor moving an AbstractRelation, because the
    // QueryEngine holds a pointer to it.
    AbstractRelation(const AbstractRelation&) = delete;
    AbstractRelation& operator=(const AbstractRelation&) = delete;
  };

  /// A relation represents an ordered collection of tuples.
  ///
  /// Tuples in the relation are classified as "stable" or "recent". Every time
  /// the `iterate` method is called, all "recent" tuples become "stable". This
  /// distinction is used to compute incremental updates of relations which
  /// depend on this one: only recent tuples need to be considered.
  ///
  /// Additionally, the Relation keeps a list of pending tuples. When `iterate`
  /// is called, all these pending tuples are added to the "recent" set. Tuples
  /// can be added to the pending list either manually, using the `add` method,
  /// or through "Horn clauses", in the form of the `from_XXX` methods.
  ///
  /// The choice of ordering, as determined by the `Compare` template type
  /// parameter, is critical: relations can only be joined on their "primary"
  /// element, i.e. the one that is sorted on first.
  ///
  /// For instance, given a binary relation `R(_, _)`, if sorted using the first
  /// element, we can compute the join `R(x, y), R(x, z)`. If on the other hand,
  /// the relation is sorted using the second element, we can compute the join
  /// `R(x, z), R(y, z)`.
  ///
  /// If, for example, one wishes to compute the transitive closure of R,
  /// i.e. `R(x, y), R(y, z)`, then it must be indexed by both its first and
  /// second element. This can be achieved by using two Relation instances R and
  /// S, each with the same contents but with different orderings. If R is
  /// ordered using the first element and S is ordered using the second, the
  /// transitive closure is computed as `S(x, y), R(y, z)`. Of course, any tuple
  /// added to R must also be added to S, hence `S(x, y) :- R(x, y)`.
  //
  /// The `Index` classes provide a convenient way of setting up the right
  /// orderings. The relations `R` and `S` described above can be respectively
  /// represented using types `Relation<tuple<T, T>, Index<_, 0, 1>>` and
  /// `Relation<tuple<T, T>, Index<_, 1, 0>>`.
  template<typename T, typename Compare>
  struct Relation : public AbstractRelation
  {
    using tuple_type = T;
    using tuple_compare = Compare;

    /// Create a new relation that is registered with the given engine: calling
    /// `iterate` on the engine will iterate this relation.
    Relation(QueryEngine& engine) : AbstractRelation(engine) {}

    /// Get all tuples that have been added to the relation.
    ///
    /// Unlike `stable_values` and `recent_values`, the slice returned by this
    /// method may not be fully sorted.
    ArrayRef<T> values() const
    {
      return values_;
    }

    /// Get tuples that have been added more than one iteration ago.
    ArrayRef<T> stable_values() const
    {
      return ArrayRef<T>(values_).take_front(stable_count);
    }

    /// Get tuples that have been added in the most recent iteration.
    ArrayRef<T> recent_values() const
    {
      return ArrayRef<T>(values_).drop_front(stable_count);
    }

    /// Check whether a given value is present in the relation.
    bool contains(const T& value) const
    {
      auto search = [&](const auto& c) {
        return std::binary_search(c.begin(), c.end(), value, tuple_compare());
      };

      // Because the two parts are sorted independently, we must perform two
      // separate binary searches.
      return search(stable_values()) || search(recent_values());
    }

    /// Returns true if the relation has new tuples.
    bool iterate() final
    {
      assert(llvm::is_sorted(stable_values(), tuple_compare()));
      assert(llvm::is_sorted(recent_values(), tuple_compare()));

      // Move all recent tuples into the stable section.
      std::inplace_merge(
        values_.begin(),
        values_.begin() + stable_count,
        values_.end(),
        tuple_compare());
      stable_count = values_.size();

      // Move all pending tuples into the recent section, if they do not yet
      // exist in the relation.
      llvm::copy_if(pending, std::back_inserter(values_), [&](const T& value) {
        return !contains(value);
      });
      std::sort(values_.begin() + stable_count, values_.end(), tuple_compare());
      pending.clear();

      return !recent_values().empty();
    }

    /// Add a new tuple to the relation.
    void add(T tuple)
    {
      this->pending.push_back(tuple);
    }

    /// Copy all tuples from another relation.
    ///
    /// The other relation must have the same tuple type, but does not need to
    /// be ordered in the same way. This method is useful to reindex a relation.
    template<typename R>
    void from_copy(const R& other)
    {
      static_assert(std::is_same_v<typename R::tuple_type, tuple_type>);
      llvm::copy(other.recent_values(), std::back_inserter(this->pending));
    }

    /// The `from_XXX` methods accept a function used to produce tuples to be
    /// added to the relation. We allow the function to return either `T` or
    /// `optional<T>`: if a nullopt value is returned, nothing is added to the
    /// relation. This predicate is used to static_assert that the callback's
    /// signature is correct.
    template<typename F, typename... Args>
    static constexpr bool is_tuple_producer = llvm::
      is_one_of<std::invoke_result_t<F, Args...>, T, std::optional<T>>::value;

    /// Apply a transformation to tuples from another relation, adding the
    /// result to this one.
    ///
    /// The transformation function may produce either a `T` or an
    /// `optional<T>`.
    template<typename R, typename F>
    void from_map(const R& other, F&& f)
    {
      static_assert(is_tuple_producer<F&, const typename R::tuple_type&>);

      for (const auto& entry : other.recent_values())
      {
        if (std::optional<T> result = f(entry))
          this->pending.push_back(result.value());
      }
    }

    /// Combine two relations together by applying a function to every element
    /// of their cartesian product. The result of the function is added to the
    /// current relation.
    ///
    /// This can be used to model conjunctions with no overlapping variables,
    /// such as the following:
    ///
    ///   A(x, y) :- B(x), C(y).
    ///
    template<typename R1, typename R2, typename F>
    void from_cross(const R1& left, const R2& right, F&& f)
    {
      using T1 = typename R1::tuple_type;
      using T2 = typename R2::tuple_type;
      static_assert(is_tuple_producer<F&, const T1&, const T2&>);

      auto on_join = [&](const T1& l, const T2& r) {
        if (std::optional<T> result = f(l, r))
          this->pending.push_back(result.value());
      };

      cross_tuples(left.stable_values(), right.recent_values(), on_join);
      cross_tuples(left.recent_values(), right.stable_values(), on_join);
      cross_tuples(left.recent_values(), right.recent_values(), on_join);
    }

    /// Combine two relations together by applying a function to every element
    /// of their cartesian product that satisfy a join requirement. The result
    /// of the function is added to the current relation.
    ///
    /// This can be used to model conjunctions with one overlapping variable,
    /// such as the following:
    ///
    ///   A(x, z) :- B(x, y), C(y, z).
    ///
    /// For each relation being joined on, a "key function" must be provided.
    /// This function projects from each tuple the element being joined on.
    /// In the example above, `kfB = (x, y) -> y` and `kfC = (y, z) -> y`.
    /// The return type of both key functions must be the same. Additionally, a
    /// comparison function operating on the key type must be provided.
    ///
    /// The ordering used by each relation must be compatible with the ordering
    /// used by the keys. Formally, the two orderings, respectively denoted `<R`
    /// and `<k` must satisfy the following property:
    ///
    ///   ∀ t1, t2 ∈ R. kf(t1) <k kf(t2) => t1 <R t2
    ///
    template<
      typename R1,
      typename R2,
      typename KF1,
      typename KF2,
      typename KeyComp,
      typename F>
    void from_join(
      const R1& left,
      const R2& right,
      KF1&& kf1,
      KF2&& kf2,
      KeyComp&& less,
      F&& f)
    {
      using T1 = typename R1::tuple_type;
      using T2 = typename R2::tuple_type;

      static_assert(is_tuple_producer<F&, const T1&, const T2&>);

      auto on_join = [&](const T1& l, const T2& r) {
        if (std::optional<T> result = f(l, r))
          this->pending.push_back(result.value());
      };
      auto execute = [&](const ArrayRef<T1>& l, const ArrayRef<T2>& r) {
        join_tuples(l, r, kf1, kf2, less, on_join);
      };

      execute(left.stable_values(), right.recent_values());
      execute(left.recent_values(), right.stable_values());
      execute(left.recent_values(), right.recent_values());
    }

    /// Convenience overload that joins on the "primary key" of tuples. This
    /// requires the relation to be ordered using an instantiation of the
    /// `Index` template. For example, joining on a relation `R(x, y, z)`
    /// ordered by `Index<_, 1, 0, 2>` will use `y` as its join key.
    template<typename R1, typename R2, typename F>
    void from_join(const R1& left, const R2& right, F&& f)
    {
      // Both relations should be ordered by an `Index` type, which, in addition
      // to being a comparator, exposes a `get_primary` method to retrieve a
      // tuple's primary key.
      using Idx1 = typename R1::tuple_compare;
      using Idx2 = typename R2::tuple_compare;
      static_assert(
        detail::is_index_v<Idx1> && detail::is_index_v<Idx1>,
        "Relations must be ordered by an `Index` instantiation");

      // Use the underlying comparator used by `Index`. We make sure both
      // relations' comparators actually are the same.
      using KeyComp = typename Idx1::value_compare;
      static_assert(
        std::is_same_v<KeyComp, typename Idx2::value_compare>,
        "Both relations should use the same underlying comparator");

      from_join(
        left,
        right,
        [](const auto& v) { return Idx1::get_primary(v); },
        [](const auto& v) { return Idx2::get_primary(v); },
        KeyComp{},
        f);
    }

    SmallVector<T, 0> finish() &&
    {
      assert(pending.empty());
      assert(stable_count == values_.size());
      return std::move(values_);
    }

  private:
    /// Advance an iterator until the result of the key-function changes.
    /// Returns the first iterator that exposes a different value, or `end` if
    /// no such position exists.
    ///
    /// TODO: Take advantage of the fact that the lists are sorted to scan in
    /// increasingly bigger steps, eg. by 1, 2, 4, ... and backtrack when we've
    /// overshot.
    template<typename I, typename KF>
    static I advance(I it, I end, KF&& key)
    {
      assert(it != end);
      const auto& current = key(*it);
      do
      {
        ++it;
      } while (it != end && key(*it) == current);
      return it;
    }

    /// Perform the join between two slices. Functions KF1 and KF2 are used to
    /// project the values on which to join, respectively from T1 and T2. For
    /// each pair in the join, the callback `on_join` will be called.
    ///
    /// The general idea is to perform a simultaneous traversal of the two
    /// slices, moving forward the iterator that has the smallest key. Once we
    /// identify a key value present in both lists, we use `cross_tuples` to
    /// yield all elements in the cartesian product of entries with that key.
    ///
    /// For example, given lists [ 1, 3, 3', 4 ] and [ 2, 3*, 5 ], the two
    /// iterators will, in turn, reference entries (1, 2), (3, 2), (3, 3*) and
    /// (4, 5). The callback will be called with arguments (3, 3*) and (3', 3*).
    /// We use ' and * to distinguish between the entries with the same key.
    ///
    template<
      typename T1,
      typename T2,
      typename KF1,
      typename KF2,
      typename KeyComp,
      typename OnJoin>
    static void join_tuples(
      ArrayRef<T1> left,
      ArrayRef<T2> right,
      const KF1& kf1,
      const KF2& kf2,
      const KeyComp& less,
      const OnJoin& on_join)
    {
      using K1 = std::invoke_result_t<const KF1&, const T1&>;
      using K2 = std::invoke_result_t<const KF2&, const T2&>;

      // This requirement could be relaxed; all we care is that keys can be
      // compared by KeyComp and operator==. We could even remove the later
      // requirement and only use KeyComp. On the other hand, it makes sense
      // to be joining on things that have the same type.
      static_assert(
        std::is_same_v<K1, K2>,
        "Elements being joined on must have the same type");

      assert(llvm::is_sorted(
        left, [&](const T1& l, const T1& r) { return less(kf1(l), kf1(r)); }));
      assert(llvm::is_sorted(
        right, [&](const T2& l, const T2& r) { return less(kf2(l), kf2(r)); }));

      const T1* left_it = left.begin();
      const T2* right_it = right.begin();

      while (left_it != left.end() && right_it != right.end())
      {
        const K1& left_key = kf1(*left_it);
        const K2& right_key = kf2(*right_it);

        if (left_key == right_key)
        {
          // Select all the values, both on the left and on the right, that
          // share that same key.
          const T1* left_last = advance(left_it, left.end(), kf1);
          const T2* right_last = advance(right_it, right.end(), kf2);
          ArrayRef<T1> left_values(left_it, left_last);
          ArrayRef<T2> right_values(right_it, right_last);

          cross_tuples(left_values, right_values, on_join);

          left_it = left_last;
          right_it = right_last;
        }
        else if (less(left_key, right_key))
        {
          // TODO: advance only moves to the first position with a different
          // key. We could go faster and skip to the first position that is not
          // less than right_key.
          left_it = advance(left_it, left.end(), kf1);
        }
        else
        {
          assert(less(right_key, left_key));
          right_it = advance(right_it, right.end(), kf2);
        }
      }
    }

    /// Perform the cross join between two slices. The object `f` will be called
    /// for every pair of elements in the cartesian product of `left` and
    /// `right`.
    template<typename T1, typename T2, typename F>
    static void cross_tuples(ArrayRef<T1> left, ArrayRef<T2> right, F&& f)
    {
      for (const T1& l : left)
      {
        for (const T2& r : right)
        {
          f(l, r);
        }
      }
    }

  private:
    // The first `stable_count` elements of `values_` are the "stable" ones, the
    // rest are the "recent". Each segment is sorted independently.
    //
    // Using a single vector allows us to use `std::inplace_merge` whenever we
    // want to merge the recent elements into the stable ones.
    SmallVector<T, 0> values_;
    size_t stable_count = 0;

    SmallVector<T, 0> pending;
  };

  /// A QueryEngine holds a set of mutally dependent Relations. It is used the
  /// drive the process of iterating the Relations. Typical usage looks as
  /// follows:
  ///
  ///    // Setup the engine and register Relation objects into it.
  ///    QueryEngine engine;
  ///    Relation<...> r1(engine);
  ///    Relation<...> r2(engine);
  ///    Relation<...> r3(engine);
  ///
  ///    // Seed the relations with known values.
  ///    r1.add(...);
  ///    r1.add(...);
  ///
  //     // Apply rules until a fixpoint is reached.
  ///    while (engine.iterate()) {
  ///      r1.from_join(r1, r1, ...);
  ///      r2.from_map(r1, ...);
  ///      r3.from_join(r1, r2, ...)
  ///    }
  ///
  ///    // Query the result
  ///    for (auto v : r1.values()) { print(v); }
  ///    if (r2.contains(x)) { ... }
  ///
  struct QueryEngine
  {
    bool iterate()
    {
      bool result = false;
      for (auto& it : relations)
      {
        result |= it->iterate();
      }
      return result;
    }

  private:
    void add(AbstractRelation* relation)
    {
      relations.push_back(relation);
    }

    SmallVector<AbstractRelation*, 0> relations;

    friend AbstractRelation;
  };

  inline AbstractRelation::AbstractRelation(QueryEngine& engine)
  {
    engine.add(this);
  }
}
