// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once
#include "ds/helpers.h"

#include <array>
#include <climits>
#include <fmt/color.h>
#include <fmt/core.h>
#include <fmt/ostream.h>
#include <fstream>
#include <iostream>
#include <pegmatite.hh>

namespace verona::compiler
{
  struct SourceManager
  {
    /**
     * Location in the source file.  This is an opaque value that can be
     * translated back to a source address by the context.
     *
     * Source locations are either stored as a compressed representation in
     * this value or as an index into a table.
     */
    using SourceLocation = uint32_t;

    /**
     * Line number in a source file.
     */
    using LineNumber = uint32_t;

    /**
     * The character column in the current line.
     */
    using ColumnNumber = uint32_t;

    /**
     * A fully expanded source location.  This supports files with four billion
     * lines of four billion characters each.  Anyone that has source files
     * larger than this will hit many other problems before they hit this one.
     */
    struct ExpandedSourceLocation
    {
      /**
       * The file name for the file containing this location.
       */
      std::string filename;

      /**
       * The line number for this location.
       */
      LineNumber line;

      /**
       * The column number for this location.
       */
      ColumnNumber column;
    };

    /**
     * A range in the source file.
     */
    struct SourceRange
    {
      /**
       * The first character in this source range.
       */
      SourceLocation first;

      /**
       * The last character in this source range.
       */
      SourceLocation last;

      bool operator<(const SourceRange& other) const
      {
        return std::tie(first, last) < std::tie(other.first, other.last);
      }
    };

    /**
     * The severity of a diagnostic.
     */
    enum class DiagnosticKind
    {
      /**
       * Purely informative.
       */
      Info,
      /**
       * Extra information associated with another error message.
       */
      Note,
      /**
       * Warning.  It is still possible to compile the source, but something
       * ambiguous has been interpreted in a way that may not be aligned with
       * the programmer's intent.
       */
      Warning,
      /**
       * Error.  It is impossible to correctly compile at least some of the
       * program, compilation will abort.
       */
      Error,
      /**
       * Fatal error.  Something has gone sufficiently wrong that the
       * compiler will now exit without attempting to report any more errors.
       * This may be as a result of incomplete features or bugs in the
       * compiler and should not be used in error reporting.
       */
      FatalError,
      /**
       * Marker for the last value in this enumerated type.  Used only to
       * define the `NumberOfDiagnosticKinds` value.
       */
      LastDiagnosticKind
    };
    static const size_t NumberOfDiagnosticKinds =
      static_cast<size_t>(DiagnosticKind::LastDiagnosticKind);

    static const char* name_for_diagnostic_kind(DiagnosticKind k)
    {
      static const std::array<const char*, NumberOfDiagnosticKinds>
        DiagnosticKindNames = {
          "information", "note", "warning", "error", "fatal error"};
      return DiagnosticKindNames.at(static_cast<int>(k));
    }

    static fmt::text_style style_for_diagnostic_kind(DiagnosticKind k)
    {
      static const std::array<fmt::text_style, NumberOfDiagnosticKinds>
        DiagnosticKindStyles = {fmt::text_style(),
                                fmt::text_style(),
                                fg(fmt::color::red),
                                fg(fmt::color::red),
                                fg(fmt::color::red)};
      return DiagnosticKindStyles.at(static_cast<int>(k));
    }

    /**
     * Enumeration for diagnostics that the compiler can display.  This is used
     * to provide a locale-agnostic handle for each diagnostic.
     */
    enum class Diagnostic
    {
      /**
       * A symbol has been referenced but not defined.
       */
      UndefinedSymbol,
      /**
       * Symbol is not a type.
       */
      SymbolNotType,
      /**
       * Symbol is not a class name,
       */
      SymbolNotClass,
      /**
       * Symbol is not a local variable
       */
      SymbolNotLocal,
      /**
       * Cannot apply type parameters to a type variable.
       */
      CannotApplyTypeParametersToTypeVariable,
      /**
       * Symbol is already defined elsewhere in this scope.
       */
      SymbolAlreadyExists,
      /**
       * Class already has a member with the same name.
       */
      MemberAlreadyExists,
      /**
       * Previous definition of an already defined symbol/member.
       * Used as a note together with SymbolAlreadyExists or MemberAlreadyExists
       */
      PreviousDefinitionHere,
      /**
       * The program does not have a `Main` class.
       */
      NoMainClass,
      /**
       * Main is not a class.
       */
      MainNotAClass,
      /**
       * The Main class has generic parameters.
       */
      MainClassIsGeneric,
      /**
       * The Main class does not have a `main` method.
       */
      NoMainMethod,
      /**
       * The main method has an invalid signature.
       */
      InvalidMainSignature,
      /**
       * Trying to read from a value whose type is not readable.
       */
      TypeNotReadable,
      /**
       * Trying to write to a value whose type is not writable.
       */
      TypeNotWritable,
      /**
       * Trying to send an unsendable type.
       */
      TypeNotSendableForWhen,
      /**
       * A subtyping relation doesn't hold even though it is expected to.
       */
      SubtypeAssertionFailed,
      /**
       * A subtyping relation holds even though it is expected not to.
       */
      NotSubtypeAssertionFailed,
      /**
       * A class' method has no body.
       */
      MissingMethodBodyInClass,
      /**
       * A primitive's method has no body.
       */
      MissingMethodBodyInPrimitive,
      /**
       */
      BuiltinMethodHasBody,
      /**
       * A primitive has a field.
       */
      FieldInPrimitive,
      /**
       * Type inference failed for method.
       */
      InferenceFailedForMethod,
      /**
       * Finaliser not generic.
       */
      FinaliserNotGeneric,
      /**
       * Finaliser has not params.
       */
      FinaliserHasNoParameters,
      /**
       * Multiple where clauses for the same variable.
       */
      MultipleWhereClauses,
      /**
       * Previous where clause for a variable.
       * Used as a note together with MultipleWhereClauses.
       */
      PreviousWhereClauseHere,
      /**
       * A `where ... from ...` clause was used for a parameter.
       */
      ParameterCannotBeWhereFrom,
      /**
       * A `where ... under ...` clause was used for a parameter.
       */
      ParameterCannotBeWhereUnder,
      /**
       * The wrong number of type argument was applied to a class name.
       */
      IncorrectNumberOfTypeArguments,
      /**
       * A type argument was applied to an entity but it does not satify the
       * required bound.
       */
      TypeArgumentDoesNotSatisfyBound,

      CannotUseVariable,
      WasPreviouslyConsumedHere,
      ParentWasConsumedHere,
      ParentWentOutOfScopeHere,
      ParentWasOverwrittenHere,
    };

    /**
     * Returns a format string for each diagnostic.  This is in the format that
     * can be used with libfmt.
     *
     * TODO: This function should return a *localized* diagnostic eventually,
     * but not for the first version of the compiler.
     */
    const char* format_string_for_diagnostic(Diagnostic k)
    {
      switch (k)
      {
        case Diagnostic::UndefinedSymbol:
          return "Cannot find value for symbol '{:s}'";
        case Diagnostic::SymbolNotType:
          return "{} is not a type";
        case Diagnostic::SymbolNotClass:
          return "{} is not a class";
        case Diagnostic::SymbolNotLocal:
          return "{} is not a local variable";
        case Diagnostic::CannotApplyTypeParametersToTypeVariable:
          return "Cannot apply type parameters to type variable {}";
        case Diagnostic::SymbolAlreadyExists:
          return "Symbol '{:s}' is already defined in this scope";
        case Diagnostic::MemberAlreadyExists:
          return "Class '{:s}' already has a member named '{:s}'";
        case Diagnostic::PreviousDefinitionHere:
          return "'{:s}' was previously defined here";
        case Diagnostic::NoMainClass:
          return "Class 'Main' is missing";
        case Diagnostic::MainNotAClass:
          return "'Main' must be a class";
        case Diagnostic::MainClassIsGeneric:
          return "Class 'Main' must not be generic";
        case Diagnostic::NoMainMethod:
          return "Class 'Main' does not have a 'main' method";
        case Diagnostic::InvalidMainSignature:
          return "Method 'main' has an invalid signature";
        case Diagnostic::TypeNotReadable:
          return "Type '{}' is not readable";
        case Diagnostic::TypeNotWritable:
          return "Type '{}' is not writable";
        case Diagnostic::TypeNotSendableForWhen:
          return "Type '{}' is not sendable for when clause for captured "
                 "variable: {}";
        case Diagnostic::SubtypeAssertionFailed:
          return "Static assertion failed, '{}' is not a subtype of '{}'";
        case Diagnostic::NotSubtypeAssertionFailed:
          return "Static assertion failed, '{}' is a subtype of '{}'";
        case Diagnostic::MissingMethodBodyInClass:
          return "Method '{}' in class '{}' must have a body";
        case Diagnostic::MissingMethodBodyInPrimitive:
          return "Method '{}' in primitive '{}' must have a body";
        case Diagnostic::BuiltinMethodHasBody:
          return "Builtin method '{}' in '{}' must not have a body";
        case Diagnostic::FieldInPrimitive:
          return "Primitives cannot have fields";
        case Diagnostic::InferenceFailedForMethod:
          return "Inference failed for method {}";
        case Diagnostic::FinaliserHasNoParameters:
          return "Finaliser should have no parameters in class {}";
        case Diagnostic::FinaliserNotGeneric:
          return "Finaliser should not take generic parameters in class {}";
        case Diagnostic::MultipleWhereClauses:
          return "Found multiple where clauses for '{}'";
        case Diagnostic::PreviousWhereClauseHere:
          return "Previous where clause for '{}' was here";
        case Diagnostic::ParameterCannotBeWhereFrom:
          return "`where ... from` clauses may not be used with parameters.";
        case Diagnostic::ParameterCannotBeWhereUnder:
          return "`where ... under` clauses may not be used with parameters.";
        case Diagnostic::IncorrectNumberOfTypeArguments:
          return "The wrong number of type arguments were specified for '{}', "
                 "expected {}, got {}.";
        case Diagnostic::TypeArgumentDoesNotSatisfyBound:
          return "Type argument '{}' for '{}' does not satisfy its bound '{}'";
        case Diagnostic::CannotUseVariable:
          return "Cannot use variable '{}'";
        case Diagnostic::WasPreviouslyConsumedHere:
          return "'{}' was previously consumed here";
        case Diagnostic::ParentWasConsumedHere:
          return "Its parent, '{}', was consumed here";
        case Diagnostic::ParentWentOutOfScopeHere:
          return "Its parent, '{}', went out of scope here";
        case Diagnostic::ParentWasOverwrittenHere:
          return "Its parent, '{}', was overwitten here";

          EXHAUSTIVE_SWITCH;
      }
    }

    /**
     * Expand a compressed source location.
     */
    ExpandedSourceLocation expand_source_location(SourceLocation s) const
    {
      if (is_small_source_location(s))
      {
        return {file_names.at(get_file(s)), get_line(s), get_column(s)};
      }
      auto loc = large_locations.at(get_large_index(s));
      return {file_names.at(loc.file_index), loc.line, loc.column};
    }

    /**
     * Construct a source range from a Pegmatite input range.
     */
    SourceRange source_range_from_input_range(const pegmatite::InputRange& r)
    {
      return {
        make_source_location(
          file_indexes[r.start.filename()], r.start.line, r.start.col),
        make_source_location(
          file_indexes[r.finish.filename()], r.finish.line, r.finish.col)};
    }

    /**
     * Add a source file.
     *
     * Note: This is not currently necessary (we could read it from the
     * Pegmatite location), but eventually we will probably want the source
     * manager to be told how to access the file, rather than having it open a
     * new fstream, and for it to manage ownership of the buffered files, so
     * adding this now ensures that there is a place in the caller to modify
     * later.
     */
    void add_source_file(const std::string& filename)
    {
      assert(file_indexes.find(filename) == file_indexes.end());
      file_indexes[filename] = static_cast<uint32_t>(file_names.size());
      file_names.emplace_back(filename);
    }

    /**
     * Print the preamble of a diagnostic.  This currently always generates
     * GCC-style prefixes (which most *NIX tools can parse), it could be
     * extended to provide cl.exe-style diagnostics for Visual Studio.
     */
    template<typename Stream, typename... Args>
    void print_diagnostic(
      Stream& s,
      SourceLocation l,
      DiagnosticKind k,
      Diagnostic d,
      Args&&... args)
    {
      diagnostic_counter(k)++;
      auto loc = expand_source_location(l);
      s << format(
        fmt::emphasis::bold, "{}:{}:{}: ", loc.filename, loc.line, loc.column);
      s << format(style_for_diagnostic_kind(k), "{}: ", k);
      s << format(
             fmt::emphasis::bold,
             format_string_for_diagnostic(d),
             std::forward<Args>(args)...)
        << std::endl;
    }

    /**
     * Print the preamble of a diagnostic which does not have associated
     * location.
     */
    template<typename Stream, typename... Args>
    void print_global_diagnostic(
      Stream& s, DiagnosticKind k, Diagnostic d, Args&&... args)
    {
      diagnostic_counter(k)++;
      s << format(style_for_diagnostic_kind(k), "{}: ", k);
      s << format(
             fmt::emphasis::bold,
             format_string_for_diagnostic(d),
             std::forward<Args>(args)...)
        << std::endl;
    }

    /**
     * Print the line containing an error.
     */
    template<typename Stream>
    void print_line_diagnostic(Stream& s, SourceRange r)
    {
      auto loc = expand_source_location(r.first);
      auto endloc = expand_source_location(r.last);
      std::string buffer;
      std::ifstream file(loc.filename, std::ios::binary);
      // Skip over all lines before the one we want.  This is horribly
      // inefficient to do on every error message.  If we stop using
      // StreamInput and buffer the input before passing it to Pegmatite then
      // we can just store iterators for each line (which we can get trivially
      // by adding a handler for the newline character).  Even without that, we
      // could cache this after the first error.
      for (SourceManager::LineNumber i = 1; i < loc.line; i++)
      {
        std::getline(file, buffer);
      }
      // Read the line that we want and print it in the diagnostic.
      std::getline(file, buffer);
      s << buffer << std::endl;
      // Align the caret under the first character.
      std::string spaces(loc.column - 1, ' ');
      for (SourceManager::ColumnNumber i = 1; i < loc.column; i++)
      {
        if (buffer[i] == '\t')
        {
          spaces[i] = '\t';
        }
      }
      s << spaces;
      // Draw tildes to either the end of the line or the last character of the
      // specified range, whichever comes first.
      size_t end = endloc.column;
      if ((loc.filename != endloc.filename) || (loc.line != endloc.line))
      {
        end = buffer.size();
      }
      std::string tildes(
        (end > loc.column + 1) ? end - loc.column - 1 : 0, '~');
      s << format(fmt::emphasis::bold | fg(fmt::color::lime), "^{}\n", tildes);
    }

    /**
     * Print a summary of the diagnostics that have been generated.
     */
    template<typename Stream>
    void print_diagnostic_summary(Stream& s)
    {
      int warns = diagnostic_counter(DiagnosticKind::Warning);
      int errors = diagnostic_counter(DiagnosticKind::Error);
      const char* ws = warns > 1 ? "warnings" : "warning";
      const char* es = errors > 1 ? "errors" : "error";
      if (warns > 0 && errors > 0)
      {
        s << format("{} {} and {} {} generated\n", warns, ws, errors, es);
      }
      else if (warns > 0)
      {
        s << format("{} {} generated\n", warns, ws);
      }
      else if (errors > 0)
      {
        s << format("{} {} generated\n", errors, es);
      }
    }

    /**
     * Returns true if any errors have occurred.
     */
    bool have_errors_occurred()
    {
      return diagnostic_counter(DiagnosticKind::Error) > 0;
    }

    void set_enable_colored_diagnostics(bool enable)
    {
      enable_colored_diagnostics = enable;
    }

  private:
    /**
     * Index of a file in the files table.
     */
    using FileIndex = uint32_t;

    /**
     * A stored file.  This is almost the same as `ExpandedSourceLocation`, but
     * does not store a copy of the filename string for every stored location.
     */
    struct StoredSourceLocation
    {
      /**
       * The index in `file_names` of the file name.
       */
      FileIndex file_index;

      /**
       * The line number.
       */
      LineNumber line;

      /**
       * The column number within the line.
       */
      ColumnNumber column;

      /**
       * Equality comparison, required for these to be used as keys in the
       * `unordered_map`.
       */
      bool operator==(const StoredSourceLocation& other) const
      {
        return (
          (file_index == other.file_index) && (line == other.line) &&
          (column == other.column));
      }
    };

    /**
     * Map from file names to indexes in the `files` vector.
     */
    std::unordered_map<std::string, FileIndex> file_indexes;

    /**
     * Vector of files.
     */
    std::vector<std::string> file_names;

    /**
     * Construct a `SourceLocation` from a line number and a character number
     * within that line.
     */
    SourceLocation
    make_source_location(FileIndex file, LineNumber line, ColumnNumber column)
    {
      if (
        (file < min_large_file) && (line < min_large_line) &&
        (column < min_large_column))
      {
        return set_file(set_line(set_column(column, 0), line), file);
      }
      auto existing = large_location_index.find({file, line, column});
      if (existing != large_location_index.end())
      {
        return existing->second;
      }
      large_location_index.insert({{file, line, column}, next_source_location});
      large_locations.push_back({file, line, column});
      return next_source_location++;
    }

    /**
     * Hash an expanded source address.  The standard library does not contain
     * a definition of `std::hash` for `std::pair`.
     */
    struct HashSourceAddress
    {
      /**
       * Compute the hash of the two fields and combine using xor.  Errors on
       * diagonals are no more common than errors anywhere else, so xor is
       * probably fine for combining values.
       */
      std::size_t operator()(StoredSourceLocation const s) const noexcept
      {
        return std::hash<LineNumber>()(s.line) ^
          std::hash<ColumnNumber>()(s.column);
      }
    };

    /**
     * The index of the most significant bit in a source location integer type.
     */
    static const unsigned small_top_bit =
      (sizeof(SourceLocation) * CHAR_BIT) - 1;

    /**
     * Number of bits used to store the file index in the compressed encoding.
     *
     * We currently support only one file, so have space for two (0 and 1) in
     * this encoding.
     */
    static const unsigned small_file_bits = 1;

    /**
     * Number of bits used to store the line number in the compressed encoding.
     *
     * A 20-bit encoding gives you a million lines, which should be far more
     * than most source files need.
     */
    static const unsigned small_line_bits = 20;

    /**
     * Number of bits used to store the column number in the compressed encoding
     *
     * This is currently 10 bits, giving 1024-character lines.  A few are
     * likely to be too long for this, but hopefully not too many.
     */
    static const unsigned small_column_bits =
      (small_top_bit - small_file_bits - small_line_bits - 1);

    /**
     * The offset of the most significant bit of the file index in the
     * compressed encoding from the least significant bit of the
     * `SourceLocation`.
     */
    static const unsigned small_file_offset =
      small_line_bits + small_column_bits + small_file_bits;

    /**
     * The offset of the most significant bit of the line number in the
     * compressed encoding from the least significant bit of the
     * `SourceLocation`.
     */
    static const unsigned small_line_offset =
      small_column_bits + small_line_bits;

    /**
     * The offset of the most significant bit of the column number in the
     * compressed encoding from the least significant bit of the
     * `SourceLocation`.
     */
    static const unsigned small_column_offset = small_column_bits;

    /**
     * `SourceLocation` is a discriminated union with a one-bit discriminator
     * for differentiating between an index into a table or a bitfield
     * containing all of the data.  This is the offset of the discriminator bit
     * from the least-significant bit in the `SourceLocation`.
     */
    static const unsigned discriminator_offset = small_file_offset + 1;

    /**
     * The smallest value of a file index that cannot be represented in the
     * compressed encoding.
     */
    static const unsigned min_large_file = 1 << small_file_bits;

    /**
     * The smallest value of a line number that cannot be represented in the
     * compressed encoding.
     */
    static const unsigned min_large_column = 1 << small_column_bits;

    /**
     * The smallest value of a column number that cannot be represented in the
     * compressed encoding.
     */
    static const unsigned min_large_line = 1 << small_line_bits;
    // The discriminator should be the highest bit, or we have not used all of
    // the bits.  If this fails, increase some of the `*_bits` values until it
    // passes!
    static_assert(discriminator_offset == small_top_bit);

    /**
     * Index of large source locations, mapping from the location to the index
     * in the `large_locations` table.  This is used to avoid inserting large
     * source locations more than once.
     */
    std::unordered_map<StoredSourceLocation, SourceLocation, HashSourceAddress>
      large_location_index;

    /**
     * Table of source locations, mapping from the index stored in the
     * compressed representation to the expanded form.
     */
    std::vector<StoredSourceLocation> large_locations;

    /**
     * The next source location to allocate for source locations that can't be
     * expressed in the compressed representation.
     */
    SourceLocation next_source_location = static_cast<SourceLocation>(1)
      << discriminator_offset;

    /**
     * Extract some bits from a bitfield stored in unsigned integer type `T`.
     * The `HighBit` is the location of the most significant bit to be
     * extracted and `Width` is the number of bits to extract.
     */
    template<typename T, unsigned HighBit, unsigned Width>
    static constexpr T extract_bits(T bitfield)
    {
      static_assert(std::is_integral_v<T>);
      static_assert(std::is_unsigned_v<T>);
      auto const upper_bound = sizeof(T) * CHAR_BIT;
      static_assert(
        (upper_bound > HighBit) || ((upper_bound == HighBit) && (Width == 1)));
      static_assert(HighBit >= Width);
      const T mask = (static_cast<T>(1) << (Width)) - 1;
      return (bitfield >> (HighBit - Width)) & mask;
    }

    /**
     * Set some bits in a bitfield stored in unsigned integer type `T`.
     * The `HighBit` is the location of the most significant bit to be
     * set and `Width` is the number of bits to set, to the value given in the
     * low `Width` bits of `value`.
     */
    template<typename T, unsigned HighBit, unsigned Width>
    static T set_bits(T bitfield, T value)
    {
      static_assert(std::is_integral_v<T>);
      static_assert(std::is_unsigned_v<T>);
      static_assert((sizeof(T) * CHAR_BIT) > HighBit);
      static_assert(HighBit >= Width);
      const T mask = (static_cast<T>(1) << Width) - 1;
      // Clear the bits in the target
      bitfield = bitfield & ~(mask << HighBit);
      // Clear any extraneous bits in the source
      value = value & mask;
      return (value << (HighBit - Width)) | bitfield;
    }

    /**
     * Extract the file index from a `SourceLocation`.
     */
    static FileIndex get_file(SourceLocation l)
    {
      return extract_bits<SourceLocation, small_file_offset, small_file_bits>(
        l);
    }

    /**
     * Extract the line number from a `SourceLocation`.
     */
    static LineNumber get_line(SourceLocation l)
    {
      return extract_bits<SourceLocation, small_line_offset, small_line_bits>(
        l);
    }

    /**
     * Extract the column number from a `SourceLocation`.
     */
    static ColumnNumber get_column(SourceLocation l)
    {
      return extract_bits<
        SourceLocation,
        small_column_offset,
        small_column_bits>(l);
    }

    /**
     * Set the file index in a `SourceLocation`.
     */
    static SourceLocation set_file(SourceLocation l, FileIndex f)
    {
      return set_bits<SourceLocation, small_file_offset, small_file_bits>(l, f);
    }

    /**
     * Set the line number in a `SourceLocation`.
     */
    static SourceLocation set_line(SourceLocation l, LineNumber f)
    {
      return set_bits<SourceLocation, small_line_offset, small_line_bits>(l, f);
    }

    /**
     * Set the column number in a `SourceLocation`.
     */
    static SourceLocation set_column(SourceLocation l, ColumnNumber f)
    {
      return set_bits<SourceLocation, small_column_offset, small_column_bits>(
        l, f);
    }

    /**
     * Returns true if this is a source location in the compressed format,
     * false otherwise.
     */
    static bool is_small_source_location(SourceLocation l)
    {
      static_assert(
        extract_bits<SourceLocation, discriminator_offset + 1, 1>(
          static_cast<SourceLocation>(1) << 31) == 1);
      return extract_bits<SourceLocation, discriminator_offset + 1, 1>(l) == 0;
    }

    /**
     * Assuming that this is a source location storing an index into the table,
     * rather than the value inline, extract the index.
     */
    static SourceLocation get_large_index(SourceLocation l)
    {
      assert(!is_small_source_location(l));
      return extract_bits<
        SourceLocation,
        discriminator_offset - 1,
        discriminator_offset - 1>(l);
    }

    /**
     * Number of diagnostics generated of each kind.
     */
    std::array<int, NumberOfDiagnosticKinds> diagnostics_count = {0};

    /**
     * Returns the counter associated with a diagnostic kind.
     */
    int& diagnostic_counter(DiagnosticKind k)
    {
      return diagnostics_count.at(static_cast<int>(k));
    }

    /**
     * Whether the diagnostics should include colors.
     */
    bool enable_colored_diagnostics = false;

    /**
     * Format a string using the given style, template and arguments.
     *
     * The style is only applied if colors are enabled for this SourceManager.
     */
    template<typename... Args>
    std::string format(
      const fmt::text_style& ts, std::string_view format_str, Args&&... args)
    {
      if (enable_colored_diagnostics)
        return fmt::format(ts, format_str, std::forward<Args>(args)...);
      else
        return fmt::format(format_str, std::forward<Args>(args)...);
    }

    /**
     * Format a string using the given template and arguments.
     */
    template<typename... Args>
    std::string format(std::string_view format_str, Args&&... args)
    {
      return fmt::format(format_str, std::forward<Args>(args)...);
      return "";
    }
  };

  using Diagnostic = SourceManager::Diagnostic;
  using DiagnosticKind = SourceManager::DiagnosticKind;
}

/**
 * Formatter for diagnostic kinds.  Inherits from the `string_view`
 * specialisation to get a default `parse` implementation.
 */
template<>
struct fmt::formatter<verona::compiler::SourceManager::DiagnosticKind>
: formatter<string_view>
{
  /**
   * Format method.  Inserts the string representation of the diagnostic kind.
   */
  template<typename FormatContext>
  auto
  format(verona::compiler::SourceManager::DiagnosticKind k, FormatContext& ctx)
  {
    return formatter<string_view>::format(
      verona::compiler::SourceManager::name_for_diagnostic_kind(k), ctx);
  }
};

namespace verona::compiler
{
  namespace
  {
    template<typename... Args>
    void report(
      SourceManager& sm,
      const std::optional<SourceManager::SourceRange>& sr,
      SourceManager::DiagnosticKind kind,
      SourceManager::Diagnostic d,
      Args&&... args)
    {
      if (sr)
      {
        sm.print_diagnostic(
          std::cerr, sr->first, kind, d, std::forward<Args>(args)...);
        sm.print_line_diagnostic(std::cerr, *sr);
      }
      else
      {
        sm.print_global_diagnostic(
          std::cerr, kind, d, std::forward<Args>(args)...);
      }
    }
  }
}
