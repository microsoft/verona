// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#pragma once
namespace sandbox
{
  /**
   * The numbers for file descriptors passed into the child.  These must match
   * between libsandbox and the library runner child process.
   */
  enum InheritedFileDescriptors
  {
    /**
     * C standard in.
     */
    StandardIn = STDIN_FILENO,
    /**
     * C standard out.
     */
    StandardOut = STDOUT_FILENO,
    /**
     * C standard error.
     */
    StandardErr = STDERR_FILENO,
    /**
     * The file descriptor used for the shared memory object that contains the
     * shared heap.
     */
    SharedMemRegion,
    /**
     * The file descriptor for the shared memory object that contains the
     * shared pagemap page.  This is mapped read-only in the child and updated
     * in the parent.
     */
    PageMapPage,
    /**
     * The file descriptor for the socket used to pass file descriptors into the
     * child.
     */
    FDSocket,
    /**
     * The file descriptor used for the main library.  This is passed to
     * `fdlopen` in the child.
     */
    MainLibrary,
    /**
     * The file descriptor for the pipe used to send pagemap updates to the
     * parent process.
     */
    PageMapUpdates,
    /**
     * The first file descriptor number used for directory descriptors of
     * library directories.  These are used by rtld in the child to locate
     * libraries that the library identified by `MainLibrary`depends on.
     */
    OtherLibraries
  };
  // POSIX does not mandate that these are 0, 1, and 2, but this composes with
  // the requirement that new file descriptors are always created with the
  // lowest unused number to make almost any other option infeasible, so we
  // assume this.  If anyone implements a POSIX system that does makes a
  // different choice that we care about then we will have to refactor some
  // code.
  static_assert(StandardIn == 0, "Standard in file descriptor is unexpected");
  static_assert(StandardOut == 1, "Standard out file descriptor is unexpected");
  static_assert(
    StandardErr == 2, "Standard error file descriptor is unexpected");
}
