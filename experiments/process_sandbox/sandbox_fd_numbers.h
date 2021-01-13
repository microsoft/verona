// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#pragma once
namespace sandbox
{
  /**
   * The numbers for file descriptors passed into the child.  These must match
   * between libsandbox and the library runner child process.
   */
  enum SandboxFileDescriptors
  {
    /**
     * The file descriptor used for the shared memory object that contains the
     * shared heap.
     */
    SharedMemRegion = 3,
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
}
