// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT

#pragma once

/**
 * This file contains the callback numbers for callbacks from the sandbox.
 */
namespace sandbox
{
  /**
   * The kind of callback.  This is used to dispatch the callback to the
   * correct handler.
   *
   * This enumeration starts from 0 then contains all of the system calls that
   * we emulate, then libc functions, and then provides a marker for
   * dynamically added callbacks.  New system calls should be added before
   * `SyscallCallbackCount`.  Entries in this enumeration must not be reordered
   * without updating all of the structures that are indexed on it.
   */
  enum CallbackKind
  {
    /**
     * Marker for the first callback that implements a system call.
     */
    FirstSyscall = 0,
    /**
     * Proxying an `open` system call.
     */
    Open = FirstSyscall,
    /**
     * Proxying a `stat` system call.
     */
    Stat,
    /**
     * Proxying an `access` system call.
     */
    Access,
    /**
     * Proxying an `openat` system call.
     */
    OpenAt,
    /**
     * Proxying a `bind` system call.
     */
    Bind,
    /**
     * Proxying an `connect` system call.
     */
    Connect,

    /**
     * The number of system-call callbacks.
     */
    SyscallCallbackCount,
    /**
     * Marker for the first built-in callback that represents a libc function.
     */
    FirstLibCCall = SyscallCallbackCount,
    /**
     * Proxying a `getaddrinfo` libc call.
     */
    GetAddrInfo = FirstLibCCall,
    /**
     * Marker for the last built-in callback that represents a libc function.
     */
    LastLibCCall = GetAddrInfo,
    /**
     * Total number of built-in callback kinds.
     */
    BuiltInCallbackKindCount,
    /**
     * User-defined callback numbers start here.  User for callbacks from the
     * sandbox into Verona code and will itself be multiplexed.
     */
    FirstUserFunction = BuiltInCallbackKindCount,
  };
}
