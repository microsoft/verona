Platform abstractions
=====================

This directory includes the platform abstractions required for the
process-based foreign-code sandbox for Verona.  This is specialized to exactly
the things required for the sandboxing code and is not a generic OS abstraction
layer.  For example, the epoll / kqueue wrapper waits for readable data on one
or more sockets, it is not a generic event abstraction.

Some of the code here may eventually be moved into generic Verona runtime
platform abstractions and made more general.
