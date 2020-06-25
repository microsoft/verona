// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#if (defined(__FreeBSD__) && defined(_KERNEL))
#  define FreeBSD_KERNEL 1
#  define SNMALLOC_USE_THREAD_DESTRUCTOR 1
#endif

#include "cpp/vaction.h"
#include "cpp/vobject.h"
#include "object/object.h"
#include "region/externalreference.h"
#include "region/freeze.h"
#include "region/immutable.h"
#include "region/region.h"
#include "sched/cown.h"
#include "sched/epoch.h"
#include "sched/multimessage.h"
#include "sched/noticeboard.h"
#include "sched/schedulerthread.h"
#include "sched/spmcq.h"
#include "test/systematic.h"

#include <snmalloc.h>
