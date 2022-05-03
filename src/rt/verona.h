// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

#if (defined(__FreeBSD__) && defined(_KERNEL))
#  define FreeBSD_KERNEL 1
#  define SNMALLOC_USE_THREAD_DESTRUCTOR 1
#endif

#include "cpp/lambdabehaviour.h"
#include "cpp/promise.h"
#include "cpp/vbehaviour.h"
#include "cpp/vobject.h"
#include "object/object.h"
#include "region/externalreference.h"
#include "region/freeze.h"
#include "region/immutable.h"
#include "region/region.h"
#include "region/region_api.h"
#include "sched/cown.h"
#include "sched/epoch.h"
#include "sched/mpmcq.h"
#include "sched/multimessage.h"
#include "sched/noticeboard.h"
#include "sched/schedulerthread.h"
#include "test/logging.h"
#include "test/systematic.h"

#include <snmalloc/snmalloc.h>
