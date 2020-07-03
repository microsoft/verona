// Copyright Microsoft and Project Verona Contributors.
// SPDX-License-Identifier: MIT
#pragma once

/**
 * ## Muting
 * A cown is muted after it runs a message where it has sent a message to an
 * overloaded/muted cown and no sender participating in the message is
 * overloaded. Once a cown is muted, the scheduler thread running the cown
 * becomes responsible for eventually unmuting it.
 *
 * ## Unmuting
 * - The cown that resulted in the muting (the key in the mute map) is no longer
 * overloaded/muted. In this case, an unmute message is sent to each scheduler
 * thread to unmute all muted cowns with the unmuted cown as its key. The entry
 * in the mute map is removed and all unmuted cowns are removed from the mute
 * set.
 * - An overloaded cown is sending a message to the muted cown. In this case, an
 * unmute message is sent to each scheduler thread to search for the cown until
 * it is found and unmuted using the mute set.
 */
