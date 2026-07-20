#!/bin/sh

# Render the three tokens below with POSIX-shell-quoted, prevalidated operational
# values. Provider arguments are intentionally ignored.
exec >/dev/null 2>&1

capture_executable=__SALIENCEGATE_EXECUTABLE_SHELL__
capture_sleep=__SALIENCEGATE_WATCHDOG_SHELL__
capture_profile=__SALIENCEGATE_PROFILE_SHELL__
capture_connection=__SALIENCEGATE_CONNECTION_SHELL__
capture_child_pid=
capture_watchdog_pid=

capture_cleanup() {
    trap - HUP INT TERM
    if [ -n "$capture_watchdog_pid" ]; then
        kill -TERM "$capture_watchdog_pid" 2>/dev/null
    fi
    if [ -n "$capture_child_pid" ]; then
        kill -TERM "$capture_child_pid" 2>/dev/null
        kill -KILL "$capture_child_pid" 2>/dev/null
    fi
    exit 0
}

trap capture_cleanup HUP INT TERM

case "$capture_executable" in
    /*) ;;
    *) exit 0 ;;
esac

if [ ! -f "$capture_executable" ] || [ -L "$capture_executable" ] || [ ! -x "$capture_executable" ]; then
    exit 0
fi

case "$capture_sleep" in
    /*) ;;
    *) exit 0 ;;
esac

if [ ! -f "$capture_sleep" ] || [ -L "$capture_sleep" ] || [ ! -x "$capture_sleep" ]; then
    exit 0
fi

case "$capture_profile" in
    codex-hooks/v1|claude-code-hooks/v1|opencode-plugin/v1|pi-extension/v1) ;;
    *) exit 0 ;;
esac

"$capture_executable" \
    --profile "$capture_profile" \
    --connection "$capture_connection" <&0 &
capture_child_pid=$!

(
    "$capture_sleep" 2
    kill -TERM "$capture_child_pid" 2>/dev/null
    kill -KILL "$capture_child_pid" 2>/dev/null
) </dev/null &
capture_watchdog_pid=$!

wait "$capture_child_pid" 2>/dev/null
capture_child_pid=
kill -TERM "$capture_watchdog_pid" 2>/dev/null
wait "$capture_watchdog_pid" 2>/dev/null
capture_watchdog_pid=
exit 0
