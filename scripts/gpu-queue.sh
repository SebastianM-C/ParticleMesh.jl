#!/usr/bin/env bash
# gpu-run: one queue and one lock for this machine's GPU, shared by every job and every session.
# Install a copy outside any worktree, so that every job runs the same queue whatever its commit:
#   install -D -m 755 scripts/gpu-queue.sh ~/gpu-queue/gpu-run
# A test suite taken through the queue holds the card for its CPU testsets too; run it with
# PM_GPU_HOLD="~/gpu-queue/gpu-run --hold -t <tag>" instead, outside the queue, and its gpu.jl
# waits for its turn (lib/ParticleMeshCore/test/gpu_switch.jl).
#
#   gpu-run [-p PRIO] [-t TAG] [-l LOG] [-C DIR] [-w SECS] [-d] -- CMD [ARGS...]
#       Wait for the GPU in priority order, run CMD holding it, release it when CMD ends.
#       PRIO 0..9, lower runs first, default 5; equal priorities run in submission order.
#       Output goes to LOG (default ~/gpu-queue/logs/TAG-<time>.log), which starts with
#       "# gpu-run <time> ..." and ends with "# gpu-run finished: exit N after Ss; waited Ws; log: LOG".
#       -d detaches (setsid nohup) and prints the log path: the form to use over ssh.
#       -w gives up after SECS of waiting with exit 75 ("lock wait exceeded"); default: wait forever.
#   gpu-run --hold [-p PRIO] [-t TAG]
#       For a process that needs the GPU only in its last part (a test suite's gpu.jl): wait in the
#       queue, print "granted" on stdout, keep the GPU until stdin closes (the caller exits).
#   gpu-run --pid PID [-p PRIO] [-t TAG]
#       Book the GPU for a job that is already running outside the queue, until PID exits.
#   gpu-run --status        holder, waiting jobs in order, nvidia-smi, the last finished jobs
#   gpu-run --cancel TAG    withdraw a waiting job (a running one is not touched)
#
# The lock is flock on PM_GPU_LOCK (default /tmp/particlemesh-gpu.lock), the same file as the
# repository's scripts/gpu-lock.sh, so jobs started through either never share the card.
# A waiting job polls every GPU_QUEUE_POLL seconds (default 2): the card is idle at most that long
# between two queued jobs.
set -uo pipefail
Q="${GPU_QUEUE_DIR:-$HOME/gpu-queue}"
LOCK="${PM_GPU_LOCK:-/tmp/particlemesh-gpu.lock}"
POLL="${GPU_QUEUE_POLL:-2}"
mkdir -p "$Q/tickets" "$Q/logs"
HIST="$Q/history.log"

now() { date -u +%Y-%m-%dT%H:%M:%SZ; }
pstart() { [ -r "/proc/$1/stat" ] && sed 's/^.*) //' "/proc/$1/stat" | awk '{print $20}'; }
alive() { [ -n "$1" ] && [ -n "$2" ] && [ "$(pstart "$1")" = "$2" ]; }
field() { sed -n "s/^$2=//p" "$1" 2>/dev/null | head -n1; }

# The first live ticket in order; tickets whose process is gone are removed.
head_ticket() {
  local t
  for t in $(ls "$Q/tickets" 2>/dev/null | sort); do
    if alive "$(field "$Q/tickets/$t" pid)" "$(field "$Q/tickets/$t" start)"; then echo "$t"; return; fi
    rm -f "$Q/tickets/$t"
  done
}

status() {
  echo "== GPU queue on $(hostname -s) at $(now) (lock $LOCK)"
  if [ -f "$Q/holder" ] && alive "$(field "$Q/holder" pid)" "$(field "$Q/holder" start)"; then
    echo "running: tag=$(field "$Q/holder" tag) since=$(field "$Q/holder" since) prio=$(field "$Q/holder" prio)"
    echo "         cmd=$(field "$Q/holder" cmd)"
    echo "         log=$(field "$Q/holder" log)"
  else
    echo "running: nothing through the queue"
    # flock(1) records its own short-lived pid in /proc/locks, so only the inode says who is locked.
    local ino; ino=$(stat -c %i "$LOCK" 2>/dev/null)
    if [ -n "$ino" ] && awk -v i=":$ino\$" '$2 == "FLOCK" && $6 ~ i {f = 1} END {exit !f}' /proc/locks; then
      echo "         but the lock is held outside the queue (scripts/gpu-lock.sh or a job started by hand)"
    fi
  fi
  echo "waiting, in the order they will run:"
  local t n=0
  for t in $(ls "$Q/tickets" 2>/dev/null | sort); do
    alive "$(field "$Q/tickets/$t" pid)" "$(field "$Q/tickets/$t" start)" || { rm -f "$Q/tickets/$t"; continue; }
    n=$((n + 1))
    echo "  $n. tag=$(field "$Q/tickets/$t" tag) prio=${t%%-*} since=$(field "$Q/tickets/$t" submitted) mode=$(field "$Q/tickets/$t" mode)"
    echo "     cmd=$(field "$Q/tickets/$t" cmd)"
  done
  [ "$n" -eq 0 ] && echo "  (none)"
  if command -v nvidia-smi >/dev/null; then
    echo "nvidia-smi:"
    nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader | sed 's/^/  /'
    nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader | sed 's/^/  app: /'
  fi
  echo "last finished:"
  tail -n 5 "$HIST" 2>/dev/null | sed 's/^/  /'
}

mode=run prio=5 tag="" log="" dir="" wait=0 detach=0 target=""
while [ $# -gt 0 ]; do
  case "$1" in
    --status) status; exit 0 ;;
    --cancel)
      for t in $(ls "$Q/tickets" 2>/dev/null); do
        [ "$(field "$Q/tickets/$t" tag)" = "$2" ] || continue
        kill "$(field "$Q/tickets/$t" pid)" 2>/dev/null && echo "cancelled $2"; rm -f "$Q/tickets/$t"
      done
      exit 0 ;;
    --hold) mode=hold; shift ;;
    --pid) mode=pid; target="$2"; shift 2 ;;
    -p) prio="$2"; shift 2 ;;
    -t) tag="$2"; shift 2 ;;
    -l) log="$2"; shift 2 ;;
    -C) dir="$2"; shift 2 ;;
    -w) wait="$2"; shift 2 ;;
    -d) detach=1; shift ;;
    --) shift; break ;;
    -*) echo "gpu-run: unknown option $1" >&2; exit 2 ;;
    *) break ;;
  esac
done
case "$prio" in [0-9]) ;; *) echo "gpu-run: PRIO must be one digit 0..9" >&2; exit 2 ;; esac
[ "$mode" = run ] && [ $# -eq 0 ] && { echo "gpu-run: no command (see the header of $0)" >&2; exit 2; }
[ "$mode" = pid ] && { [ -n "$target" ] && [ -d "/proc/$target" ] || { echo "gpu-run: --pid needs a running PID" >&2; exit 2; }; }
tag="${tag:-$(basename "${1:-$mode}")}"
tag="${tag//[^A-Za-z0-9._-]/_}"

if [ "$mode" = run ]; then
  [ -n "$log" ] || log="$Q/logs/$tag-$(date -u +%Y%m%dT%H%M%SZ).log"
  case "$log" in /*) ;; *) log="$PWD/$log" ;; esac
  if [ "$detach" -eq 1 ]; then
    args=(-p "$prio" -t "$tag" -l "$log" -C "${dir:-$PWD}" -w "$wait" -- "$@")
    setsid nohup "$0" "${args[@]}" >/dev/null 2>&1 </dev/null &
    echo "$log"
    exit 0
  fi
  exec >>"$log" 2>&1
fi

cmdline="$*"; [ "$mode" = hold ] && cmdline="(hold for the caller, pid $PPID)"; [ "$mode" = pid ] && cmdline="(book for running pid $target: $(tr '\0' ' ' </proc/"$target"/cmdline 2>/dev/null))"
me="$prio-$(date +%s%N)-$$"
mystart="$(pstart $$)"
printf 'tag=%s\npid=%s\nstart=%s\nmode=%s\nsubmitted=%s\ncmd=%s\nlog=%s\n' \
  "$tag" "$$" "$mystart" "$mode" "$(now)" "$cmdline" "${log:-}" >"$Q/tickets/$me"
cleanup() {
  rm -f "$Q/tickets/$me"
  [ "$(field "$Q/holder" pid 2>/dev/null)" = "$$" ] && rm -f "$Q/holder"
}
trap cleanup EXIT
trap 'exit 143' TERM INT HUP

[ "$mode" = run ] && echo "# gpu-run $(now) tag=$tag prio=$prio host=$(hostname -s) cwd=${dir:-$PWD} cmd=$cmdline"
exec 9>>"$LOCK"
t0=$(date +%s)
while :; do
  if [ "$(head_ticket)" = "$me" ] && flock -n 9; then break; fi
  if [ "$wait" -gt 0 ] && [ $(( $(date +%s) - t0 )) -ge "$wait" ]; then
    msg="gpu-run: lock wait exceeded: the GPU was not free within ${wait}s, so the command did not run"
    echo "$msg" >&2; [ "$mode" = run ] && echo "# gpu-run finished: exit 75 after 0s; waited ${wait}s; log: $log"
    exit 75
  fi
  sleep "$POLL"
done
rm -f "$Q/tickets/$me"
waited=$(( $(date +%s) - t0 ))
printf 'tag=%s\npid=%s\nstart=%s\nprio=%s\nsince=%s\ncmd=%s\nlog=%s\n' \
  "$tag" "$$" "$mystart" "$prio" "$(now)" "$cmdline" "${log:-}" >"$Q/holder"
t1=$(date +%s)
rc=0
case "$mode" in
  run)
    echo "# gpu-run $(now) got the GPU after waiting ${waited}s"
    ( if [ -n "$dir" ]; then cd "$dir" || exit 2; fi; exec "$@" ) 9>&- || rc=$? ;;
  hold)
    echo granted
    cat >/dev/null 9>&- ;;
  pid)
    while [ -d "/proc/$target" ]; do sleep "$POLL"; done ;;
esac
ran=$(( $(date +%s) - t1 ))
echo "$(now) tag=$tag mode=$mode prio=$prio exit=$rc waited=${waited}s ran=${ran}s" >>"$HIST"
[ "$mode" = run ] && echo "# gpu-run finished: exit $rc after ${ran}s; waited ${waited}s; log: $log"
exit "$rc"
