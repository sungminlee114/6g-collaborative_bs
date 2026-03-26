#!/bin/bash
# Monitor datagen progress.
# Usage:
#   watch -n1 -d bash scripts/monitor_datagen.sh    ← recommended (diff highlight)
#   bash scripts/monitor_datagen.sh --loop           ← self-refreshing fallback

print_status() {
    echo "═══════════════════════════════════════════════════════════════════════"
    echo "  Channel Generation Monitor  $(date '+%H:%M:%S')"
    echo "═══════════════════════════════════════════════════════════════════════"

    for dir in assets/data/channels_*_temporal; do
        [ -d "$dir" ] || continue
        name=$(basename "$dir" | sed 's/channels_//;s/_temporal//')

        # Auto-detect target from trajectory_info or progress
        target=200
        for tj in "$dir/trajectory_info.json" "assets/data/shared_trajectories/trajectory_info.json"; do
            if [ -f "$tj" ]; then
                t=$(python3 -c "import json; print(json.load(open('$tj')).get('num_snapshots', 200))" 2>/dev/null)
                [ -n "$t" ] && target=$t && break
            fi
        done

        snap_count=0; speed="-"; eta="-"
        if [ -f "$dir/progress.json" ]; then
            read -r snap_count speed eta < <(python3 -c "
import json
d=json.load(open('$dir/progress.json'))
done = d.get('done', 0)
avg = d.get('avg_snap_s', 0)
eta_s = d.get('eta_s', 0)
if eta_s > 3600:
    eta_str = f'{eta_s/3600:.1f}h'
elif eta_s > 60:
    eta_str = f'{eta_s/60:.0f}m'
else:
    eta_str = f'{eta_s:.0f}s'
print(f'{done} {avg:.1f}s {eta_str}')
" 2>/dev/null)
        fi

        # Also check h5 file existence as completion indicator
        if [ -f "$dir/channels.h5" ] && [ "$snap_count" = "0" ]; then
            snap_count=$target
        fi

        if [ "$target" -gt 0 ] 2>/dev/null; then
            pct=$((snap_count * 100 / target))
        else
            pct=0
        fi
        # Clamp to 0-100
        [ "$pct" -gt 100 ] 2>/dev/null && pct=100
        [ "$pct" -lt 0 ] 2>/dev/null && pct=0

        filled=$((pct / 2))
        empty=$((50 - filled))
        bar=$(printf '%0.s█' $(seq 1 $filled 2>/dev/null))
        space=$(printf '%0.s░' $(seq 1 $empty 2>/dev/null))

        printf "  %-20s %s%s %5s/%s (%2d%%) %s/snap ETA %s\n" \
            "$name" "$bar" "$space" "$snap_count" "$target" "$pct" "$speed" "$eta"
    done

    # CPU / RAM
    echo ""
    n_workers=$(pgrep -fc "generate_worker" 2>/dev/null || echo 0)
    read -r cpu_user cpu_idle < <(top -bn1 2>/dev/null | awk '/^%Cpu/{print $2, $8}' || echo "? ?")
    mem_info=$(free -h 2>/dev/null | awk '/^Mem:/{printf "%s / %s", $3, $2}')
    printf "  CPU: %s%% user (%s idle)  |  Workers: %s  |  RAM: %s\n" \
        "$cpu_user" "$cpu_idle" "$n_workers" "$mem_info"

    # GPU
    echo ""
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | \
        while IFS=, read -r idx util mem total; do
            util=$(echo $util | tr -d ' ')
            mem=$(echo $mem | tr -d ' ')
            total=$(echo $total | tr -d ' ')
            ufilled=$((util / 5))
            uempty=$((20 - ufilled))
            ubar=$(printf '%0.s▓' $(seq 1 $ufilled 2>/dev/null))
            uespace=$(printf '%0.s░' $(seq 1 $uempty 2>/dev/null))
            printf "    GPU %s: %s%s %3d%%  VRAM %5d/%5dMB\n" \
                "$idx" "$ubar" "$uespace" "$util" "$mem" "$total"
        done

    # Disk
    echo ""
    df -h /home/sungmin/Projects/ 2>/dev/null | awk 'NR==2{printf "  Disk: %s / %s (%s)\n", $3, $2, $5}'
}

if [ "$1" = "--loop" ]; then
    while true; do clear; print_status; sleep 1; done
else
    print_status
fi
