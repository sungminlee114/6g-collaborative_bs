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

        if [ -f "$dir/progress.json" ]; then
            read -r snap_count speed eta < <(python3 -c "
import json, os
d=json.load(open('$dir/progress.json'))
# Count snapshots via listdir (faster than ls -d glob)
snaps=sum(1 for x in os.listdir('$dir') if x.startswith('snapshot_'))
print(f'{snaps} {d.get(\"avg_snap_s\",0):.1f}s {d.get(\"eta_s\",0)/3600:.1f}h')
" 2>/dev/null)
        else
            snap_count=0; speed="-"; eta="-"
        fi

        # Auto-detect target from trajectory_info or progress
        target=20000
        for tj in "$dir/trajectory_info.json" "assets/data/shared_trajectories/trajectory_info.json"; do
            [ -f "$tj" ] && target=$(python3 -c "import json; print(json.load(open('$tj')).get('num_snapshots', 20000))" 2>/dev/null) && break
        done

        pct=$((snap_count * 100 / target))
        filled=$((pct / 2))
        bar=$(printf '%0.s█' $(seq 1 $filled 2>/dev/null))
        empty=$(printf '%0.s░' $(seq 1 $((50 - filled)) 2>/dev/null))

        printf "  %-20s %s%s %5d/%d (%2d%%) %s/snap ETA %s\n" \
            "$name" "$bar" "$empty" "$snap_count" "$target" "$pct" "$speed" "$eta"
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
            ubar=$(printf '%0.s▓' $(seq 1 $ufilled 2>/dev/null))
            uempty=$(printf '%0.s░' $(seq 1 $((20 - ufilled)) 2>/dev/null))
            printf "    GPU %s: %s%s %3d%%  VRAM %5d/%5dMB\n" \
                "$idx" "$ubar" "$uempty" "$util" "$mem" "$total"
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
