#!/bin/bash
# Monitor all temporal datagen progress. Run: bash scripts/monitor_datagen.sh
while true; do
    clear
    echo "═══════════════════════════════════════════════════════════════════════"
    echo "  Channel Generation Monitor  $(date '+%H:%M:%S')"
    echo "═══════════════════════════════════════════════════════════════════════"

    # ─── Dataset progress ─────────────────────────────────────────────
    for dir in assets/data/channels_*_temporal; do
        [ -d "$dir" ] || continue
        name=$(basename "$dir" | sed 's/channels_//;s/_temporal//')
        total=$(ls -d "$dir"/snapshot_* 2>/dev/null | wc -l)
        target=80000
        pct=$((total * 100 / target))

        filled=$((pct / 2))
        bar=$(printf '%0.s█' $(seq 1 $filled 2>/dev/null))
        empty=$(printf '%0.s░' $(seq 1 $((50 - filled)) 2>/dev/null))

        if [ -f "$dir/progress.json" ]; then
            speed=$(python3 -c "import json; d=json.load(open('$dir/progress.json')); print(f'{d[\"avg_snap_s\"]:.1f}s')" 2>/dev/null)
            eta=$(python3 -c "import json; d=json.load(open('$dir/progress.json')); print(f'{d[\"eta_s\"]/3600:.1f}h')" 2>/dev/null)
        else
            speed="-"; eta="-"
        fi

        printf "  %-20s %s%s %5d/%d (%2d%%) %s/snap ETA %s\n" \
            "$name" "$bar" "$empty" "$total" "$target" "$pct" "$speed" "$eta"
    done

    # ─── CPU ──────────────────────────────────────────────────────────
    echo ""
    n_workers=$(pgrep -fc "generate_worker" 2>/dev/null || echo 0)
    cpu_usage=$(top -bn1 2>/dev/null | grep "Cpu(s)" | awk '{print $2}' 2>/dev/null || echo "?")
    n_cores=$(nproc)
    mem_info=$(free -h 2>/dev/null | awk '/^Mem:/{printf "%s / %s (%.0f%%)", $3, $2, $3/$2*100}')

    printf "  CPU:  %s%% used (%d cores)  |  Workers: %d  |  RAM: %s\n" \
        "$cpu_usage" "$n_cores" "$n_workers" "$mem_info"

    # Per-worker CPU (top 5)
    if [ "$n_workers" -gt 0 ]; then
        echo "  Top workers by CPU:"
        ps aux --sort=-%cpu | grep "generate_worker" | grep -v grep | head -5 | \
            awk '{printf "    PID %s: CPU %s%%  RSS %dMB\n", $2, $3, $6/1024}'
    fi

    # ─── GPU ──────────────────────────────────────────────────────────
    echo ""
    echo "  GPU:"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | \
        while IFS=, read -r idx util mem total; do
            util=$(echo $util | tr -d ' ')
            mem=$(echo $mem | tr -d ' ')
            total=$(echo $total | tr -d ' ')
            mem_pct=$((mem * 100 / total))
            # Util bar (20 chars)
            ufilled=$((util / 5))
            ubar=$(printf '%0.s▓' $(seq 1 $ufilled 2>/dev/null))
            uempty=$(printf '%0.s░' $(seq 1 $((20 - ufilled)) 2>/dev/null))
            printf "    GPU %s: %s%s %3d%%  |  VRAM %5d/%5d MB (%d%%)\n" \
                "$idx" "$ubar" "$uempty" "$util" "$mem" "$total" "$mem_pct"
        done

    # ─── Disk ─────────────────────────────────────────────────────────
    echo ""
    disk_info=$(df -h /home/sungmin/Projects/ 2>/dev/null | awk 'NR==2{printf "Disk: %s used / %s total (%s)", $3, $2, $5}')
    echo "  $disk_info"

    sleep 1
done
