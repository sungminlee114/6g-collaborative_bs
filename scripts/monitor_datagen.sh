#!/bin/bash
# Monitor all temporal datagen progress. Run: bash scripts/monitor_datagen.sh
# Updates every 10 seconds.
while true; do
    clear
    echo "═══════════════════════════════════════════════════════════"
    echo "  Channel Generation Monitor  $(date '+%H:%M:%S')"
    echo "═══════════════════════════════════════════════════════════"
    for dir in assets/data/channels_*_temporal; do
        [ -d "$dir" ] || continue
        name=$(basename "$dir" | sed 's/channels_//;s/_temporal//')
        total=$(ls -d "$dir"/snapshot_* 2>/dev/null | wc -l)
        target=80000
        pct=$((total * 100 / target))

        # Bar
        filled=$((pct / 2))
        bar=$(printf '%0.s█' $(seq 1 $filled 2>/dev/null))
        empty=$(printf '%0.s░' $(seq 1 $((50 - filled)) 2>/dev/null))

        # Speed from progress.json
        if [ -f "$dir/progress.json" ]; then
            speed=$(python3 -c "import json; d=json.load(open('$dir/progress.json')); print(f'{d[\"avg_snap_s\"]:.1f}s')" 2>/dev/null)
            eta=$(python3 -c "import json; d=json.load(open('$dir/progress.json')); print(f'{d[\"eta_s\"]/3600:.1f}h')" 2>/dev/null)
        else
            speed="-"
            eta="-"
        fi

        printf "  %-20s %s%s %5d/%d (%2d%%) %s/snap ETA %s\n" \
            "$name" "$bar" "$empty" "$total" "$target" "$pct" "$speed" "$eta"
    done
    echo ""
    echo "  GPU utilization:"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | \
        while IFS=, read -r idx util mem; do
            printf "    GPU %s: %s%% util, %s MB\n" "$idx" "$(echo $util | tr -d ' ')" "$(echo $mem | tr -d ' ')"
        done
    sleep 10
done
