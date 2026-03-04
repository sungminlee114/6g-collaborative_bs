#!/bin/bash
# Download arXiv papers for related work survey
DIR="/disk1/Sungmin/Projects/6G-collaborativeBS/relworks"
cd "$DIR"

declare -A papers
# 1차 검색 논문들
papers["2410.03747"]="Distributed_AI_Platform_6G_RAN"
papers["2411.17712"]="Generative_AI_on_the_Edge"
papers["2410.18790"]="Large_GenAI_Models_Open_Networks_6G"
papers["2602.23623"]="Toward_E2E_Intelligence_6G_AI_Agent_RAN_CN"
papers["2512.04405"]="Towards_6G_Native_AI_Edge_Networks"
papers["2603.02156"]="How_Small_Can_6G_Reason_Tiny_LM"
papers["2505.00321"]="Edge_Large_AI_Models_6G"
papers["2507.08403"]="Towards_AI_Native_RAN_Operator_Perspective"

# Split learning / inference
papers["2306.12194"]="Split_Learning_6G_Edge_Networks"
papers["2309.16739"]="Pushing_LLMs_to_6G_Edge"
papers["2501.02001"]="Communication_Efficient_Cooperative_Edge_AI"
papers["2512.23310"]="Splitwise_Collaborative_Edge_Cloud_LLM"

# FL + O-RAN + channel
papers["2404.06324"]="Dynamic_D2D_FL_over_ORAN"
papers["2404.03088"]="Robust_FL_Wireless_Channel_Estimation"

# Beam management
papers["2602.22796"]="Multimodal_Virtual_BS_MIMO_Beam_Alignment"
papers["2511.02260"]="DL_Beam_Management_mmWave_Vehicular"
papers["2602.18151"]="Rethinking_Beam_Management_HW_Heterogeneity"
papers["2512.05680"]="Meta_Learning_MAB_Beam_Tracking"

# Edge inference
papers["2506.12210"]="Machine_Intelligence_Wireless_Edge"
papers["2505.09214"]="Efficient_Large_AI_Inference_Wireless_Edge"
papers["2501.03265"]="Optimizing_Edge_AI_Survey"
papers["2503.06027"]="On_Device_AI_Models_Survey"
papers["2403.02619"]="Training_ML_at_Edge_Survey"
papers["2512.20946"]="SLIDE_Simultaneous_Model_Download_Inference"
papers["2503.00298"]="Energy_Efficient_Edge_Inference_ISCC"

# Sionna / Dataset
papers["2504.21719"]="Sionna_RT_Technical_Report"
papers["2508.14507"]="DeepTelecom_Digital_Twin_DL_Dataset"

# O-RAN dApp/xApp
papers["2501.16502"]="dApps_Realtime_AI_Open_RAN"
papers["2601.17534"]="Self_Learning_Model_Versioning_AI_ORAN"
papers["2507.06911"]="Beyond_Connectivity_AI_RAN_Convergence_6G"
papers["2511.17514"]="XAI_on_RAN"
papers["2508.09197"]="MX_AI_Agentic_Platform_Open_AI_RAN"

# LLM/SLM for networks
papers["2412.15304"]="TinyLLM_Edge_Deployment"
papers["2503.13819"]="LLM_Empowered_IoT_6G"
papers["2412.20772"]="LLM_MultiTask_Physical_Layer"
papers["2602.06819"]="Bridging_6G_IoT_LLM_Physical_Layer"

SUCCESS=0
FAIL=0
FAILED_LIST=""

for arxiv_id in "${!papers[@]}"; do
    filename="${papers[$arxiv_id]}_${arxiv_id}.pdf"
    if [ -f "$filename" ]; then
        echo "[SKIP] $filename already exists"
        ((SUCCESS++))
        continue
    fi
    url="https://arxiv.org/pdf/${arxiv_id}"
    echo "[DOWNLOADING] $filename ..."
    if curl -sS -L -o "$filename" --connect-timeout 10 --max-time 60 "$url" 2>/dev/null; then
        # Check if it's actually a PDF (not an HTML error page)
        file_type=$(file -b "$filename" | head -1)
        if echo "$file_type" | grep -qi "pdf"; then
            echo "[OK] $filename"
            ((SUCCESS++))
        else
            echo "[FAIL] $filename - Not a valid PDF (got: $file_type)"
            rm -f "$filename"
            ((FAIL++))
            FAILED_LIST="$FAILED_LIST\n  - $arxiv_id: ${papers[$arxiv_id]}"
        fi
    else
        echo "[FAIL] $filename - Download failed"
        rm -f "$filename"
        ((FAIL++))
        FAILED_LIST="$FAILED_LIST\n  - $arxiv_id: ${papers[$arxiv_id]}"
    fi
done

echo ""
echo "==============================="
echo "Download Summary:"
echo "  Success: $SUCCESS"
echo "  Failed:  $FAIL"
if [ $FAIL -gt 0 ]; then
    echo -e "  Failed papers:$FAILED_LIST"
fi
echo "==============================="
