# Trains the three configurations and evaluates them (PowerShell).
# Usage:  .\run_all.ps1            (default 5000 samples)   or   .\run_all.ps1 -Samples 3000
param([int]$Samples = 5000)

foreach ($config in "baseline", "rank16", "highlr") {
    python train_lora.py --config $config --max-samples $Samples
}
python evaluate_lora.py --configs baseline rank16 highlr
