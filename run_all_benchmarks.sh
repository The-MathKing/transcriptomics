#!/usr/bin/env bash
set -e

PYTHON="/Volumes/2TB/realhyunh/spatial_calibration_benchmark/.venv/bin/python"

echo "=== 1. Running Subclass (k=23) Benchmark Pipeline (DestVI) ==="
$PYTHON src/benchmark_v4.py

echo "=== 2. Running Class (k=4) Benchmark Pipeline (DestVI) ==="
$PYTHON src/benchmark_class.py

echo "=== 3. Running cell2location Fine Benchmark (k=23) ==="
$PYTHON src/benchmark_c2l.py

echo "=== 4. Running cell2location Coarse Benchmark (k=4) ==="
$PYTHON src/benchmark_c2l_coarse.py

echo "=== 5. Regenerating Figures and Table Outputs ==="
$PYTHON src/plot_results.py
$PYTHON src/plot_knots.py
$PYTHON src/plot_granularity.py
$PYTHON src/plot_multimodel.py

echo "=== All Sequential Benchmarks Complete Successfully! ==="
