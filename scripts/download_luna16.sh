#!/bin/bash
# Script to download LUNA16 dataset

echo "LUNA16 Dataset Download Script"
echo "================================"
echo ""
echo "This script will help you download the LUNA16 dataset."
echo "The dataset is approximately 60GB compressed."
echo ""
echo "Please visit: https://luna16.grand-challenge.org/"
echo "to download the dataset subsets."
echo ""
echo "After downloading, extract all subsets (subset0-subset9) to:"
echo "  data/raw/luna16/"
echo ""
echo "Then run the preprocessing script:"
echo "  python src/data/preprocess.py --input_dir data/raw/luna16 --output_dir data/processed"
