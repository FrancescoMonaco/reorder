#!/bin/bash
# Test script for count_nonzero_blocks.py

set -e

echo "Testing count_nonzero_blocks.py"
echo "================================"

# Find a test matrix (adjust path as needed)
TEST_MATRIX="${1:-../../test/test_matrix.mtx}"

if [ ! -f "$TEST_MATRIX" ]; then
    echo "Error: Test matrix not found: $TEST_MATRIX"
    echo "Usage: bash test_count_blocks.sh [path/to/matrix.mtx]"
    exit 1
fi

echo "Test matrix: $TEST_MATRIX"
echo ""

# Test 1: Default block sizes
echo "Test 1: Default block sizes"
python count_nonzero_blocks.py "$TEST_MATRIX"
echo ""

# Test 2: Custom block sizes
echo "Test 2: Custom block sizes (4, 8, 16)"
python count_nonzero_blocks.py "$TEST_MATRIX" -b 4 8 16
echo ""

# Test 3: With output file
echo "Test 3: Save to JSON file"
python count_nonzero_blocks.py "$TEST_MATRIX" -o test_output.json --pretty
cat test_output.json
rm test_output.json
echo ""

echo "All tests completed successfully!"
