#!/bin/bash

# Build and run Docker container for face recognition with GPU/NPU support

set -e

echo "🚀 Building and running Face Recognition Docker container..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if docker-compose is available
if command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
elif command -v docker &> /dev/null && docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
else
    echo "❌ Neither docker-compose nor 'docker compose' is available."
    exit 1
fi

echo "📦 Building Docker image..."
$COMPOSE_CMD build

echo ""
echo "🔍 Testing GPU/NPU access..."
echo "Running device detection tests..."

# Run the test container
$COMPOSE_CMD run --rm face-recognition

echo ""
echo "🎯 To run the face recognition application:"
echo "docker-compose run --rm face-recognition python3 /app/main.py"
echo ""
echo "🎯 To run with live display:"
echo "docker-compose run --rm -e DISPLAY=\$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix face-recognition python3 /app/main.py"
echo ""
echo "🎯 To run the comprehensive test suite:"
echo "docker-compose run --rm face-recognition python3 /app/test_gpu_npu.py"
