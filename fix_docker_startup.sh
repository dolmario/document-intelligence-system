#!/bin/bash
echo "🔧 Fixing Docker Startup Issues"

# 1. Kompletter Stop und Cleanup
echo "Stopping all containers..."
docker compose down --volumes --remove-orphans

# 2. Docker Netzwerk cleanup
echo "Cleaning Docker networks..."
docker network prune -f

# 3. Restart Docker Desktop (Windows/Mac)
echo "⚠️  Please restart Docker Desktop now if on Windows/Mac"
echo "Press Enter when Docker Desktop is running..."
read

# 4. Verify Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "❌ Docker is not running!"
    exit 1
fi

# 5. Start PostgreSQL first and wait
echo "Starting PostgreSQL..."
docker compose up -d postgres
echo "Waiting for PostgreSQL..."
sleep 15

# 6. Check PostgreSQL health
echo "Checking PostgreSQL health..."
for i in {1..30}; do
    if docker compose exec -T postgres pg_isready -U semanticuserl -d semantic_doc_finder >/dev/null 2>&1; then
        echo "✅ PostgreSQL is ready"
        break
    fi
    echo -n "."
    sleep 2
done

# 7. Start remaining services
echo "Starting all services..."
docker compose up -d

# 8. Show status
echo "Service Status:"
docker compose ps

echo "✅ Startup fixed! Check logs with: docker compose logs -f"