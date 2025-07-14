#!/bin/bash

# === Build Guard Script ===
# Sicherstellen, dass alle Docker- und Compose-Komponenten baubar und korrekt sind

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() {
  echo -e "$1$2${NC}"
}

log "$GREEN" "[1/5] Prüfe docker-compose.yml ..."
docker compose config > /dev/null || {
  log "$RED" "Fehler in docker-compose.yml"
  exit 1
}

log "$GREEN" "[2/5] Prüfe auf .dockerignore ..."
if [ ! -f .dockerignore ]; then
  log "$YELLOW" ".dockerignore fehlt - sollte erstellt werden."
else
  log "$GREEN" ".dockerignore gefunden."
fi

log "$GREEN" "[3/5] Prüfe Dockerfiles in agents/..."
MISSING=0
for AGENT in agents/*; do
  if [ -d "$AGENT" ]; then
    if [ ! -f "$AGENT/Dockerfile" ]; then
      log "$YELLOW" "Fehlender Dockerfile in $AGENT"
      MISSING=1
    fi
  fi

done
if [ $MISSING -eq 1 ]; then
  log "$RED" "Mindestens ein Agent hat keinen Dockerfile."
  exit 1
fi

log "$GREEN" "[4/5] Testweise Build-Check aller Services ..."
docker compose build --dry-run || {
  log "$RED" "Build-Vorprüfung fehlgeschlagen."
  exit 1
}

log "$GREEN" "[5/5] OK - Alles bereit zum Build 🚀"
exit 0
