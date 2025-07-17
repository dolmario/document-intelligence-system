.PHONY: help install up down logs clean reset models

help:
	@echo "Document Intelligence System V2 Commands:"
	@echo "  make install    - Initial setup"
	@echo "  make up         - Start all services"
	@echo "  make down       - Stop all services"
	@echo "  make logs       - View logs"
	@echo "  make clean      - Remove containers and volumes"
	@echo "  make reset      - Full reset (CAUTION: deletes all data)"
	@echo "  make models     - List available Ollama models"
	@echo "  make pull-model MODEL=<name> - Pull new model"

install:
	chmod +x install.sh
	./install.sh

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f

clean:
	docker compose down -v

reset:
	docker compose down -v
	rm -rf data/* logs/*
	docker compose build --no-cache

models:
	docker exec doc-intel-ollama ollama list

pull-model:
	docker exec doc-intel-ollama ollama pull $(MODEL)
