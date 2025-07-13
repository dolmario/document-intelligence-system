# 📁 Document Intelligence System (WIP 🚧)

Ein modulares, DSGVO-konformes Dokumenten- und Wissensmanagement-System mit intelligenter Volltextsuche, semantischer Verknüpfung, automatischer Indexierung und OCR-Unterstützung.  

> ⚠️ **Dieses Projekt befindet sich noch im Aufbau.** Viele Funktionen sind experimentell oder unvollständig. Feedback und Beteiligung sind willkommen!

---

[![Tests](https://github.com/ [USERNAME]/document-intelligence-system/actions/workflows/test.yml/badge.svg)](https://github.com/ [USERNAME]/document-intelligence-system/actions/workflows/test.yml)
[![Docker](https://img.shields.io/docker/pulls/ [USERNAME]/doc-intelligence)](https://hub.docker.com/r/ [USERNAME]/doc-intelligence)
[![License](https://img.shields.io/badge/License-MIT-blue.svg )](LICENSE)

Ein modulares, DSGVO-konformes Dokumenten- und Wissensmanagement-System mit intelligenter Volltextsuche, semantischer Verknüpfung und KI-Unterstützung.

## 🚀 Features

- 🔍 **Intelligente Suche**: Volltext- und semantische Suche mit KI  
- 🤖 **Lokale OCR**: Tesseract-basierte Texterkennung  
- 🧠 **Lernfähig**: Verbessert sich durch Nutzungsmuster  
- 🔒 **DSGVO-konform**: Vollständig lokale Verarbeitung  
- 🔗 **Automatische Verknüpfungen**: Erkennt Zusammenhänge  
- 🎯 **N8N Integration**: Workflow-Automatisierung  
- 🤖 **Ollama Integration**: Lokale KI-Modelle  
- 📊 **Open WebUI**: Moderne Benutzeroberfläche  

## 📋 Voraussetzungen

| Komponente     | Anforderung                              |
|----------------|-------------------------------------------|
| **OS**         | Windows 11, Linux, macOS                  |
| **Python**     | 3.10+                                     |
| **Docker**     | Docker Desktop (Windows/Mac) oder Engine  |
| **RAM**        | Mindestens 8GB (16GB empfohlen)           |
| **GPU**        | NVIDIA GPU (optional)                     |
| **Speicher**   | 20GB+ freier Speicherplatz                |

## 🛠️ Installation

### Windows (PowerShell als Administrator)

```powershell
git clone https://github.com/ [USERNAME]/document-intelligence-system.git
cd document-intelligence-system
.\install.ps1
