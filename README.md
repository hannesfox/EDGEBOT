# EDGEBOT — Lokaler RAG-Chatbot für die ESPRIT-Edge-Dokumentation

EDGEBOT ist ein **vollständig lokaler** RAG-Chatbot für die **ESPRIT Edge (TNG)** CAD/CAM-Dokumentation von Hexagon/DP Technology.
Er läuft komplett auf dem eigenen Rechner – ohne Cloud, ohne API-Kosten.

- **Retrieval:** Hybrid-Suche über **ChromaDB** (Vektoren) + **SQLite/FTS5** (BM25-Volltext), basierend auf der Doxygen-API-Referenz.
- **Chat & Übersetzung:** **Ollama** (`qwen2.5-coder:7b`) lokal.
- **UI:** schlanke **Textual**-Terminal-TUI im OpenCode-Stil.

## Funktionen

- **Hybride Suche** – Vektor-Embeddings (`all-MiniLM-L6-v2`) fusioniert mit BM25-Volltextsuche über 5 Collections (`classes`, `structs`, `members`, `enums`, `general`).
- **Automatische Frage-Übersetzung** – deutsche Fragen werden vom LLM ins Englische übersetzt, damit sie besser auf die englische API-Referenz treffen.
- **Deutsche Antworten** – der Bot antwortet immer auf Deutsch, englische Kontextpassagen werden sinngemäß übertragen.
- **Streaming-Antworten** – Antworten erscheinen tokenweise.
- **Kopierbarer Code** – jeder Code-Block hat einen `⧉ kopieren`-Button (macOS `pbcopy`).
- **Slash-Befehle** – `/models`, `/topk`, `/srcs`, `/connect`, `/clear`, `/help`, `/quit`.
- **MPS-Beschleunigung** – nutzt die Apple-Silicon-GPU (Fallback: CPU).

## Projektstruktur

```
├── EDGEBOT_V1.py   # Textual-TUI-Chatbot (Ollama + Hybrid-Suche)
├── indexer.py      # Baut die Suchdatenbank aus Doxygen-HTML auf
├── index_data/     # Suchdatenbank (ChromaDB + metadata.db) – NICHT im Repo
└── API-Help-html/  # ESPRIT-Edge-Hilfedateien – NICHT im Repo
```

> **Hinweis:** `API-Help-html/` (die ESPRIT-Edge-Hilfedateien) und `index_data/`
> (die generierte Suchdatenbank) sind bewusst **nicht** im Repository enthalten.

## Voraussetzungen (Apple Silicon)

- **Nativer arm64-Python** (z. B. `python.org`-arm64-Installer, `pyenv install 3.13` oder Homebrew-Python) – **kein** Rosetta-Python.

  Sanity-Check:
  ```bash
  python -c "import platform; print(platform.machine())"   # muss arm64 liefern
  ```

- **Ollama** als native macOS-App (Metal-beschleunigt):
  ```bash
  ollama pull qwen2.5-coder:7b
  ```

## Installation

```bash
git clone https://github.com/hannesfox/EDGEBOT.git
cd EDGEBOT

pip install --upgrade pip
pip install torch sentence-transformers chromadb beautifulsoup4 tqdm
pip install "textual>=0.60" rich numpy requests
```

## Suchdatenbank erstellen

Die ESPRIT-Edge-Hilfedateien liegen lokal unter `./API-Help-html` (nicht Teil des Repos).
Daraus wird der Index aufgebaut:

```bash
# Standard (Eingabe ./API-Help-html, Ausgabe ./index_data, 4 Worker)
python indexer.py

# Explizit
python indexer.py --input-dir ./API-Help-html --output-dir ./index_data --workers 4
```

Am Ende sollte u. a. ausgegeben werden:

- `Chunks pro ChromaDB-Collection:` → alle 5 Collections sind befüllt.
- `FTS5-Einträge (chunks_fts):` → Anzahl entspricht den Chunks.
- `Embedding-Device:` → `mps` (Apple Silicon), sonst `cpu`.

> **Neuindexierung:** Wird die Struktur geändert, `index_data/` löschen und neu aufbauen:
> `rm -rf index_data`

## Chatbot starten

```bash
python EDGEBOT_V1.py --db index_data/metadata.db --topk 12
```

Beim Start erscheint u. a.:

```
✓ Embedding-Modell 'all-MiniLM-L6-v2' erfolgreich geladen (device=mps).
```

### Bedienung

| Eingabe | Bedeutung |
| --- | --- |
| Frage tippen | Antwort aus der Dokumentation erhalten (Antwort auf Deutsch, Code kopierbar) |
| `Tab` | Popup-Auswahl bestätigen |
| `Ctrl+P` | Befehlspalette |
| `Ctrl+Q` | Beenden |

### Slash-Befehle

| Befehl | Funktion |
| --- | --- |
| `/models` | Installierte Ollama-Modelle anzeigen/wechseln |
| `/topk <1-15>` | Anzahl Kontext-Chunks einstellen |
| `/srcs` | Quellen der letzten Frage anzeigen |
| `/connect` | Ollama-Status prüfen |
| `/clear` | Chatverlauf löschen |
| `/help` | Hilfe anzeigen |
| `/quit` | Beenden |

## Konfiguration

Alle relevanten Einstellungen stehen oben in `EDGEBOT_V1.py`:

```python
OLLAMA_URL   = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5-coder:7b"   # Chat-Modell
DEFAULT_EMBED = "all-MiniLM-L6-v2"   # Embedding-Modell (muss mit indexer.py übereinstimmen)
MAX_HISTORY  = 6                     # Verlauf, der an das LLM geht
```

CLI-Parameter des Bots:

```
--db      Pfad zur metadata.db     (Standard: index_data/metadata.db)
--model   Chat-Modell              (Standard: qwen2.5-coder:7b)
--embed   Embedding-Modell         (Standard: all-MiniLM-L6-v2)
--topk    Kontext-Chunks 1-15      (Standard: 12)
```

## Wie es funktioniert

1. **Indexer** liest die Doxygen-HTML-Dateien, chunked sie und erzeugt
   - Embeddings (`all-MiniLM-L6-v2`) in ChromaDB (5 Collections, Cosine),
   - Metadaten in SQLite (`metadata.db`, WAL-Modus),
   - eine FTS5-Tabelle für die BM25-Volltextsuche.
2. **EDGEBOT** übersetzt die (deutsche) Frage per LLM ins Englische.
3. Die Frage wird **hybrid** gesucht: ChromaDB-Embeddings + FTS5-BM25, fusioniert und dedupliziert.
4. Die besten Chunks werden zusammen mit der Frage an das LLM gegeben.
5. Das LLM streamt die Antwort – auf Deutsch, mit kopierbaren Code-Blöcken.

## Fehlerbehebung

| Problem | Lösung |
| --- | --- |
| `Embedding-Modell ... nicht geladen` | Modell einmalig vorladen: `python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"` |
| `Ollama nicht erreichbar` | Ollama-App starten, `ollama list` prüfen |
| `ChromaDB-Ordner nicht gefunden` | `indexer.py` ausführen (erzeugt `index_data/chroma_db`) |
| Antworten nur englisch | System-Prompt enthält die Regel „Antworte IMMER auf Deutsch" – Modelleinstellungen prüfen |
| Kopieren funktioniert nicht | `pbcopy` wird genutzt (macOS); andere Terminals als Terminal.app verwenden, falls nötig |

## Lizenz

[MIT](LICENSE)
