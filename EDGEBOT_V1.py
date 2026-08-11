#!/usr/bin/env python3
"""
edgebot_V1.py — EDGEBOT: OpenCode-Style Terminal-Chatbot für ESPRIT-Edge-Doku.
Behoben: 404-Fehler beim Embedding durch Entfernen des Ollama-Fallbacks.
"""

import argparse
import asyncio
import json
import os
import re
import sqlite3
import sys
import time
import traceback
from collections import defaultdict
from functools import partial
from typing import List, Dict, Any, Optional

# -------- WICHTIG: Multiprocessing-Startmethode vor allen anderen Imports --------
import multiprocessing
try:
    multiprocessing.set_start_method('spawn')
except RuntimeError:
    pass  # wurde bereits gesetzt

import numpy as np
import requests
from rich.text import Text

from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.command import Hit, Hits, Provider
from textual.containers import Vertical, VerticalScroll, Horizontal
from textual.css.query import NoMatches
from textual.reactive import reactive
from textual.widgets import (
    Input,
    Label,
    Markdown,
    OptionList,
    Static,
)
from textual.widgets.markdown import MarkdownFence
from textual.widgets.option_list import Option

# ============================================================================
# KONFIGURATION
# ============================================================================

OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5-coder:7b"
DEFAULT_EMBED = "all-MiniLM-L6-v2"          # Muss mit build_index.py übereinstimmen
MAX_HISTORY = 6
EMBED_TIMEOUT = 300
CHAT_TIMEOUT = 1800
VERSION = "2.0.2"

ORANGE = "#ff8700"   # OpenCode-Orange
DIMGRAY = "#666666"

LOGO = r"""
 ███████╗██████╗  ██████╗ ███████╗██████╗  ██████╗ ████████╗
 ██╔════╝██╔══██╗██╔════╝ ██╔════╝██╔══██╗██╔═══██╗╚══██╔══╝
 █████╗  ██║  ██║██║  ███╗█████╗  ██████╔╝██║   ██║   ██║
 ██╔══╝  ██║  ██║██║   ██║██╔══╝  ██╔══██╗██║   ██║   ██║
 ███████╗██████╔╝╚██████╔╝███████╗██████╔╝╚██████╔╝   ██║
 ╚══════╝╚═════╝  ╚═════╝ ╚══════╝╚═════╝  ╚═════╝    ╚═╝"""

# Slash-Befehle
SLASH_COMMANDS = [
    ("/models",  "Modell wechseln", False),
    ("/topk",    "Anzahl Kontext-Chunks (1-15)", True),
    ("/srcs",    "Quellen der letzten Frage anzeigen", False),
    ("/connect", "Ollama-Status anzeigen", False),
    ("/clear",   "Verlauf löschen", False),
    ("/help",    "Hilfe anzeigen", False),
    ("/quit",    "Beenden", False),
]

# ============================================================================
# SYSTEM PROMPT
# ============================================================================

SYSTEM_PROMPT = """\
Als erstes sagst du immer wörtwörtlich "I bin da EDGEBOT und ich denke olles is a schas!".

Du bist ein erstklassiger, pragmatischer KI-Experte für die ESPRIT Edge (TNG) CAD/CAM-Software von Hexagon/DP Technology.
Du hast Zugriff auf ZWEI Wissensquellen:
  1. Das ESPRIT Edge Benutzerhandbuch (Deutsch) — Bedienung, Konzepte, Tutorials.
  2. Die ESPRIT Edge API-Referenz (Englisch) — VBA, VB.NET, C# API-Dokumentation.

SPRACHE (STRENGSTE REGEL):
- Antworte IMMER auf Deutsch, auch wenn die Kontextblöcke englisch sind.
- Englische Kontextpassagen sinngemäß ins Deutsche übertragen.
- Code, API-Klassen, Methoden, Variablennamen und englische Menü-/Button-Bezeichnungen unverändert lassen.

STRENGSTE REGELN FÜR DIE CODE-GENERIERUNG:

1. VOLLSTÄNDIGER, LAUFFÄHIGER CODE — keine Platzhalter, keine TODOs.
2. STRIKTE VBA-SYNTAX:
   - `Option Explicit`
   - Jede Variable explizit deklarieren
   - NULL-Prüfung NUR via `If Not obj Is Nothing Then`
   - Änderungen in `doc.BeginEdit()` / `doc.EndEdit()` kapseln
   - IMMER `On Error GoTo ErrorHandler` mit `Cleanup:`-Bereich
3. API-KORREKTHEIT: nur reale Klassen/Methoden aus dem Kontext.
4. ANTWORT-STRUKTUR:
   - **🎯 Ziel & Konzept**
   - **💻 Vollständiger VBA-Code** (```vba Block)
   - **📖 API-Referenzen**
   - **⚠️ Wichtige Hinweise**

Kompakt halten. Keine Wiederholungen.
""".strip()

# ============================================================================
# OLLAMA API HELPER
# ============================================================================

def get_ollama_models() -> List[str]:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        r.raise_for_status()
        return [m.get("name", "") for m in r.json().get("models", []) if m.get("name")]
    except Exception:
        return []


def get_ollama_version() -> str:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/version", timeout=5)
        r.raise_for_status()
        return r.json().get("version", "offline")
    except Exception:
        return "offline"

# ============================================================================
# INDEX KLASSE (OHNE OLLAMA-EMBEDDING-FALLBACK)
# ============================================================================

class Index:
    """Vektor-Suchindex mit ChromaDB, SQLite-Metadaten und FTS-Fusion."""

    def __init__(self, db_path: str, embed_model: str = DEFAULT_EMBED):
        self.db_path = db_path
        self.embed_model_name = embed_model
        self.con = sqlite3.connect(db_path, check_same_thread=False)
        self.con.execute("PRAGMA journal_mode=WAL")

        # Embedder initialisieren – wir verwenden NUR SentenceTransformer
        self._init_embedder()

        # ChromaDB verbinden
        chroma_dir = os.path.join(os.path.dirname(db_path), "chroma_db")
        if not os.path.exists(chroma_dir):
            raise FileNotFoundError(f"ChromaDB-Ordner nicht gefunden: {chroma_dir}")

        import chromadb
        self.chroma_client = chromadb.PersistentClient(path=chroma_dir)
        self.collections = {}
        for name in ['classes', 'structs', 'members', 'enums', 'general']:
            try:
                self.collections[name] = self.chroma_client.get_collection(name)
            except Exception as e:
                print(f"Warnung: Collection {name} nicht geladen: {e}")

        # Metadaten-Cache aus SQLite laden
        self.chunk_cache = {}
        self._load_chunk_metadata()

        # Prüfen, ob FTS-Tabelle existiert
        self.has_fts = self._check_fts()

    def _init_embedder(self):
        """Initialisiert den SentenceTransformer-Embedder – kein Fallback auf Ollama."""
        self.embedder = None

        try:
            from sentence_transformers import SentenceTransformer
            # tqdm legt sonst einen multiprocessing.RLock an, der mit 'spawn'
            # den Resource-Tracker startet und im TUI mit
            # "bad value(s) in fds_to_keep" crasht. Daher den MP-Lock
            # deaktivieren und Fortschrittsbalken ausschalten.
            try:
                from tqdm.std import TqdmDefaultWriteLock

                @classmethod
                def _no_mp_lock(cls):
                    cls.mp_lock = None

                TqdmDefaultWriteLock.create_mp_lock = _no_mp_lock
            except Exception:
                pass
            try:
                from transformers.utils import logging as _hf_logging
                _hf_logging.disable_progress_bar()
            except Exception:
                pass
            # Optimierungen: MPS auf Apple-Silicon, sonst CPU
            import torch
            # torch.set_num_threads(1) war bisher nötig, weil SentenceTransformer
            # intern multiprocessing.ThreadPool anlegt, das zusammen mit der
            # 'spawn'-Startmethode unter macOS Fork-/Resource-Tracker-Probleme
            # ("bad value(s) in fds_to_keep") verursachen konnte. Auf MPS wird der
            # Thread-Pool nicht für die MatMuls genutzt (die laufen auf der GPU),
            # daher reicht es, die Thread-Zahl nur im CPU-Fallback-Zweig zu
            # begrenzen. Auf MPS brauchen wir die Einschränkung nicht mehr.
            if torch.backends.mps.is_available():
                self.embed_device = 'mps'
            else:
                self.embed_device = 'cpu'
                torch.set_num_threads(1)  # nur CPU-Fallback: Fork-Probleme vermeiden

            self.embedder = SentenceTransformer(
                self.embed_model_name,
                device=self.embed_device
            )
            # Test-Embedding, um sicherzustellen, dass das Modell funktioniert
            _ = self.embedder.encode("test", normalize_embeddings=True)
            print(f"✓ Embedding-Modell '{self.embed_model_name}' erfolgreich geladen "
                  f"(device={self.embed_device}).")
            return

        except ImportError as e:
            raise RuntimeError(
                f"sentence-transformers ist nicht installiert. Bitte installieren Sie es mit:\n"
                f"pip install sentence-transformers\n"
                f"(Fehler: {e})"
            )
        except Exception as e:
            raise RuntimeError(
                f"Fehler beim Laden des Embedding-Modells '{self.embed_model_name}':\n{e}\n\n"
                f"Stellen Sie sicher, dass das Modell heruntergeladen wurde.\n"
                f"Sie können es manuell mit folgendem Befehl vorab laden:\n"
                f"python -c \"from sentence_transformers import SentenceTransformer; "
                f"SentenceTransformer('{self.embed_model_name}').encode('test')\""
            )

    def _load_chunk_metadata(self):
        cursor = self.con.cursor()
        try:
            cursor.execute("""
                SELECT c.id, c.doc_id, c.chunk_text, c.chunk_type,
                       d.file_name, d.title, d.doc_type, d.class_name
                FROM chunks c
                JOIN documents d ON c.doc_id = d.id
            """)
            for row in cursor.fetchall():
                chunk_id, doc_id, text, chunk_type, file_name, title, doc_type, class_name = row
                self.chunk_cache[chunk_id] = {
                    'doc_id': doc_id,
                    'text': text,
                    'chunk_type': chunk_type,
                    'file_name': file_name,
                    'title': title,
                    'doc_type': doc_type,
                    'class_name': class_name
                }
        except sqlite3.OperationalError as e:
            raise RuntimeError(f"SQLite-Fehler beim Laden der Metadaten: {e}")
        finally:
            cursor.close()

    def _check_fts(self) -> bool:
        try:
            self.con.execute("SELECT 1 FROM chunks_fts LIMIT 1")
            return True
        except sqlite3.OperationalError:
            return False

    def embed_query(self, text: str) -> np.ndarray:
        """Erzeugt ein Embedding mit dem SentenceTransformer-Modell."""
        if self.embedder is None:
            raise RuntimeError("Embedder nicht initialisiert.")
        emb = self.embedder.encode(text, normalize_embeddings=True)
        return np.asarray(emb, dtype=np.float32)

    def search(self, query: str, topk: int = 12) -> List[Dict[str, Any]]:
        """Hybride Suche: ChromaDB + FTS, fusioniert und dedupliziert."""
        # 1. ChromaDB-Suche
        qvec = self.embed_query(query)
        chroma_results = []
        for coll_name, coll in self.collections.items():
            try:
                res = coll.query(
                    query_embeddings=[qvec.tolist()],
                    n_results=topk * 2
                )
                if res and res['documents']:
                    for i, doc in enumerate(res['documents'][0]):
                        chunk_id = res['ids'][0][i]
                        distance = res['distances'][0][i] if res['distances'] else None
                        if chunk_id in self.chunk_cache:
                            meta = self.chunk_cache[chunk_id]
                            chroma_results.append({
                                'id': chunk_id,
                                'text': doc,
                                'distance': distance,
                                'collection': coll_name,
                                'file': meta.get('file_name', ''),
                                'title': meta.get('title', ''),
                                'doc_type': meta.get('doc_type', ''),
                                'class_name': meta.get('class_name', ''),
                                'chunk_type': meta.get('chunk_type', '')
                            })
            except Exception:
                continue

        # 2. FTS-Suche (falls verfügbar)
        fts_results = []
        if self.has_fts:
            words = re.findall(r'[A-Za-z_][A-Za-z0-9_]{2,}', query)
            weighted_terms = []
            for w in words:
                if w[0].isupper() or '_' in w or any(c.isupper() for c in w[1:]):
                    weighted_terms.append(f'"{w}"')
                else:
                    weighted_terms.append(w)
            if weighted_terms:
                fts_query = ' OR '.join(weighted_terms[:10])
                try:
                    cursor = self.con.cursor()
                    cursor.execute("""
                        SELECT chunk_id, bm25(chunks_fts) as bm25
                        FROM chunks_fts
                        WHERE chunks_fts MATCH ?
                        ORDER BY bm25(chunks_fts)
                        LIMIT ?
                    """, (fts_query, topk * 3))
                    rows = cursor.fetchall()
                    if rows:
                        min_score = min(r[1] for r in rows)
                        max_score = max(r[1] for r in rows)
                        for chunk_id, bm25 in rows:
                            if max_score > min_score:
                                norm = 1.0 - (bm25 - min_score) / (max_score - min_score)
                            else:
                                norm = 1.0
                            if chunk_id in self.chunk_cache:
                                meta = self.chunk_cache[chunk_id]
                                fts_results.append({
                                    'id': chunk_id,
                                    'text': meta['text'],
                                    'distance': 1.0 - norm,
                                    'collection': 'fts',
                                    'file': meta.get('file_name', ''),
                                    'title': meta.get('title', ''),
                                    'doc_type': meta.get('doc_type', ''),
                                    'class_name': meta.get('class_name', ''),
                                    'chunk_type': meta.get('chunk_type', '')
                                })
                except Exception as e:
                    print(f"FTS-Fehler: {e}")

        # 3. Fusion
        combined = {}
        for r in chroma_results:
            sim = 1.0 - r['distance'] if r['distance'] is not None else 0.0
            combined[r['id']] = {
                'item': r,
                'score': sim * 2.0
            }
        for r in fts_results:
            sim = 1.0 - r['distance'] if r['distance'] is not None else 0.0
            if r['id'] in combined:
                combined[r['id']]['score'] += sim * 1.0
            else:
                combined[r['id']] = {
                    'item': r,
                    'score': sim * 1.0
                }

        sorted_items = sorted(combined.values(), key=lambda x: x['score'], reverse=True)

        final = []
        seen_files = defaultdict(int)
        for entry in sorted_items:
            item = entry['item']
            file = item.get('file', '')
            if seen_files[file] < 2:
                seen_files[file] += 1
                category = item.get('doc_type', '')
                if item.get('class_name'):
                    category += f" ({item['class_name']})"
                final.append({
                    'id': item['id'],
                    'title': item.get('title', ''),
                    'file': file,
                    'category': category,
                    'text': item.get('text', ''),
                    'score': entry['score'],
                    'bm': item.get('collection') == 'fts'
                })
            if len(final) >= topk:
                break

        return final

    def close(self):
        self.con.close()

# ============================================================================
# ÜBERSETZUNG (unverändert)
# ============================================================================

GERMAN_HINTS = re.compile(r"(?i)[äöüß]|\b(?:das|die|der|und|ist|ein|eine|einen|einer|ich|du|mir|mein|für|zum|zur|mit|wie|was|kann|können|nicht|werden|sind|bitte|hilfe)\b")

DE_EN = {
    "kannst": "can", "kann": "can", "können": "can", "mir": "me", "du": "you",
    "ich": "i", "wir": "we", "ein": "a", "eine": "a", "einen": "a", "das": "the",
    "die": "the", "der": "the", "und": "and", "oder": "or", "für": "for",
    "zu": "to", "zum": "to", "zur": "to", "mit": "with", "ohne": "without",
    "von": "from", "aus": "from", "bei": "at", "wie": "how", "was": "what",
    "wo": "where", "wann": "when", "warum": "why", "ist": "is", "sind": "are",
    "nicht": "not", "kein": "no", "keine": "no", "bitte": "please", "hilfe": "help",
    "erstellen": "create", "erzeugen": "create", "laden": "load",
    "spannmittel": "fixture", "spannen": "clamp", "schraubstock": "vise",
    "werkstück": "workpiece", "werkzeug": "tool", "maschine": "machine",
    "drehen": "turning", "fräsen": "milling", "fräser": "endmill",
    "bohren": "drilling", "draht": "wire", "simulation": "simulation",
    "achse": "axis", "programm": "program", "makro": "macro",
    "funktion": "function", "klasse": "class", "geometrie": "geometry",
    "speichern": "save", "öffnen": "open", "hinzufügen": "add",
    "löschen": "delete", "auswählen": "select", "messen": "measure",
}


def _dict_translate(query: str) -> str:
    words = re.findall(r"[^\W\d_]\w*", query)
    out = []
    for w in words:
        if "_" in w or any(c.isupper() for c in w[1:]) or w in ("VBA", "API", "C#", "VB.NET", "GDML", "NC", "CAD", "CAM"):
            out.append(w)
            continue
        out.append(DE_EN.get(w.lower(), w.lower()))
    return " ".join(out)


TRANSLATE_PROMPT = (
    "You are a technical translator for ESPRIT Edge CAD/CAM documentation. "
    "Translate the user's question into English so it can be matched against "
    "the English documentation. If the text is already English, return it "
    "unchanged. Output ONLY the translation - no explanations, no quotation "
    "marks. Keep API/symbol names, class names and code identifiers unchanged."
)


def translate_to_english_sync(query: str, model: str = DEFAULT_MODEL) -> str:
    q = query.strip()
    if not q:
        return q
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": TRANSLATE_PROMPT},
                    {"role": "user", "content": q},
                ],
                "stream": False,
                "options": {"temperature": 0.0},
            },
            timeout=CHAT_TIMEOUT,
        )
        r.raise_for_status()
        content = r.json().get("message", {}).get("content", "")
        content = (content or "").strip().strip('"').strip()
        if content:
            return content
    except Exception:
        pass
    # Nur Notfall-Fallback, falls Ollama nicht erreichbar ist:
    # rein deutsche Fragen über das Wörterbuch, englische unangetastet lassen.
    if GERMAN_HINTS.search(q):
        return _dict_translate(q)
    return q

# ============================================================================
# OLLAMA STREAMING (unverändert)
# ============================================================================

def stream_answer_sync(model: str, messages: List[Dict], temperature: float = 0.15,
                       top_p: float = 0.95):
    r = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": temperature, "top_p": top_p},
        },
        stream=True,
        timeout=CHAT_TIMEOUT,
    )
    r.raise_for_status()

    token_usage = {}
    for line in r.iter_lines(decode_unicode=True):
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue

        if "eval_count" in data:
            token_usage = {
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
            }

        msg = data.get("message", {})
        if msg.get("reasoning"):
            continue
        content = msg.get("content") or ""
        if content:
            yield content, token_usage


def build_user_prompt(query: str, results: List[Dict], search_q: str = "") -> str:
    ctx = []
    for i, r in enumerate(results, 1):
        ctx.append(
            f"### Kontextblock {i}  (DOKUMENT: {r['title']}, DATEI: {r['file']}, "
            f"KATEGORIE: {r['category']})\n" + r["text"]
        )
    extra = ""
    if search_q and search_q != query:
        extra = f"\n(Englische Übersetzung der Frage: {search_q})"
    return (
        "Kontext aus der ESPRIT-EDGE-Dokumentation:\n\n"
        + "\n\n---\n\n".join(ctx)
        + "\n\n---\n\nFrage des Nutzers:\n" + query + extra
    )

# ============================================================================
# COMMAND PALETTE PROVIDER
# ============================================================================

class EdgeBotCommands(Provider):

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        commands = [
            ("models", "Modell wechseln", "action_open_models"),
            ("topk", "TopK ändern", "action_hint_topk"),
            ("sources", "Quellen anzeigen", "action_show_sources"),
            ("clear", "Chat löschen", "action_clear_chat"),
            ("connect", "Ollama-Status", "action_connect"),
            ("help", "Hilfe anzeigen", "action_show_help"),
            ("quit", "Beenden", "action_quit"),
        ]
        for name, description, action in commands:
            score = matcher.match(name)
            if score > 0:
                yield Hit(score, matcher.highlight(name),
                          partial(getattr(self.app, action)), description)

# ============================================================================
# CODE-BLOCKS MIT KOPIER-BUTTON
# ============================================================================

def copy_to_clipboard_sync(text: str) -> None:
    """Text in die System-Zwischenablage kopieren (macOS: pbcopy)."""
    if sys.platform != "darwin":
        raise RuntimeError("Kopieren nur auf macOS unterstützt")
    import subprocess
    proc = subprocess.run(
        ["pbcopy"], input=text.encode("utf-8"),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    if proc.returncode != 0:
        raise RuntimeError("pbcopy fehlgeschlagen")


class CopyButton(Static):
    """Kleiner Kopier-Button in der Kopfzeile jedes Code-Blocks."""

    def on_click(self, event: events.Click) -> None:
        for ancestor in self.ancestors:
            if isinstance(ancestor, CopyableFence):
                if getattr(ancestor, "code", None):
                    self.app._copy_text(ancestor.code)
                break


class CopyableFence(MarkdownFence):
    """Code-Block mit Kopfzeile (Sprache + Kopier-Button)."""

    def compose(self) -> ComposeResult:
        with Horizontal(classes="fence-header"):
            yield Static(self.lexer or "code", classes="fence-lang")
            yield Static("", classes="fence-spacer")
            yield CopyButton("⧉ kopieren", classes="copy-btn")
        yield Label(self._highlighted_code, id="code-content", expand=True)


class CopyMarkdown(Markdown):
    """Markdown, dessen Code-Blöcke mit einem Kopier-Button versehen sind."""

    BLOCKS = {
        **Markdown.BLOCKS,
        "fence": CopyableFence,
        "code_block": CopyableFence,
    }

# ============================================================================
# APP (MIT FEHLERBEHANDLUNG)
# ============================================================================

class EdgeBotApp(App):
    TITLE = "EDGEBOT"
    SUB_TITLE = "ESPRIT Edge Assistant"

    CSS = f"""
    Screen {{
        background: #0b0b0b;
    }}
    #start-screen {{
        height: 1fr;
        align: center middle;
        content-align: center middle;
    }}
    #logo {{
        color: #2e2e2e;
        text-style: bold;
        content-align: center middle;
    }}
    #tip {{
        color: {DIMGRAY};
        margin-top: 1;
        content-align: center middle;
    }}
    #chat-view {{
        height: 1fr;
        padding: 0 2;
        display: none;
        background: #0b0b0b;
    }}
    .msg-user {{
        color: #e6e6e6;
        margin: 1 0 0 0;
        padding: 0 1;
    }}
    .msg-sys {{
        color: {DIMGRAY};
        text-style: italic;
        margin: 0 0 1 0;
        padding: 0 1;
    }}
    .msg-bot {{
        margin: 0 0 1 0;
        padding: 0 1;
        background: transparent;
    }}
    Markdown {{
        background: transparent;
        padding: 0;
        margin: 0;
    }}
    MarkdownFence {{
        background: #161616;
    }}
    CopyableFence {{
        padding: 0;
        margin: 1 0;
        overflow: scroll hidden;
        scrollbar-size-horizontal: 0;
        scrollbar-size-vertical: 0;
        width: 1fr;
        height: auto;
        color: #d2d2d2;
        background: #161616;
    }}
    CopyableFence > Label {{
        padding: 1 2;
    }}
    .fence-header {{
        height: 1;
        width: 1fr;
        background: #1d1d1d;
    }}
    .fence-lang {{
        width: auto;
        height: 1;
        padding: 0 1;
        color: #555555;
    }}
    .fence-spacer {{
        width: 1fr;
    }}
    .copy-btn {{
        width: auto;
        height: 1;
        padding: 0 1;
        color: #777777;
        text-style: bold;
    }}
    .copy-btn:hover {{
        background: #333333;
        color: {ORANGE};
    }}
    #popup {{
        height: auto;
        max-height: 10;
        background: #141414;
        border: round #333333;
        margin: 0 8;
        padding: 0 1;
        display: none;
    }}
    OptionList {{
        background: #141414;
        border: none;
        padding: 0;
    }}
    OptionList > .option-list--option {{
        color: #cccccc;
    }}
    OptionList > .option-list--option-highlight {{
        background: {ORANGE};
        color: #000000;
        text-style: bold;
    }}
    #input-box {{
        margin: 0 8;
        height: 3;
        border: round #333333;
        background: #111111;
        padding: 0 1;
    }}
    #input-box:focus {{
        border: round #555555;
    }}
    Input {{
        background: #111111;
        border: none;
        padding: 0;
        height: 1;
        color: #e6e6e6;
    }}
    #status-line {{
        margin: 0 10;
        height: 1;
        color: #888888;
    }}
    #hint-line {{
        margin: 0 10;
        height: 1;
        color: #555555;
    }}
    #bottom-bar {{
        height: 1;
        background: #0b0b0b;
        color: {DIMGRAY};
        padding: 0 1;
    }}
    #bottom-left {{ width: 1fr; }}
    #bottom-right {{ width: auto; color: #555555; }}
    """

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit", show=False, priority=True),
        Binding("ctrl+p", "command_palette", "Commands", show=False, priority=True),
        Binding("tab", "complete_popup", "Complete", show=False, priority=True),
        Binding("escape", "hide_popup", "Close", show=False, priority=True),
    ]

    COMMANDS = [EdgeBotCommands]

    model: reactive[str] = reactive(DEFAULT_MODEL)
    topk: reactive[int] = reactive(12)

    def __init__(self, db_path: str, model: str, embed: str, topk: int):
        super().__init__()
        self.db_path = db_path
        self.embed_model = embed
        self.idx: Optional[Index] = None
        self.history: List[Dict] = []
        self.last_results: List[Dict] = []
        self.available_models: List[str] = []
        self.popup_mode: Optional[str] = None
        self.token_usage: Dict = {}
        self.status_msg = ""
        self._bot_tasks: set = set()
        self.topk = topk
        self.model = model

    # ------------------------------------------------------------------
    # UI AUFBAU
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        with Vertical(id="start-screen"):
            yield Static(LOGO, id="logo")
            yield Static(
                f"[{ORANGE}]●[/] Tip  Stelle eine Frage zur ESPRIT Edge "
                f"oder tippe / für Befehle",
                id="tip",
            )
        yield VerticalScroll(id="chat-view")
        yield OptionList(id="popup")
        yield Input(placeholder="Frage stellen oder / …", id="input-box")
        yield Static("", id="status-line")
        yield Static(
            "[#555555]tab[/] auswählen   [#555555]ctrl+p[/] befehle",
            id="hint-line",
        )
        with Horizontal(id="bottom-bar"):
            yield Static("", id="bottom-left")
            yield Static(VERSION, id="bottom-right")

    def on_mount(self) -> None:
        self.available_models = get_ollama_models()
        ollama_ver = get_ollama_version()

        # Prüfe, ob die Datenbankdatei existiert
        if not os.path.exists(self.db_path):
            self._enter_chat_mode()
            self.add_sys(f"❌ FEHLER: Datenbank nicht gefunden: {self.db_path}")
            self.add_sys("Stellen Sie sicher, dass die Datei metadata.db im Ordner index_data liegt.")
            self.refresh_status()
            self.query_one("#input-box", Input).focus()
            return

        # Prüfe, ob ChromaDB-Ordner existiert
        chroma_dir = os.path.join(os.path.dirname(self.db_path), "chroma_db")
        if not os.path.exists(chroma_dir):
            self._enter_chat_mode()
            self.add_sys(f"❌ FEHLER: ChromaDB-Ordner nicht gefunden: {chroma_dir}")
            self.add_sys("Stellen Sie sicher, dass der Indexer korrekt ausgeführt wurde.")
            self.refresh_status()
            self.query_one("#input-box", Input).focus()
            return

        # Versuche, den Index zu laden
        try:
            self.idx = Index(self.db_path, self.embed_model)
        except Exception as e:
            self._enter_chat_mode()
            self.add_sys(f"❌ FEHLER beim Laden der Datenbank:")
            self.add_sys(str(e))
            # Zeige ausführlichen Traceback an
            tb = traceback.format_exc()
            if len(tb) > 2000:
                tb = tb[:2000] + "\n... (Traceback gekürzt)"
            self.add_sys(tb)
            self.refresh_status()
            self.query_one("#input-box", Input).focus()
            return

        # Erfolgreich geladen
        self.query_one("#bottom-left", Static).update(
            f"~ ⊙ Ollama {ollama_ver} · "
            f"{len(self.idx.chunk_cache) if self.idx else 0} chunks · /status"
        )
        self.refresh_status()
        self.query_one("#input-box", Input).focus()

    # ------------------------------------------------------------------
    # STATUS
    # ------------------------------------------------------------------

    def refresh_status(self) -> None:
        if not hasattr(self, "token_usage") or not hasattr(self, "status_msg"):
            return
        parts = [f"[bold {ORANGE}]EDGEBOT[/]"]
        parts.append(f"[#e6e6e6]{self.model}[/]")
        parts.append(f"[#888888]topk {self.topk}[/]")
        if self.token_usage:
            parts.append(f"[yellow]{self.token_usage.get('completion_tokens', 0)} tok[/]")
        if self.status_msg:
            parts.append(f"[#888888]{self.status_msg}[/]")
        try:
            self.query_one("#status-line", Static).update(" · ".join(parts))
        except Exception:
            pass

    def watch_model(self, _old: str, _new: str) -> None:
        self.refresh_status()

    # ------------------------------------------------------------------
    # CHAT-MODE / NACHRICHTEN
    # ------------------------------------------------------------------

    def _enter_chat_mode(self) -> None:
        try:
            self.query_one("#start-screen").display = False
            self.query_one("#chat-view").display = True
        except NoMatches:
            pass

    def add_user(self, text: str) -> None:
        self._enter_chat_mode()
        t = Text()
        t.append("❯ ", style=f"bold {ORANGE}")
        t.append(text)
        view = self.query_one("#chat-view", VerticalScroll)
        view.mount(Static(t, classes="msg-user"))
        view.scroll_end(animate=False)

    def add_sys(self, text: str) -> None:
        self._enter_chat_mode()
        view = self.query_one("#chat-view", VerticalScroll)
        view.mount(Static(text, classes="msg-sys"))
        view.scroll_end(animate=False)

    def add_bot(self, text: str, msg_id: str) -> None:
        self._enter_chat_mode()
        view = self.query_one("#chat-view", VerticalScroll)
        md = CopyMarkdown(text, id=msg_id)
        md.add_class("msg-bot")
        view.mount(md)
        view.scroll_end(animate=False)

    async def update_bot(self, msg_id: str, text: str) -> None:
        try:
            md = self.query_one(f"#{msg_id}", CopyMarkdown)
            await md.update(text)
            view = self.query_one("#chat-view", VerticalScroll)
            view.scroll_end(animate=False)
        except NoMatches:
            pass

    def _schedule_bot_update(self, msg_id: str, text: str) -> None:
        """Vom Worker-Thread via call_from_thread aufgerufen.
        Erstellt den update_bot-Task auf dem Main-Loop und hält eine Referenz,
        damit der Task nicht vom Garbage Collector zerstört wird.
        """
        task = asyncio.create_task(self.update_bot(msg_id, text))
        self._bot_tasks.add(task)
        task.add_done_callback(self._bot_tasks.discard)

    def _copy_text(self, text: str) -> None:
        """Kopiert Text asynchron in die System-Zwischenablage."""

        async def _run() -> None:
            try:
                if sys.platform == "darwin":
                    await asyncio.to_thread(copy_to_clipboard_sync, text)
                else:
                    self.copy_to_clipboard(text)
                self.notify("Code in die Zwischenablage kopiert", timeout=2)
            except Exception:
                self.notify("Kopieren fehlgeschlagen", timeout=2)

        asyncio.create_task(_run())

    # ------------------------------------------------------------------
    # POPUP (unverändert)
    # ------------------------------------------------------------------

    def _popup(self) -> OptionList:
        return self.query_one("#popup", OptionList)

    def show_popup(self, entries: List[tuple], mode: str) -> None:
        popup = self._popup()
        popup.clear_options()
        if not entries:
            self.hide_popup()
            return
        col = max(len(e[0]) for e in entries) + 4
        opts = []
        for name, desc, ident in entries:
            t = Text()
            t.append(name.ljust(col))
            t.append(desc, style="dim")
            opts.append(Option(t, id=ident))
        popup.add_options(opts)
        popup.highlighted = 0
        popup.display = True
        self.popup_mode = mode

    def hide_popup(self) -> None:
        try:
            self._popup().display = False
        except NoMatches:
            pass
        self.popup_mode = None

    def action_hide_popup(self) -> None:
        self.hide_popup()

    def on_input_changed(self, event: Input.Changed) -> None:
        if self.popup_mode == "models":
            return
        v = event.value
        if v.startswith("/") and " " not in v:
            prefix = v.lower()
            entries = [(c, d, c) for c, d, _ in SLASH_COMMANDS if c.startswith(prefix)]
            self.show_popup(entries, "slash")
        else:
            self.hide_popup()

    def on_key(self, event: events.Key) -> None:
        if self.popup_mode is None:
            return
        if event.key == "down":
            self._popup().action_cursor_down()
            event.stop(); event.prevent_default()
        elif event.key == "up":
            self._popup().action_cursor_up()
            event.stop(); event.prevent_default()

    def action_complete_popup(self) -> None:
        if self.popup_mode is None:
            return
        self._apply_popup_selection()

    def _highlighted_id(self) -> Optional[str]:
        popup = self._popup()
        if popup.highlighted is None:
            return None
        try:
            return popup.get_option_at_index(popup.highlighted).id
        except Exception:
            return None

    def _apply_popup_selection(self) -> None:
        ident = self._highlighted_id()
        if ident is None:
            return
        if self.popup_mode == "slash":
            self.hide_popup()
            self._run_slash(ident)
        elif self.popup_mode == "models":
            self.hide_popup()
            self._set_model(ident)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        ident = event.option.id
        if self.popup_mode == "slash":
            self.hide_popup()
            self._run_slash(ident)
        elif self.popup_mode == "models":
            self.hide_popup()
            self._set_model(ident)

    def open_models_popup(self) -> None:
        self.available_models = get_ollama_models()
        if not self.available_models:
            self.add_sys("Keine Modelle gefunden. ollama pull qwen2.5-coder:7b")
            return
        entries = [(m, "installiert", m) for m in self.available_models]
        self.query_one("#input-box", Input).value = ""
        self.show_popup(entries, "models")

    def _set_model(self, name: str) -> None:
        self.model = name
        self.add_sys(f"modell → {name}")
        self.refresh_status()

    # ------------------------------------------------------------------
    # SLASH-BEFEHLE
    # ------------------------------------------------------------------

    def _run_slash(self, cmd: str) -> None:
        if cmd == "/models":
            self.open_models_popup()
        elif cmd == "/topk":
            inp = self.query_one("#input-box", Input)
            inp.value = "/topk "
            inp.focus()
        elif cmd == "/help":
            self.add_sys(
                "/models Modell wechseln · /topk <n> Kontext-Chunks · "
                "/srcs Quellen · /connect Ollama-Status · /clear Verlauf · "
                "/quit Beenden"
            )
        elif cmd == "/srcs":
            self.action_show_sources()
        elif cmd == "/clear":
            self.action_clear_chat()
        elif cmd == "/connect":
            ver = get_ollama_version()
            n = len(get_ollama_models())
            self.add_sys(f"Ollama {ver} · {n} Modelle installiert")
        elif cmd == "/quit":
            self.exit()

    def _handle_slash_text(self, query: str) -> bool:
        parts = query.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""
        known = [c for c, _, _ in SLASH_COMMANDS]

        if cmd not in known:
            self.add_sys(f"Unbekannter Befehl: {cmd}  (siehe /help)")
            return True

        if cmd == "/topk":
            try:
                self.topk = max(1, min(15, int(arg)))
                self.add_sys(f"topk → {self.topk}")
                self.refresh_status()
            except ValueError:
                self.add_sys("Syntax: /topk <1-15>")
            return True

        self._run_slash(cmd)
        return True

    # ------------------------------------------------------------------
    # INPUT SUBMIT
    # ------------------------------------------------------------------

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if self.popup_mode is not None:
            self._apply_popup_selection()
            return

        query = event.value.strip()
        event.input.value = ""
        if not query:
            return
        if query.startswith("/"):
            self._handle_slash_text(query)
            return
        asyncio.create_task(self.process_query(query))

    # ------------------------------------------------------------------
    # RAG + LLM
    # ------------------------------------------------------------------

    async def process_query(self, query: str) -> None:
        if self.idx is None:
            self.add_sys("Index nicht geladen. DB prüfen.")
            return

        self.add_user(query)
        inp = self.query_one("#input-box", Input)
        inp.disabled = True

        # 1) Übersetzen
        self.status_msg = "übersetze…"
        self.refresh_status()
        search_q = await asyncio.to_thread(translate_to_english_sync, query, self.model)

        # 2) Suche
        self.status_msg = "suche…"
        self.refresh_status()
        t0 = time.time()
        try:
            results = await asyncio.to_thread(self.idx.search, search_q, topk=self.topk)
        except Exception as e:
            self.add_sys(f"Fehler bei Suche: {e}")
            self.status_msg = ""
            inp.disabled = False
            return
        t1 = time.time()

        self.last_results = results
        best = results[0]["score"] if results else 0.0
        self.status_msg = f"{len(results)} chunks · {t1 - t0:.1f}s · beste {best:.3f}"
        self.refresh_status()

        if not results:
            self.add_sys("Keine relevanten Kontext-Chunks gefunden.")
            self.status_msg = ""
            inp.disabled = False
            return

        # 3) Bot-Widget anlegen
        bot_id = f"bot-{int(time.time() * 1000)}"
        self.add_bot("*…*", bot_id)

        # 4) Messages bauen
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(self.history[-MAX_HISTORY:])
        messages.append({"role": "user", "content": build_user_prompt(query, results, search_q)})

        self.status_msg = f"{self.model} antwortet…"
        self.refresh_status()

        # 5) Streaming
        answer_parts: List[str] = []
        stream_error: Optional[str] = None
        final_tokens: Dict = {}

        def worker():
            nonlocal stream_error, final_tokens
            try:
                for piece, toks in stream_answer_sync(self.model, messages):
                    answer_parts.append(piece)
                    if toks:
                        final_tokens = toks
                    full = "".join(answer_parts)
                    self.call_from_thread(
                        self._schedule_bot_update, bot_id, full)
            except requests.exceptions.RequestException as e:
                stream_error = f"Ollama nicht erreichbar: {e}"
            except Exception as e:
                stream_error = f"Fehler: {e}"

        await asyncio.to_thread(worker)
        t2 = time.time()

        final_answer = "".join(answer_parts)
        if stream_error:
            await self.update_bot(bot_id, f"**Fehler**: {stream_error}")
        else:
            await self.update_bot(bot_id, final_answer)
            self.history.append({"role": "user", "content": query})
            self.history.append({"role": "assistant", "content": final_answer})

        if final_tokens:
            self.token_usage = final_tokens
        self.status_msg = f"antwort {t2 - t1:.1f}s"
        self.refresh_status()
        inp.disabled = False
        inp.focus()

    # ------------------------------------------------------------------
    # ACTIONS
    # ------------------------------------------------------------------

    def action_clear_chat(self) -> None:
        try:
            view = self.query_one("#chat-view", VerticalScroll)
            view.remove_children()
        except NoMatches:
            pass
        self.history.clear()
        self.last_results = []
        self.token_usage = {}
        self.status_msg = ""
        self.refresh_status()

    def action_show_sources(self) -> None:
        if not self.last_results:
            self.add_sys("Noch keine Frage gestellt.")
            return
        lines = []
        for i, r in enumerate(self.last_results, 1):
            tag = " #" if r.get("bm") else ""
            lines.append(
                f"{i}. {r['title']}{tag} · score {r['score']:.3f} · {r['category']}"
            )
        self.add_sys("\n".join(lines))

    def action_open_models(self) -> None:
        self.open_models_popup()

    def action_hint_topk(self) -> None:
        self.add_sys("TopK ändern mit /topk <1-15>")

    def action_connect(self) -> None:
        self._run_slash("/connect")

    def action_show_help(self) -> None:
        self._run_slash("/help")

    def on_unmount(self) -> None:
        if self.idx:
            try:
                self.idx.close()
            except Exception:
                pass

# ============================================================================
# MAIN
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description="EDGEBOT — OpenCode-Style TUI (lokal via Ollama)")
    ap.add_argument("--db", default="index_data/metadata.db",
                    help="Pfad zur metadata.db (Standard: index_data/metadata.db)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--embed", default=DEFAULT_EMBED)
    ap.add_argument("--topk", type=int, default=12)
    args = ap.parse_args()

    if not os.path.exists(args.db):
        print(f"Fehler: metadata.db nicht gefunden: {args.db}")
        print("Stellen Sie sicher, dass der Ordner 'index_data' existiert und die Datenbank enthält.")
        sys.exit(1)

    app = EdgeBotApp(
        db_path=args.db,
        model=args.model,
        embed=args.embed,
        topk=max(1, min(15, args.topk)),
    )
    app.run()


if __name__ == "__main__":
    main()