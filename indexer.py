"""
ESPRIT EDGE API Documentation Indexer - Korrigierte Version
Behebt Threading-Probleme mit SQLite, fixiert das Collection-Routing,
aktiviert echte FTS5-Hybridsuche und läuft nativ auf Apple-Silicon (M4).
"""

import argparse
import os
import json
import hashlib
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
import sqlite3
import threading
import pickle
from queue import Queue

from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
import chromadb
from chromadb.utils import embedding_functions

# ----------------------------------------------------------------------------
# Collection-Routing: doc_type-Werte aus den HTML-Seiten -> ChromaDB-Collection
# ----------------------------------------------------------------------------
DOC_TYPE_TO_COLLECTION = {
    'class': 'classes',
    'struct': 'structs',
    'member_list': 'members',
    'enum': 'enums',
}

FALLBACK_COLLECTION = 'general'

ALL_COLLECTIONS = ['classes', 'structs', 'members', 'enums', 'general']


def collection_for_doc_type(doc_type: str) -> str:
    """Mappt einen doc_type (aus der HTML-Extraktion) auf den Collection-Namen.

    Alles, was nicht explizit gemappt ist ('class', 'struct', 'member_list',
    'enum'), landet in der 'general'-Collection.
    """
    return DOC_TYPE_TO_COLLECTION.get(doc_type or '', FALLBACK_COLLECTION)


def get_device() -> str:
    """Wählt das PyTorch-Gerät: MPS auf Apple-Silicon, sonst CPU."""
    try:
        import torch
        if torch.backends.mps.is_available():
            return 'mps'
    except Exception:
        pass
    return 'cpu'


class ThreadSafeSQLite:
    """Thread-sicherer SQLite-Wrapper (WAL-Modus, Timeout, Transaktionen)"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.local = threading.local()
        self._create_tables()

    def _get_connection(self):
        """Holt eine thread-lokale SQLite-Verbindung"""
        if not hasattr(self.local, 'connection'):
            self.local.connection = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=30,          # verhindert sporadische "database is locked"
            )
            # WAL-Modus + synchronous=NORMAL: robuster paralleler Schreibzugriff
            self.local.connection.execute("PRAGMA journal_mode=WAL")
            self.local.connection.execute("PRAGMA synchronous=NORMAL")
            self.local.cursor = self.local.connection.cursor()
        return self.local.connection, self.local.cursor

    def _create_tables(self):
        """Erstellt die Tabellen (wird nur einmal ausgeführt)"""
        conn, cursor = self._get_connection()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                file_name TEXT,
                title TEXT,
                doc_type TEXT,
                class_name TEXT,
                inheritance TEXT,
                content TEXT,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                doc_id TEXT,
                chunk_text TEXT,
                chunk_type TEXT,
                start_pos INTEGER,
                end_pos INTEGER,
                FOREIGN KEY (doc_id) REFERENCES documents(id)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id TEXT,
                target_id TEXT,
                relation_type TEXT,
                FOREIGN KEY (source_id) REFERENCES documents(id),
                FOREIGN KEY (target_id) REFERENCES documents(id)
            )
        ''')

        # FTS5-Virtual-Table für die hybride Suche (BM25-Volltext)
        try:
            cursor.execute('''
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    chunk_id UNINDEXED,
                    chunk_text
                )
            ''')
        except sqlite3.OperationalError as e:
            raise RuntimeError(
                "Diese SQLite-Build hat keine FTS5-Unterstützung (für die "
                f"hybride Volltextsuche). Bitte Python mit SQLite-FTS5 nutzen. "
                f"(Fehler: {e})"
            )

        conn.commit()

    def execute(self, query: str, params: tuple = (), commit: bool = True):
        """Führt eine SQL-Abfrage aus (Commit nur bei Bedarf, sonst transaktional)"""
        conn, cursor = self._get_connection()
        cursor.execute(query, params)
        if commit:
            conn.commit()
        return cursor

    def begin(self):
        """Startet eine Transaktion auf der thread-lokalen Verbindung"""
        conn, _ = self._get_connection()
        conn.execute('BEGIN')

    def commit(self):
        """Committet die aktuelle Transaktion"""
        conn, _ = self._get_connection()
        conn.commit()

    def rollback(self):
        """Rollt die aktuelle Transaktion zurück"""
        conn, _ = self._get_connection()
        conn.rollback()

    def fetch_one(self, query: str, params: tuple = ()):
        """Führt eine Abfrage aus und gibt ein Ergebnis zurück"""
        _, cursor = self._get_connection()
        cursor.execute(query, params)
        return cursor.fetchone()

    def fetch_all(self, query: str, params: tuple = ()):
        """Führt eine Abfrage aus und gibt alle Ergebnisse zurück"""
        _, cursor = self._get_connection()
        cursor.execute(query, params)
        return cursor.fetchall()

    def close(self):
        """Schließt alle Verbindungen"""
        if hasattr(self.local, 'connection'):
            self.local.connection.close()


class ESPRITDocumentationIndexer:
    """Hauptklasse für die Indexierung der ESPRIT API Dokumentation"""

    def __init__(self, input_dir: str, output_dir: str = "./index_data"):
        """
        Initialisiert den Indexer

        Args:
            input_dir: Pfad zum Ordner mit den HTML-Dateien
            output_dir: Pfad für die Ausgabe der Index-Datenbank
        """
        self.input_dir = Path(input_dir).expanduser()
        self.output_dir = Path(output_dir).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Embedding-Device: MPS auf Apple-Silicon, sonst CPU (Fix 6)
        self.device = get_device()

        # Initialisiere ChromaDB
        self.chroma_path = self.output_dir / "chroma_db"
        self.chroma_client = chromadb.PersistentClient(path=str(self.chroma_path))

        # Erstelle Collections für verschiedene Dokumenttypen
        self.collections = {}
        self.init_collections()

        # Thread-sichere SQLite
        self.db_path = self.output_dir / "metadata.db"
        self.db = ThreadSafeSQLite(str(self.db_path))

        # Statistik
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'total_chunks': 0,
            'errors': 0,
            'no_content': 0
        }

        # Queue für Ergebnisse (thread-sicher)
        self.result_queue = Queue()

    def init_collections(self):
        """Initialisiert die ChromaDB Collections (cosine, normalisierte Embeddings)"""
        for name in ALL_COLLECTIONS:
            try:
                # Lösche existierende Collection falls vorhanden
                try:
                    self.chroma_client.delete_collection(name)
                except:
                    pass

                self.collections[name] = self.chroma_client.create_collection(
                    name=name,
                    # Cosine-Distanz: konsistent mit der Normalisierung beider
                    # Seiten (Indexer + edgebot) (Fix 3)
                    metadata={"hnsw:space": "cosine"},
                    embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                        model_name='all-MiniLM-L6-v2',
                        device=self.device,
                        normalize_embeddings=True
                    )
                )
            except Exception as e:
                print(f"Fehler beim Erstellen der Collection {name}: {e}")

    def extract_content_from_html(self, html_path: Path) -> Dict[str, Any]:
        """
        Extrahiert strukturierte Inhalte aus einer Doxygen HTML-Datei

        Args:
            html_path: Pfad zur HTML-Datei

        Returns:
            Dictionary mit extrahierten Inhalten
        """
        try:
            with open(html_path, 'r', encoding='utf-8', errors='replace') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')

            # Extrahiere den Hauptinhalt
            content = soup.find('div', class_='contents')
            if not content:
                content = soup.find('body')

            if not content:
                return None

            # Grundlegende Informationen
            result = {
                'file_name': html_path.name,
                'title': '',
                'content': '',
                'doc_type': 'general',
                'class_name': '',
                'inheritance': [],
                'methods': [],
                'attributes': [],
                'enums': []
            }

            # Titel erkennen
            title_elem = soup.find('div', class_='headertitle')
            if title_elem:
                title_text = title_elem.get_text().strip()
                result['title'] = title_text

                # Dokumenttyp erkennen
                if 'Class' in title_text and 'Reference' in title_text:
                    result['doc_type'] = 'class'
                    result['class_name'] = title_text.replace('Class Reference', '').strip()
                elif 'Struct' in title_text and 'Reference' in title_text:
                    result['doc_type'] = 'struct'
                    result['class_name'] = title_text.replace('Struct Reference', '').strip()
                elif 'Member List' in title_text:
                    result['doc_type'] = 'member_list'
                    # Extrahiere Klassennamen aus Member List
                    class_match = re.search(r'([A-Za-z_]\w*)\s+Member List', title_text)
                    if class_match:
                        result['class_name'] = class_match.group(1)
                elif 'Enum' in title_text or 'enum' in title_text:
                    result['doc_type'] = 'enum'
                    enum_match = re.search(r'([A-Za-z_]\w*)\s+(?:Enum|enum)', title_text)
                    if enum_match:
                        result['class_name'] = enum_match.group(1)

            # Vererbung extrahieren
            inheritance_div = soup.find('div', class_='inheritance')
            if inheritance_div:
                inheritance_links = inheritance_div.find_all('a')
                result['inheritance'] = [link.get_text().strip() for link in inheritance_links]

            # Member-Funktionen extrahieren (für Klassen und Strukturen)
            if result['doc_type'] in ['class', 'struct']:
                # Methoden aus der Member-Liste extrahieren
                for mem_item in soup.find_all(['tr', 'div'],
                                              class_=['memitem', 'memproto', 'memItemLeft', 'memItemRight']):
                    try:
                        # Suche nach Methodennamen
                        name_elem = mem_item.find(['a', 'span'], class_=['el', 'memname'])
                        if name_elem:
                            method_name = name_elem.get_text().strip()
                            if method_name and not method_name.startswith('//') and not method_name.startswith('/*'):
                                if '::' in method_name:
                                    method_name = method_name.split('::')[-1]

                                # Extrahiere Parameter
                                params = []
                                param_spans = mem_item.find_all('span', class_='paramtype')
                                for param in param_spans:
                                    params.append(param.get_text().strip())

                                result['methods'].append({
                                    'name': method_name,
                                    'params': params,
                                    'signature': mem_item.get_text().strip()
                                })
                    except Exception:
                        continue

                # Attribute extrahieren
                for attr in soup.find_all(['td', 'div', 'span'], class_=['memItemLeft', 'memItemRight', 'memname']):
                    try:
                        attr_text = attr.get_text().strip()
                        if attr_text and not attr_text.startswith('//') and len(attr_text) > 2:
                            result['attributes'].append(attr_text)
                    except:
                        continue

            # Enum-Werte extrahieren
            if result['doc_type'] == 'enum':
                enum_values = soup.find_all(['a', 'span'], class_=['enumvalue', 'enumvalue-name'])
                for enum in enum_values:
                    value_text = enum.get_text().strip()
                    if value_text:
                        result['enums'].append(value_text)

            # Haupttext extrahieren (bereinigt)
            if content:
                # Kopie für Text-Extraktion
                content_copy = content

                # Entferne Navigations-Elemente
                for nav in content_copy.find_all(['div', 'ul', 'table'],
                                                 class_=['tabs', 'tablist', 'directory', 'memitem', 'memproto']):
                    nav.decompose()

                # Entferne leere Zeilen
                text = content_copy.get_text()
                text = re.sub(r'\n\s*\n', '\n\n', text)
                text = re.sub(r'[ \t]+', ' ', text)
                text = re.sub(r'\n{3,}', '\n\n', text)
                result['content'] = text.strip()

            # Wenn kein Inhalt gefunden wurde, versuche alternative Extraktion
            if not result['content'] and result['doc_type'] in ['class', 'struct']:
                # Versuche den gesamten Text zu extrahieren
                all_text = soup.get_text()
                all_text = re.sub(r'\n\s*\n', '\n\n', all_text)
                all_text = re.sub(r'[ \t]+', ' ', all_text)
                result['content'] = all_text.strip()

            # Wenn immer noch kein Inhalt, aber eine Member-Liste, erstelle Inhalt aus Methoden
            if not result['content'] and result['doc_type'] == 'member_list' and result['methods']:
                content_parts = [f"Members of {result['class_name']}:"]
                for method in result['methods'][:20]:  # Begrenzen für Übersichtlichkeit
                    content_parts.append(f"- {method['signature']}")
                result['content'] = '\n'.join(content_parts)

            return result

        except Exception as e:
            print(f"Fehler beim Verarbeiten von {html_path.name}: {e}")
            return None

    def create_enhanced_chunks(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Erstellt optimierte Text-Chunks mit Kontexterhaltung

        Args:
            data: Extrahierte Daten aus extract_content_from_html

        Returns:
            Liste von Chunk-Dictionaries
        """
        chunks = []

        if not data:
            return chunks

        # Prüfe ob Inhalt vorhanden ist
        content = data.get('content', '').strip()
        if not content and not data.get('methods') and not data.get('attributes') and not data.get('enums'):
            return chunks

        base_metadata = {
            'file_name': data.get('file_name', ''),
            'title': data.get('title', ''),
            'doc_type': data.get('doc_type', 'general'),
            'class_name': data.get('class_name', '')
        }

        # 1. Haupttext in sinnvolle Abschnitte teilen
        if content:
            # Teile bei Absätzen
            sections = re.split(r'\n\s*\n', content)

            for section in sections:
                section = section.strip()
                if len(section) < 50:  # Mindestlänge
                    continue

                chunks.append({
                    'text': section,
                    'metadata': {
                        **base_metadata,
                        'chunk_type': 'section',
                        'chunk_size': len(section)
                    }
                })

        # 2. Methoden als separate Chunks
        for method in data.get('methods', []):
            if method.get('name'):
                method_text = f"Method: {method['name']}\nSignature: {method.get('signature', '')}"
                chunks.append({
                    'text': method_text,
                    'metadata': {
                        **base_metadata,
                        'chunk_type': 'method',
                        'method_name': method['name']
                    }
                })

        # 3. Attribute als separate Chunks
        for attr in data.get('attributes', []):
            if len(attr) > 5:
                chunks.append({
                    'text': f"Attribute: {attr}",
                    'metadata': {
                        **base_metadata,
                        'chunk_type': 'attribute'
                    }
                })

        # 4. Enum-Werte als separate Chunks
        if data.get('enums'):
            enum_text = f"Enums in {data.get('class_name', 'Unknown')}:\n" + '\n'.join(data['enums'])
            chunks.append({
                'text': enum_text,
                'metadata': {
                    **base_metadata,
                    'chunk_type': 'enum'
                }
            })

        # 5. Vererbungsinformationen
        if data.get('inheritance'):
            inheritance_text = f"Inheritance: {' -> '.join(data['inheritance'])}"
            chunks.append({
                'text': inheritance_text,
                'metadata': {
                    **base_metadata,
                    'chunk_type': 'inheritance'
                }
            })

        return chunks

    def process_single_file(self, html_path: Path) -> Dict[str, Any]:
        """
        Verarbeitet eine einzelne HTML-Datei

        Args:
            html_path: Pfad zur HTML-Datei

        Returns:
            Dictionary mit Verarbeitungsergebnissen
        """
        try:
            # Extrahiere Inhalt
            data = self.extract_content_from_html(html_path)
            if not data:
                return {'success': False, 'error': 'Keine Daten extrahiert', 'no_content': True}

            # Erstelle Chunks
            chunks = self.create_enhanced_chunks(data)

            if not chunks:
                return {'success': False, 'error': 'Keine Chunks erstellt', 'no_content': True}

            # Dokument-ID generieren
            doc_id = hashlib.md5(f"{data['file_name']}_{data['title']}".encode()).hexdigest()[:16]

            # Alle SQLite-Inserts dieser Datei in EINER Transaktion bündeln
            # (statt Commit nach jedem Insert) -> weniger Lock-Konflikte (Fix 4)
            self.db.begin()
            try:
                self.db.execute('''
                    INSERT OR REPLACE INTO documents 
                    (id, file_name, title, doc_type, class_name, inheritance, content)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    doc_id,
                    data['file_name'],
                    data['title'],
                    data['doc_type'],
                    data['class_name'],
                    '|'.join(data['inheritance']),
                    data['content'][:10000] if data['content'] else ''
                ), commit=False)

                # Chunks in SQLite und ChromaDB speichern
                for i, chunk in enumerate(chunks):
                    chunk_id = f"{doc_id}_{i}"

                    # SQLite
                    self.db.execute('''
                        INSERT OR REPLACE INTO chunks 
                        (id, doc_id, chunk_text, chunk_type, start_pos, end_pos)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        chunk_id,
                        doc_id,
                        chunk['text'][:5000],
                        chunk['metadata'].get('chunk_type', 'unknown'),
                        i * 1000,
                        (i + 1) * 1000
                    ), commit=False)

                    # FTS5-Spiegel (Fix 2): DELETE+INSERT statt INSERT OR REPLACE,
                    # da chunk_id UNINDEXED ist und REPLACE dadurch nicht dedupliziert
                    self.db.execute(
                        'DELETE FROM chunks_fts WHERE chunk_id = ?',
                        (chunk_id,), commit=False)
                    self.db.execute('''
                        INSERT INTO chunks_fts (chunk_id, chunk_text)
                        VALUES (?, ?)
                    ''', (chunk_id, chunk['text'][:5000]), commit=False)

                    # ChromaDB: doc_type -> Collection routen (Fix 1)
                    collection_name = collection_for_doc_type(data['doc_type'])

                    try:
                        self.collections[collection_name].add(
                            documents=[chunk['text']],
                            metadatas=[{
                                **chunk['metadata'],
                                'doc_id': doc_id,
                                'chunk_id': chunk_id
                            }],
                            ids=[chunk_id]
                        )
                    except Exception as e:
                        # Fallback zu general Collection
                        try:
                            self.collections['general'].add(
                                documents=[chunk['text']],
                                metadatas=[{
                                    **chunk['metadata'],
                                    'doc_id': doc_id,
                                    'chunk_id': chunk_id
                                }],
                                ids=[chunk_id]
                            )
                        except Exception as e2:
                            print(f"ChromaDB Fehler für {chunk_id}: {e2}")

                self.db.commit()
            except Exception:
                self.db.rollback()
                raise

            return {
                'success': True,
                'doc_id': doc_id,
                'chunk_count': len(chunks),
                'doc_type': data['doc_type']
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def process_all_files(self, max_workers: int = 4):
        """
        Verarbeitet alle HTML-Dateien im Eingabeordner

        Args:
            max_workers: Anzahl der parallelen Worker
        """
        # Sammle alle HTML-Dateien
        html_files = list(self.input_dir.glob('*.html'))
        self.stats['total_files'] = len(html_files)

        print(f"Gefundene HTML-Dateien: {len(html_files)}")
        print(f"Starte Verarbeitung mit {max_workers} parallelen Workern...")
        print(f"Hinweis: SQLite-Fehler wurden durch thread-sichere Verbindungen behoben")
        print("=" * 60)

        # Parallele Verarbeitung mit Fortschrittsbalken
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.process_single_file, f): f for f in html_files}

            with tqdm(total=len(futures), desc="Verarbeite Dateien", unit="Datei") as pbar:
                for future in as_completed(futures):
                    file_path = futures[future]
                    try:
                        result = future.result(timeout=10)  # 10 Sekunden Timeout
                        if result and result.get('success'):
                            self.stats['processed_files'] += 1
                            self.stats['total_chunks'] += result.get('chunk_count', 0)
                        else:
                            if result and result.get('no_content'):
                                self.stats['no_content'] += 1
                            else:
                                self.stats['errors'] += 1
                                error_msg = result.get('error', 'Unbekannter Fehler') if result else 'Kein Ergebnis'
                                # Nur bei echten Fehlern anzeigen
                                if not result or not result.get('no_content'):
                                    print(f"\nFehler in {file_path.name}: {error_msg}")
                    except Exception as e:
                        self.stats['errors'] += 1
                        print(f"\nZeitüberschreitung oder Fehler bei {file_path.name}: {e}")

                    pbar.update(1)
                    pbar.set_postfix({
                        'Erfolg': self.stats['processed_files'],
                        'Leer': self.stats['no_content'],
                        'Fehler': self.stats['errors'],
                        'Chunks': self.stats['total_chunks']
                    })

        # Abschließende Statistiken
        self.save_statistics()
        self.optimize_database()

        print("\n" + "=" * 60)
        print("VERARBEITUNG ABGESCHLOSSEN!")
        print("=" * 60)
        print(f"Gesamt: {self.stats['total_files']} Dateien")
        print(f"Erfolgreich verarbeitet: {self.stats['processed_files']}")
        print(f"Leere Dateien (kein Inhalt): {self.stats['no_content']}")
        print(f"Fehler: {self.stats['errors']}")
        print(f"Erstellte Chunks: {self.stats['total_chunks']}")
        print(f"Datenbank-Pfad: {self.output_dir}")

        # Verifikation: Chunks pro Collection (Fix 1)
        print("\nChunks pro ChromaDB-Collection:")
        for name in ALL_COLLECTIONS:
            if name in self.collections:
                try:
                    print(f"  {name}: {self.collections[name].count()}")
                except Exception as e:
                    print(f"  {name}: Fehler bei count() -> {e}")

        # Verifikation: FTS-Tabelle (Fix 2)
        try:
            fts_count = self.db.fetch_one("SELECT COUNT(*) FROM chunks_fts")[0]
            print(f"FTS5-Einträge (chunks_fts): {fts_count}")
        except Exception as e:
            print(f"FTS5-Check fehlgeschlagen: {e}")

        print("=" * 60)

    def save_statistics(self):
        """Speichert Verarbeitungsstatistiken"""
        stats_file = self.output_dir / "processing_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2)

    def optimize_database(self):
        """Optimiert die Datenbanken für bessere Suchleistung"""
        try:
            # SQLite optimieren
            self.db.execute('VACUUM')
            self.db.execute('ANALYZE')

            # Index erstellen für häufige Abfragen
            self.db.execute('CREATE INDEX IF NOT EXISTS idx_doc_type ON documents(doc_type)')
            self.db.execute('CREATE INDEX IF NOT EXISTS idx_class_name ON documents(class_name)')
            self.db.execute('CREATE INDEX IF NOT EXISTS idx_chunk_type ON chunks(chunk_type)')

            print("Datenbank optimiert")
        except Exception as e:
            print(f"Fehler bei Datenbankoptimierung: {e}")

    def close(self):
        """Schließt Datenbankverbindungen"""
        if hasattr(self, 'db'):
            self.db.close()


class ESPRITRAGChatbot:
    """RAG-Chatbot für die ESPRIT API Dokumentation"""

    def __init__(self, index_dir: str = "./index_data"):
        self.index_dir = Path(index_dir)
        self.chroma_path = self.index_dir / "chroma_db"
        self.db_path = self.index_dir / "metadata.db"

        # Lade Embedding-Modell (MPS auf Apple-Silicon, sonst CPU)
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device=get_device())

        # Verbinde zur ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=str(self.chroma_path))
        self.collections = {}

        # Lade Collections
        for name in ['classes', 'structs', 'members', 'enums', 'general']:
            try:
                self.collections[name] = self.chroma_client.get_collection(name)
            except:
                pass

        # SQLite für Metadaten (thread-safe)
        self.db = ThreadSafeSQLite(str(self.db_path))

    def query(self, query_text: str, n_results: int = 5, collection_filter: str = None) -> Dict[str, Any]:
        """
        Führt eine RAG-Abfrage durch

        Args:
            query_text: Die Anfrage
            n_results: Anzahl der Ergebnisse
            collection_filter: Filter für spezifische Collection

        Returns:
            Dictionary mit Ergebnissen
        """
        # Embedding für die Anfrage (normalisiert, konsistent zum Indexer)
        query_embedding = self.model.encode(query_text, normalize_embeddings=True).tolist()

        # Sammle Ergebnisse aus allen Collections
        all_results = []
        collections_to_search = [collection_filter] if collection_filter else self.collections.keys()

        for coll_name in collections_to_search:
            if coll_name not in self.collections:
                continue

            try:
                results = self.collections[coll_name].query(
                    query_embeddings=[query_embedding],
                    n_results=n_results
                )

                if results and results['documents']:
                    for i, doc in enumerate(results['documents'][0]):
                        all_results.append({
                            'text': doc,
                            'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                            'collection': coll_name,
                            'distance': results['distances'][0][i] if results['distances'] else None
                        })
            except Exception as e:
                continue

        # Sortiere nach Distanz
        all_results.sort(key=lambda x: x.get('distance', float('inf')))

        # Hole zusätzliche Metadaten aus SQLite
        for result in all_results[:n_results]:
            if 'doc_id' in result['metadata']:
                doc_info = self.db.fetch_one('''
                    SELECT title, doc_type, class_name, inheritance 
                    FROM documents WHERE id = ?
                ''', (result['metadata']['doc_id'],))
                if doc_info:
                    result['doc_info'] = {
                        'title': doc_info[0],
                        'type': doc_info[1],
                        'class_name': doc_info[2],
                        'inheritance': doc_info[3]
                    }

        # Kontext für LLM erstellen
        context = "\n\n---\n\n".join([
            f"Quelle: {r['metadata'].get('title', 'Unbekannt')}\n"
            f"Dokumenttyp: {r['metadata'].get('doc_type', 'Unbekannt')}\n"
            f"Klasse: {r['metadata'].get('class_name', 'Unbekannt')}\n\n"
            f"{r['text']}"
            for r in all_results[:n_results]
        ])

        return {
            'context': context,
            'results': all_results[:n_results],
            'total_found': len(all_results)
        }

    def close(self):
        """Schließt Datenbankverbindungen"""
        if hasattr(self, 'db'):
            self.db.close()


def main():
    """Hauptfunktion für die Ausführung des Programms"""

    # Konfiguration über CLI (Fix 5): kein hartcodierter Pfad mehr
    ap = argparse.ArgumentParser(description="ESPRIT EDGE API Documentation Indexer")
    ap.add_argument("--input-dir", default="./API-Help-html",
                    help="Ordner mit den Doxygen-HTML-Dateien (Standard: ./API-Help-html)")
    ap.add_argument("--output-dir", default="./index_data",
                    help="Ausgabeordner für ChromaDB + metadata.db (Standard: ./index_data)")
    ap.add_argument("--workers", type=int, default=4,
                    help="Anzahl paralleler Worker (Standard: 4)")
    args = ap.parse_args()

    INPUT_DIR = os.path.expanduser(args.input_dir)
    OUTPUT_DIR = os.path.expanduser(args.output_dir)

    print("=" * 60)
    print("ESPRIT EDGE API DOCUMENTATION INDEXER")
    print("=" * 60)
    print(f"Eingabeordner: {INPUT_DIR}")
    print(f"Ausgabeordner: {OUTPUT_DIR}")
    print("=" * 60)

    # Prüfe Eingabeordner
    if not os.path.exists(INPUT_DIR):
        print(f"FEHLER: Eingabeordner '{INPUT_DIR}' nicht gefunden!")
        return

    # Initialisiere Indexer
    indexer = ESPRITDocumentationIndexer(INPUT_DIR, OUTPUT_DIR)
    print(f"Embedding-Device: {indexer.device}")

    # Verarbeite alle Dateien
    try:
        indexer.process_all_files(max_workers=args.workers)
    finally:
        indexer.close()

    # Test-Abfrage durchführen
    print("\nFühre Test-Abfragen durch...")
    print("=" * 60)

    chatbot = ESPRITRAGChatbot(OUTPUT_DIR)

    test_queries = [
        "Was ist espDedSubTechnoType?",
        "Erkläre die FeatureHoleItemStraightData Struktur",
        "Wie verwende ich TechMillSpiral?",
        "Was sind die Methoden der Klasse Document?",
        "Zeige mir alle Enums"
    ]

    for query in test_queries:
        print(f"\n{'=' * 40}")
        print(f"Frage: {query}")
        print('=' * 40)
        try:
            result = chatbot.query(query, n_results=3)
            print(f"Gefunden: {result['total_found']} Einträge")
            if result['results']:
                print(f"\nErstes Ergebnis:")
                first = result['results'][0]
                print(f"Typ: {first['metadata'].get('doc_type', 'Unbekannt')}")
                print(f"Klasse: {first['metadata'].get('class_name', 'Unbekannt')}")
                print(f"Text: {first['text'][:200]}...")
        except Exception as e:
            print(f"Fehler bei Abfrage: {e}")

    chatbot.close()

    print("\n" + "=" * 60)
    print("Programm erfolgreich abgeschlossen!")
    print("=" * 60)
    print(f"Die Index-Datenbank befindet sich in: {OUTPUT_DIR}")
    print("\nVerwendung im Chatbot:")
    print("  from indexer import ESPRITRAGChatbot")
    print("  bot = ESPRITRAGChatbot('./index_data')")
    print("  result = bot.query('Ihre Frage zur ESPRIT API')")
    print("=" * 60)


if __name__ == "__main__":
    main()