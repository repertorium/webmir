import ast
import asyncio
import copy
import jwt
import os
import pickle
import sqlite3
import re
import time
import threading
import uuid
from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta, timezone
from typing import List, Optional
from contextlib import asynccontextmanager
from functools import cmp_to_key
from urllib.parse import urlparse, parse_qs
from tqdm import tqdm

from omr import OMRprocessor
from similarityhybridindex import compute_similarity_hybrid, get_chant_fulltext
from sporkapi import SporkWorkspaceSession
from utils import Trie


# ----------------------------
# Config
# ----------------------------
#STOREMANID_PATH = './backend/storage/manID/'
#DB_QUEUE_PATH = "./backend/storage/queue.db"
#CANTUS_DB_FILE = "./backend/storage/chant_info.db"
#PROGRESS_FILE = "./backend/storage/progress.txt"

STOREMANID_PATH = '/app/storage/manID/'   # production
DB_QUEUE_PATH = "/app/storage/queue.db"   # production
DB_USERS_PATH = "/app/storage/users.db"   # production
CANTUS_DB_FILE = "/app/storage/chant_info.db"   # production
PROGRESS_FILE = "/app/storage/progress.txt" # production

JWT_SECRET = "<0875436_SECRET_KEY>"
JWT_ALGO = "HS256"
TOKEN_EXPIRE_MINUTES = 60


# ----------------------------
# DB helpers (sqlite simple queue)
# ----------------------------
def load_users_from_db(db_path: str) -> dict:
    users = {}

    con = sqlite3.connect(db_path)
    cur = con.cursor()

    cur.execute("SELECT username, password, is_admin FROM users")
    rows = cur.fetchall()

    for username, password, is_admin in rows:
        users[username] = {
            "password": password,
            "is_admin": bool(is_admin)
        }

    con.close()
    return users


def init_db(reset=False):
    con = sqlite3.connect(DB_QUEUE_PATH)
    cur = con.cursor()
    if reset:
        cur.execute("DROP TABLE IF EXISTS queue")
    cur.execute(
        """CREATE TABLE IF NOT EXISTS queue (
            id TEXT PRIMARY KEY,
            manuscript_id TEXT,
            url TEXT,
            status TEXT,
            enqueued_at TEXT,
            started_at TEXT,
            finished_at TEXT,
            result_pickle TEXT
        )"""
    )
    con.commit()
    con.close()

def enqueue_task(manuscript_id: str, url: str):
    con = sqlite3.connect(DB_QUEUE_PATH)
    cur = con.cursor()
    task_id = str(uuid.uuid4())
    cur.execute(
        "INSERT INTO queue (id, manuscript_id, url, status, enqueued_at) VALUES (?, ?, ?, ?, ?)",
        (task_id, manuscript_id, url, "queued", datetime.now(timezone.utc).isoformat()),
    )
    con.commit()
    con.close()
    return task_id

def get_queue():
    con = sqlite3.connect(DB_QUEUE_PATH)
    cur = con.cursor()
    cur.execute("SELECT id, manuscript_id, url, status, enqueued_at, started_at, finished_at, result_pickle FROM queue ORDER BY enqueued_at")
    rows = cur.fetchall()
    con.close()
    keys = ["id","manuscript_id","url","status","enqueued_at","started_at","finished_at","result_pickle"]
    return [dict(zip(keys,row)) for row in rows]

def get_next_queued():
    con = sqlite3.connect(DB_QUEUE_PATH)
    cur = con.cursor()
    cur.execute("SELECT id, manuscript_id, url FROM queue WHERE status='queued' ORDER BY enqueued_at LIMIT 1")
    row = cur.fetchone()
    con.close()
    return row

def set_task_status(task_id, status, started_at=None, finished_at=None, result_pickle=None):
    con = sqlite3.connect(DB_QUEUE_PATH)
    cur = con.cursor()
    updates = []
    params = []
    updates.append("status=?"); params.append(status)
    if started_at is not None:
        updates.append("started_at=?"); params.append(started_at)
    if finished_at is not None:
        updates.append("finished_at=?"); params.append(finished_at)
    if result_pickle is not None:
        updates.append("result_pickle=?"); params.append(result_pickle)
    params.append(task_id)
    cur.execute(f"UPDATE queue SET {', '.join(updates)} WHERE id=?", params)
    con.commit()
    con.close()


USERS = load_users_from_db(DB_USERS_PATH)


# ----------------------------
# Index_manuscript functions
# ----------------------------
def get_bounding_box(bbox_txt):
    bbox = ast.literal_eval(bbox_txt)
    xmin = min(point[0] for point in bbox)
    xmax = max(point[0] for point in bbox)
    ymin = min(point[1] for point in bbox)
    ymax = max(point[1] for point in bbox)
    return [xmin, ymin, xmax, ymax]

def horizontal_overlap(box1, box2):
    return not (box1[2] < box2[0] or box2[2] < box1[0])

def vertical_overlap(box1, box2):
    # return not (box1[3] < box2[1] or box2[3] < box1[1])
    y_top = max(box1[1], box2[1])
    y_bottom = min(box1[3], box2[3])
    overlap = max(0, y_bottom - y_top)
    height1 = box1[3] - box1[1]
    height2 = box2[3] - box2[1]
    return (overlap >= 0.5 * height1) and (overlap >= 0.5 * height2)

def group_boxes_into_columns(boxes):
    columns = []
    for box in boxes:
        added = False
        for column in columns:
            if any(horizontal_overlap(box, other) for other in column):
                column.append(box)
                added = True
                break
        if not added:
            columns.append([box])
    return columns

def sort_column(column):
    def compare_boxes(b1, b2):
        if vertical_overlap(b1, b2):
            return -1 if b1[0] < b2[0] else 1
        else:
            return -1 if b1[1] < b2[1] else 1

    return sorted(column, key=cmp_to_key(compare_boxes))

def sort_annotation_groups(data):
    bboxes = [get_bounding_box(ag["boundingBox"]) for ag in data['annotationGroups']]
    columns = group_boxes_into_columns(bboxes)
    columns.sort(key=lambda col: min(b[0] for b in col))  # ordenar columnas por xmin
    sorted_boxes = []
    for column in columns:
        sorted_boxes.extend(sort_column(column))
    sorted_idx = [bboxes.index(x) for x in sorted_boxes]
    sorted_annotation_groups = [data['annotationGroups'][i] for i in sorted_idx]
    data['annotationGroups'] = sorted_annotation_groups
    return data


# def index_manuscript(url):
#     # get matuscriptID
#     path_parts = urlparse(url).path.strip("/").split("/")
#     if len(path_parts) >= 2 and path_parts[0] == "manuscript":
#         manuscriptID = path_parts[1]
#     else:
#         return
#     print(f"\nIndexing manuscript {manuscriptID}...")
    
#     # get page list
#     pagelist = session.get_manuscript_pagelist(manuscriptID)
#     if len(pagelist) == 0:
#         print(f"No pages in manuscriptID {manuscriptID}")
#         return
#     print(f"Manuscript {manuscriptID} has {len(pagelist)} pages...")

#     # pagelist = pagelist[129:]
#     # print(f"Procesando solo {len(pagelist)} páginas...")

#     # download pages info/annotations
#     data = len(pagelist) * [{}]
#     dataorig = len(pagelist) * [{}]

#     for i in tqdm(range(len(pagelist)), desc="Downloading manuscript pages"):
#         imageID = str(pagelist[i]['id'])
#         # get annotations from web page
#         datapage = session.get_annotations_from_page(manuscriptID, imageID)
#         # discard annotationGroups with only CHANTS
#         datapage['annotationGroups'] = [
#             group for group in datapage['annotationGroups']
#             if not all( annotation.get('type')=='CHANT' for annotation in group.get('annotations', []) )
#         ]
#         dataorig[i] = copy.deepcopy(datapage)

#         # check if an annotationgroup has an empty notation. If so, remove it
#         datapage['annotationGroups'] = [
#             group for group in datapage['annotationGroups']
#             if group['notation'] is not None
#         ]
#         # sort annotation groups
#         datapage = sort_annotation_groups(datapage)
#         # filter annotationGroups with notes
#         datapage['annotationGroups'] = [
#             group for group in datapage['annotationGroups']
#             if any(annotation.get('type')=='NOTATION' for annotation in group.get('annotations', []))
#         ]
#         data[i] = datapage

#     # OMR processing
#     for i in tqdm(range(len(pagelist)), desc="OMR processing"):
#         if len(data[i]['annotationGroups']) == 0:
#             data[i]['annotationGroups'] = omr_processor.process_image(
#                 data[i]['iiifImageUrl'], 
#                 data[i]['width'], 
#                 data[i]['height']
#             )

#     # Index pages
#     datamir = len(pagelist) * [{}]
#     for i in range(len(pagelist)):
#         print(f"\nIndexing page {i} ({pagelist[i]['id']})...")
#         neighbors = [
#             data[i-5] if i-5 >= 0 else {},
#             data[i-4] if i-4 >= 0 else {},
#             data[i-3] if i-3 >= 0 else {},
#             data[i-2] if i-2 >= 0 else {},
#             data[i-1] if i-1 >= 0 else {},
#             data[i],
#             data[i+1] if i+1 < len(data) else {},
#             data[i+2] if i+2 < len(data) else {},
#             data[i+3] if i+3 < len(data) else {},
#             data[i+4] if i+4 < len(data) else {},
#             data[i+5] if i+5 < len(data) else {}
#         ]
#         dataout = compute_similarity_hybrid(
#             neighbors,
#             dataorig[i],
#             session, dictionary, cantusidall, trie
#         )
#         datamir[i] = dataout

#         progress_value = int(round((i+1)*100 / len(pagelist)))
#         with open(PROGRESS_FILE, "w") as f:
#             f.write(str(progress_value) + "\n")

#     # Save results as a piclke file
#     with open(STOREMANID_PATH + f'{manuscriptID}.pkl', 'wb') as f:
#         pickle.dump({
#             'datamir': datamir
#         }, f)

#     return


def index_manuscript(url):
    # get matuscriptID
    path_parts = urlparse(url).path.strip("/").split("/")
    if len(path_parts) >= 2 and path_parts[0] == "manuscript":
        manuscriptID = path_parts[1]
    else:
        return
    print(f"\nIndexing manuscript {manuscriptID}...")
    
    # get page list
    pagelist = session.get_manuscript_pagelist(manuscriptID)
    if len(pagelist) == 0:
        print(f"No pages in manuscriptID {manuscriptID}")
        return
    print(f"Manuscript {manuscriptID} has {len(pagelist)} pages...")

    # pagelist = pagelist[359:380]
    # print(f"Procesando solo {len(pagelist)} páginas...")

    # download pages info/annotations
    data = len(pagelist) * [{}]
    dataorig = len(pagelist) * [{}]

    for i in tqdm(range(len(pagelist)), desc="Downloading manuscript pages"):
        imageID = str(pagelist[i]['id'])
        # get annotations from web page
        datapage = session.get_annotations_from_page(manuscriptID, imageID)
        # discard annotationGroups with only CHANTS
        datapage['annotationGroups'] = [
            group for group in datapage['annotationGroups']
            if not all( annotation.get('type')=='CHANT' for annotation in group.get('annotations', []) )
        ]
        dataorig[i] = copy.deepcopy(datapage)

        # check if an annotationgroup has an empty notation. If so, remove it
        datapage['annotationGroups'] = [
            group for group in datapage['annotationGroups']
            if group['notation'] is not None
        ]
        # sort annotation groups
        datapage = sort_annotation_groups(datapage)
        # filter annotationGroups with notes
        datapage['annotationGroups'] = [
            group for group in datapage['annotationGroups']
            if any(annotation.get('type')=='NOTATION' for annotation in group.get('annotations', []))
        ]
        data[i] = datapage

    # OMR processing
    for i in tqdm(range(len(pagelist)), desc="OMR processing"):
        if len(data[i]['annotationGroups']) == 0:
            data[i]['annotationGroups'] = omr_processor.process_image(
                data[i]['iiifImageUrl'], 
                data[i]['width'], 
                data[i]['height']
            )

    # Index pages
    BLOCK_SIZE, CONTEXT = 30, 5

    datamir = [{} for _ in range(len(pagelist))]
    num_pages = len(pagelist)

    for block_start in range(0, num_pages, BLOCK_SIZE):
        block_end = min(block_start + BLOCK_SIZE, num_pages)
        print(
            f"\nIndexing pages {block_start} - {block_end - 1} "
            f"({pagelist[block_start]['id']} - {pagelist[block_end - 1]['id']})..."
        )

        # --- Build neighbors with EXPLICIT padding ---
        neighbors = []
        # Left context (5 pages)
        for i in range(block_start - CONTEXT, block_start):
            if 0 <= i < num_pages:
                neighbors.append(data[i])
            else:
                neighbors.append({})
        # Block pages
        neighbors.extend(data[block_start:block_end])
        # Right context (5 pages)
        for i in range(block_end, block_end + CONTEXT):
            if 0 <= i < num_pages:
                neighbors.append(data[i])
            else:
                neighbors.append({})

        # Pages to be indexed
        dataorig_block = dataorig[block_start:block_end]
        # Compute MIR
        dataout_block = compute_similarity_hybrid(
            neighbors,
            dataorig_block,
            session, dictionary, cantusidall, trie
        )
        # Store results
        for i, page_result in enumerate(dataout_block):
            datamir[block_start + i] = page_result

        progress_value = int(round((block_end+1)*100 / num_pages))
        with open(PROGRESS_FILE, "w") as f:
            f.write(str(progress_value) + "\n")

    # Save results as a piclke file
    with open(STOREMANID_PATH + f'{manuscriptID}.pkl', 'wb') as f:
        pickle.dump({
            'datamir': datamir
        }, f)

    return



# ----------------------------
# Worker thread: procesa la cola de tareas secuencialmente
# ----------------------------
stop_worker = False

def worker_loop():
    print("[worker] Iniciando worker")
    while not stop_worker:
        # Si ya hay un task en status=running, duerme (aseguramos 1 en ejecución)
        con = sqlite3.connect(DB_QUEUE_PATH)
        cur = con.cursor()
        cur.execute("SELECT id FROM queue WHERE status='running' LIMIT 1")
        running = cur.fetchone()
        con.close()
        if running:
            time.sleep(2)
            continue

        # Obtener siguiente queued
        nxt = get_next_queued()
        if not nxt:
            time.sleep(2)
            continue

        task_id, manuscript_id, url = nxt
        set_task_status(task_id, "running", started_at=datetime.now(timezone.utc).isoformat())
        try:
            with open(PROGRESS_FILE, "w") as f:
              f.write(str(0) + "\n")
            # pickle_path = index_manuscript(url, manuscript_id)
            index_manuscript(url)
            pickle_path = ""
            set_task_status(task_id, "done", finished_at=datetime.now(timezone.utc).isoformat(), result_pickle=pickle_path)
        except Exception as e:
            print("[worker] Error procesando task", e)
            set_task_status(task_id, "error", finished_at=datetime.now(timezone.utc).isoformat())


# ----------------------------
# FastAPI app and models
# ----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    PASSWORD = os.getenv("PLATFORM_PASSWORD")
    USER = os.getenv("PLATFORM_USER")
    loop = asyncio.get_event_loop()
    result, error_msg = await loop.run_in_executor(
        None,
        session.login_platform,
        USER,
        PASSWORD
    )
    if not result:
        raise RuntimeError(f"Login failed: {error_msg}")
    yield

app = FastAPI(lifespan=lifespan)

# Permitir peticiones desde el frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # en producción, reemplazar el dominio
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelo de entrada
class SearchRequest(BaseModel):
    query: str

# Modelo de salida
class SearchResult(BaseModel):
    cantusid: str
    lyrics: str
    link: str
    prob: float
    page: int

class EnqueueRequest(BaseModel):
    url: str

def create_token(username: str):
    payload = {
        "sub": username,
        "exp": datetime.now(timezone.utc) + timedelta(minutes=TOKEN_EXPIRE_MINUTES)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)

def verify_token(token: str) -> Optional[str]:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
        return payload.get("sub")
    except Exception:
        return None


# ----------------------------
# Endpoints
# ----------------------------
@app.post("/login")
def login(body: dict):
    username = body.get("username")
    password = body.get("password")
    if not username or not password:
        raise HTTPException(status_code=400, detail="username & password required")
    u = USERS.get(username)
    if not u or u["password"] != password:
        raise HTTPException(status_code=401, detail="invalid credentials")
    token = create_token(username)
    return {"access_token": token, "token_type": "bearer"}


@app.post("/enqueue")
def enqueue(body: EnqueueRequest, authorization: Optional[str] = Header(None)):
    """
    Usuario logueado encola una URL para procesar.
    Header: Authorization: Bearer <token>
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="missing authorization")
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="invalid auth header")
    token = authorization.split(" ",1)[1]
    user = verify_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="invalid token")
    
    # Crear manuscript id a partir de url (normalmente usar un hash)
    manuscript_id = str(uuid.uuid4())[:8]
    task_id = enqueue_task(manuscript_id, body.url)
    return {"task_id": task_id, "manuscript_id": manuscript_id}


@app.get("/manuscripts")
def list_manuscripts():
    """
    Devuelve:
    {
      processed: [manuscript_id, ...],
      processing: { running: {task...} | null, queue: [{task...}, ...] }
    }
    """
    # processed: detecta pickles en directory
    processed = []
    for fname in os.listdir(STOREMANID_PATH):
        if fname.endswith(".pkl"):
            # read data
            with open(os.path.join(STOREMANID_PATH, fname), "rb") as f:
                datamir = pickle.load(f)
            # get manuscript title
            title = None
            for item in datamir.get('datamir', []):
                if item:
                    title = item.get('manuscript').get('title')
                    break
            processed.append(os.path.splitext(fname)[0] + (f" ({title})" if title else ""))

    queue = get_queue()
    running = next((q for q in queue if q["status"] == "running"), None)
    queued = [q for q in queue if q["status"] == "queued"]
    try:
        with open(PROGRESS_FILE, "r") as f:
            progress = f.read().strip()
            if progress.isdigit() and running is not None:
                running['manuscript_id'] = f"{running['url']}  ({progress} %)"
    except FileNotFoundError:
        pass
    for i in range(len(queued)):
        queued[i]['manuscript_id'] = f"{queued[i]['url']}"
    return {"processed": processed, "processing": {"running": running, "queue": queued}}


@app.post("/search", response_model=List[SearchResult])
def search(url: SearchRequest):
    """
    Endpoint de devolución de resultados de cantos.
    """
    manuscript = url.query
    if " (" in manuscript:
        manuscript = manuscript.split(" (")[0]
    with open(os.path.join(STOREMANID_PATH, manuscript + ".pkl"), "rb") as f:
        datamir = pickle.load(f)

    results = []
    for i, page in enumerate(datamir.get('datamir', [])):
        for ag in page.get('annotationGroups', []):
            annotations = ag.get('annotations', [])
            if annotations and annotations[0].get('type') == 'CHANT':
                notes = ast.literal_eval(ag.get('notes', '{}'))
                prob = round(notes[0], 3)
                idlast = notes[1]
                # unknown content before
                if idlast != None:
                    results.append({
                        "cantusid": 'unknown',
                        "lyrics": 'UNKNOWN CONTENT',
                        "page": i,
                        "prob": 0.0,
                        "link": f"https://repertorium.spork-infra.uk/manuscript/{manuscript}/{idlast}"
                    })
                cantusid = ag.get('chantRecordVersion', {}).get('chantRecord', {}).get('chant', {}).get('cantusId', 'unknown')
                lyrics = ag.get('notation', 'No lyrics available')
                if re.match(r'^\$\d+$', lyrics):
                    lyrics = dictionary[cantusidall.index(cantusid)]
                else:
                    lyrics = lyrics.split("\n", 1)[0][2:]
                link = f"https://repertorium.spork-infra.uk/manuscript/{manuscript}/{page.get('id', '')}"

                results.append({
                    "cantusid": cantusid,
                    "lyrics": lyrics,
                    "page": i,
                    "prob": prob,
                    "link": link
                })

    return results


# ----------------------------
# Startup:
# ----------------------------
# session handler
session = SporkWorkspaceSession()

# omr processor
omr_processor = OMRprocessor()

# Lyrics dictionary
print("Loading chant dictionary...")
dictionary, cantusidall = get_chant_fulltext()
trie = Trie()
for frase, id_ in zip(dictionary, cantusidall):
    trie.insert(frase, id_)


init_db(reset=True)
worker_thread = threading.Thread(target=worker_loop, daemon=True)
worker_thread.start()
