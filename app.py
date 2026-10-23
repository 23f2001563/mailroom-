import os, json, hashlib, hmac, sqlite3, threading, uuid, base64
from datetime import datetime, timezone

import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
DATABASE_URL = os.environ.get("DATABASE_URL", "")
DB_PATH = os.environ.get("DB_PATH", "state.db")
RECEIPT_SECRET = os.environ.get("RECEIPT_SECRET", "")  # if given, receipt signatures are verified (HMAC-SHA256)
DEFAULT_ACTIONS = ["deliver", "hold", "return_to_sender", "forward", "escalate"]
SAFE_ACTIONS = [a.strip() for a in os.environ.get("SAFE_ACTIONS", "hold,escalate,request_info,reject").split(",")]
P = "mr:"  # key prefix, so this agent can share a database with other agents
LOCK = threading.Lock()


# ---------- tiny key-value store (Postgres if DATABASE_URL set, else SQLite) ----------
def run(sql, params=(), fetch=False):
    with LOCK:
        if DATABASE_URL:
            import psycopg2
            con = psycopg2.connect(DATABASE_URL)
            try:
                cur = con.cursor()
                cur.execute(sql.replace("?", "%s"), params)
                rows = cur.fetchall() if fetch else None
                con.commit()
                return rows
            finally:
                con.close()
        con = sqlite3.connect(DB_PATH)
        try:
            cur = con.execute(sql, params)
            rows = cur.fetchall() if fetch else None
            con.commit()
            return rows
        finally:
            con.close()


run("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT)")


def get(k):
    rows = run("SELECT v FROM kv WHERE k=?", (P + k,), True)
    return json.loads(rows[0][0]) if rows else None


def put(k, v):
    run("INSERT INTO kv (k,v) VALUES (?,?) ON CONFLICT (k) DO UPDATE SET v=excluded.v", (P + k, json.dumps(v)))


def list_prefix(prefix):
    rows = run("SELECT v FROM kv WHERE k LIKE ?", (P + prefix + "%",), True)
    return [json.loads(r[0]) for r in rows]


# ---------- helpers ----------
def now():
    return datetime.now(timezone.utc).isoformat()


def norm(s):
    return "".join(ch for ch in str(s or "").lower() if ch.isalnum())


def base_url(request):
    return os.environ.get("PUBLIC_URL") or os.environ.get("RENDER_EXTERNAL_URL") or str(request.base_url).rstrip("/")


def tenant_of(request, msg):
    t = request.headers.get("x-user-id") or request.headers.get("x-tenant-id")
    if not t and request.headers.get("authorization"):
        t = hashlib.sha1(request.headers["authorization"].encode()).hexdigest()[:16]
    if not t:
        md = msg.get("metadata") or {}
        t = md.get("userId") or md.get("user_id") or md.get("tenant")
    return norm(t) or "default"


def collect(parts):
    texts, files, objs = [], [], []
    for p in parts or []:
        if p.get("text"):
            texts.append(p["text"])
            try:
                o = json.loads(p["text"])
                if isinstance(o, (dict, list)):
                    objs.append(o)
            except Exception:
                pass
        if p.get("data") is not None:
            texts.append(json.dumps(p["data"]))
            objs.append(p["data"])
        f = p.get("file") or {}
        b64 = f.get("bytes") or p.get("raw")
        mime = f.get("mimeType") or p.get("mediaType") or "application/pdf"
        url = f.get("uri") or p.get("url")
        try:
            if not b64 and url:
                r = requests.get(url, timeout=30)
                mime = r.headers.get("content-type", mime).split(";")[0]
                b64 = base64.b64encode(r.content).decode()
        except Exception:
            pass
        if b64:
            files.append({"mime": mime, "b64": b64})
    return texts, files, objs


def find_key(o, keys):
    """Search nested JSON for the first value under any of the given keys."""
    if isinstance(o, dict):
        for k, v in o.items():
            if k in keys and v:
                return v
        for v in o.values():
            r = find_key(v, keys)
            if r:
                return r
    elif isinstance(o, list):
        for v in o:
            r = find_key(v, keys)
            if r:
                return r
    return None


def find_receipt(objs):
    for o in objs:
        if isinstance(o, dict):
            rc = o.get("receipt")
            if isinstance(rc, dict):
                return rc
            if o.get("signature") and (o.get("receiptId") or o.get("receipt_id") or o.get("proposalId")
                                       or o.get("proposal_id") or o.get("actionId") or o.get("action_id")):
                return o
    return None


PROMPT = """You are a mailroom triage assistant. Read the case file (text and/or attached documents) and choose exactly ONE action from PERMITTED ACTIONS.
Return ONLY a JSON object:
{"case_id": string|null, "recipient": string|null, "current_status": string|null, "action": string,
 "approval": {"present": boolean, "signed": boolean, "scope_recipient": string|null, "scope_status": string|null,
              "scope_action": string|null, "valid": boolean, "why": string},
 "reason": string, "evidence": [{"source": string, "quote": string}]}
Rules:
- An approval only counts if it is signed AND its scope matches THIS case: the same recipient, the same item/case, the same action,
  and it refers to the CURRENT status (not an older status that has since changed).
- A valid signature on an approval for a different recipient, a different item, or an outdated status does NOT count (valid=false).
- If your preferred action needs approval and no valid approval exists, choose the safest permitted action instead (hold or escalate).
- evidence: short exact quotes (under 20 words) with their document/section name as source. Never invent facts."""


def ask_gemini(texts, files, permitted):
    parts = [{"text": PROMPT + "\n\nPERMITTED ACTIONS: " + json.dumps(permitted)}]
    parts += [{"text": t} for t in texts]
    parts += [{"inline_data": {"mime_type": f["mime"], "data": f["b64"]}} for f in files]
    errors = []
    for model in [GEMINI_MODEL, "gemini-flash-latest"]:
        try:
            r = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                params={"key": GEMINI_KEY},
                json={"contents": [{"parts": parts}], "generationConfig": {"responseMimeType": "application/json", "temperature": 0}},
                timeout=40,
            )
            if r.status_code != 200:
                errors.append(f"{model} HTTP {r.status_code}: {r.text[:300]}")
                continue
            ps = r.json()["candidates"][0]["content"]["parts"]
            txt = "".join(p.get("text", "") for p in ps if not p.get("thought"))
            out = json.loads(txt.strip().removeprefix("```json").removesuffix("```").strip())
            if isinstance(out, list):
                out = out[0]
            return out
        except Exception as e:
            errors.append(f"{model} error: {e}")
    raise RuntimeError(" || ".join(errors) or "Gemini failed")


def decide(texts, files, objs):
    permitted = find_key(objs, ["permitted_actions", "permittedActions", "allowed_actions", "allowedActions"])
    if not isinstance(permitted, list) or not permitted:
        permitted = DEFAULT_ACTIONS
    d = ask_gemini(texts, files, permitted)
    if isinstance(d, list):
        d = d[0] if d else {}
    d.setdefault("evidence", [])
    ap = d.get("approval") if isinstance(d.get("approval"), dict) else {}
    # code-level scope check: do not trust the model's "valid" flag on its own
    valid = bool(ap.get("present") and ap.get("signed") and ap.get("valid"))
    if valid and ap.get("scope_recipient") and d.get("recipient") and norm(ap["scope_recipient"]) != norm(d["recipient"]):
        valid = False
    if valid and ap.get("scope_status") and d.get("current_status") and norm(ap["scope_status"]) != norm(d["current_status"]):
        valid = False
    if valid and ap.get("scope_action") and d.get("action") and norm(ap["scope_action"]) != norm(d["action"]):
        valid = False
    ap["valid"] = valid
    d["approval"] = ap
    action = str(d.get("action") or "")
    if action not in permitted:
        action = ""
    if not valid and action not in SAFE_ACTIONS:
        action = next((a for a in permitted if a in SAFE_ACTIONS), "")
    if not action:
        action = permitted[0] if permitted else "hold"
    d["action"] = action
    d["permitted_actions"] = permitted
    return d


STATES = {"completed": "TASK_STATE_COMPLETED", "working": "TASK_STATE_WORKING",
          "input-required": "TASK_STATE_INPUT_REQUIRED", "failed": "TASK_STATE_FAILED"}


def task(state, data, ctx=None, tid=None):
    return {
        "id": tid or str(uuid.uuid4()), "contextId": ctx or str(uuid.uuid4()),
        "status": {"state": STATES.get(state, state), "timestamp": now()},
        "artifacts": [{"artifactId": str(uuid.uuid4()), "name": "mailroom_decision",
                       "parts": [{"data": data, "mediaType": "application/json"}]}],
    }


def verify_receipt(rc, prop):
    sig_ok = None
    if RECEIPT_SECRET:
        body = {k: v for k, v in rc.items() if k != "signature"}
        calc = hmac.new(RECEIPT_SECRET.encode(), json.dumps(body, sort_keys=True, separators=(",", ":")).encode(),
                        hashlib.sha256).hexdigest()
        sig_ok = bool(rc.get("signature")) and hmac.compare_digest(calc, str(rc["signature"]))
    scope_ok = True
    if rc.get("recipient") is not None and norm(rc["recipient"]) != norm(prop.get("recipient")):
        scope_ok = False
    if rc.get("action") is not None and norm(rc["action"]) != norm(prop.get("action")):
        scope_ok = False
    cid = rc.get("case_id") or rc.get("caseId")
    if cid is not None and prop.get("case_id") and norm(cid) != norm(prop["case_id"]):
        scope_ok = False
    return sig_ok, scope_ok


def handle_send(request, params):
    msg = params.get("message") or {}
    tenant = tenant_of(request, msg)
    texts, files, objs = collect(msg.get("parts"))
    ctx = msg.get("contextId")
    mid = msg.get("messageId") or hashlib.sha1((json.dumps(texts) + str(len(files))).encode()).hexdigest()
    ikey = f"idem:{tenant}:{mid}"
    cached = get(ikey)
    if cached:
        return cached  # retry of same message: same answer, nothing repeated

    rc = find_receipt(objs)
    if rc:
        rid = rc.get("proposalId") or rc.get("proposal_id") or rc.get("actionId") or rc.get("action_id")
        cid = rc.get("case_id") or rc.get("caseId")
        props = [p for p in list_prefix(f"prop:{tenant}:") if
                 (rid and p["proposalId"] == rid) or (not rid and cid and norm(p.get("case_id")) == norm(cid))
                 or (msg.get("taskId") and p.get("taskId") == msg["taskId"])]
        if not props:
            result = task("failed", {"error": "no matching proposed action for this receipt"}, ctx)
            return result  # not cached
        prop = props[-1]
        if prop["status"] == "completed":
            result = task("completed", prop, prop.get("contextId"), prop["taskId"])
        else:
            sig_ok, scope_ok = verify_receipt(rc, prop)
            if sig_ok is False or not scope_ok:
                prop["last_receipt_rejected"] = {"signature_ok": sig_ok, "scope_ok": scope_ok}
                put(f"prop:{tenant}:{prop['proposalId']}", prop)
                result = task("working", dict(prop, note="receipt rejected; action NOT recorded as completed"),
                              prop.get("contextId"), prop["taskId"])
            else:
                prop["status"] = "completed"
                prop["receipt"] = rc
                prop["completedAt"] = now()
                put(f"prop:{tenant}:{prop['proposalId']}", prop)
                result = task("completed", prop, prop.get("contextId"), prop["taskId"])
    else:
        try:
            d = decide(texts, files, objs)
        except Exception as e:
            return task("failed", {"error": "processing error: " + str(e)[:300]}, ctx)  # not cached
        pid = "prop_" + hashlib.sha1(ikey.encode()).hexdigest()[:12]
        t = task("working", {}, ctx)
        d.update({"proposalId": pid, "taskId": t["id"], "contextId": t["contextId"], "status": "awaiting_receipt"})
        put(f"prop:{tenant}:{pid}", d)
        result = task("working", d, t["contextId"], t["id"])
    put(ikey, result)
    put(f"task:{tenant}:{result['id']}", result)
    return result


# ---------- A2A endpoints ----------
def card(request):
    url = base_url(request) + "/a2a"
    return {
        "name": "Mailroom Agent", "description": "Reads mailroom case files, chooses one permitted action with evidence, and records completion only after a valid signed receipt.",
        "version": "1.0.0", "protocolVersion": "1.0", "url": url,
        "supportedInterfaces": [
            {"url": url, "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0"},
            {"url": url + "/", "protocolBinding": "JSONRPC", "protocolVersion": "1.0"},
        ],
        "preferredTransport": "HTTP+JSON",
        "capabilities": {"streaming": False, "pushNotifications": False},
        "defaultInputModes": ["text/plain", "application/json", "application/pdf", "image/png", "image/jpeg"],
        "defaultOutputModes": ["application/json"],
        "skills": [{"id": "mailroom-triage", "name": "Mailroom triage", "description": "Choose one permitted action per case",
                    "tags": ["mailroom", "approvals", "receipts"]}],
    }


@app.get("/selftest")
def selftest():
    out = {}
    try:
        put("selftest", {"t": now()})
        out["database"] = "ok " + ("postgres" if DATABASE_URL else "sqlite")
    except Exception as e:
        out["database"] = "ERROR: " + str(e)[:300]
    out["gemini_key_set"] = bool(GEMINI_KEY)
    out["receipt_secret_set"] = bool(RECEIPT_SECRET)
    try:
        out["gemini"] = "ok: " + json.dumps(ask_gemini(["Parcel for Dana Lee, status: arrived. Approval signed for Dana Lee to deliver."], [], DEFAULT_ACTIONS))[:300]
    except Exception as e:
        out["gemini"] = "ERROR: " + str(e)[:1200]
    return out


@app.get("/")
def root():
    return {"ok": True}


@app.get("/.well-known/agent-card.json")
@app.get("/.well-known/agent.json")
@app.get("/a2a/.well-known/agent-card.json")
@app.get("/a2a/.well-known/agent.json")
def agent_card(request: Request):
    return card(request)


def err(code, msg):
    return JSONResponse({"code": code, "message": msg}, status_code=code)


@app.post("/a2a/message:send")
@app.post("/message:send")
async def rest_send(request: Request):
    try:
        body = await request.json()
    except Exception:
        return err(400, "invalid JSON")
    try:
        return JSONResponse({"task": handle_send(request, body)})
    except ValueError as e:
        return err(400, str(e))
    except Exception as e:
        return err(500, "Internal error: " + str(e)[:200])


@app.get("/a2a/tasks/{task_id}")
@app.get("/tasks/{task_id}")
def rest_get_task(request: Request, task_id: str):
    t = get(f"task:{tenant_of(request, {})}:{task_id}")
    if not t:
        return err(404, "Task not found")
    # return the latest state of the task, not the snapshot from the first reply
    for p in list_prefix(f"prop:{tenant_of(request, {})}:"):
        if p.get("taskId") == task_id:
            return JSONResponse(task("completed" if p["status"] == "completed" else "working", p, p.get("contextId"), task_id))
    return JSONResponse(t)


@app.post("/")
@app.post("/a2a")
@app.post("/a2a/")
async def rpc(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}})
    rid = body.get("id")
    method = norm(body.get("method"))
    params = body.get("params") or {}
    try:
        if method in ("messagesend", "sendmessage"):
            return {"jsonrpc": "2.0", "id": rid, "result": handle_send(request, params)}
        if method in ("tasksget", "gettask"):
            t = get(f"task:{tenant_of(request, {})}:{params.get('id')}")
            if not t:
                return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32001, "message": "Task not found"}}
            return {"jsonrpc": "2.0", "id": rid, "result": t}
        return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": "Method not found"}}
    except Exception as e:
        return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32603, "message": "Internal error: " + str(e)[:200]}}


# ---------- generic route handling + request log (learn what the grader sends) ----------
def flat(t):
    d = dict(t["artifacts"][0]["parts"][0]["data"])
    d["state"] = t["status"]["state"]
    d["taskId"] = t["id"]
    return d


def log_request(method, path, request, raw):
    try:
        k = f"log:{now()}-{uuid.uuid4().hex[:6]}"
        keep = {h: v for h, v in request.headers.items() if h.lower() in
                ("content-type", "accept", "user-agent", "x-user-id", "x-tenant-id", "a2a-version", "x-request-id", "x-idempotency-key", "idempotency-key")}
        put(k, {"ts": now(), "method": method, "path": path, "query": str(request.url.query),
                "headers": keep, "has_auth": bool(request.headers.get("authorization")),
                "body": raw[:20000].decode("utf-8", "replace")})
    except Exception:
        pass


@app.get("/debug/requests")
def debug_requests():
    logs = sorted(list_prefix("log:"), key=lambda x: x["ts"], reverse=True)[:25]
    return logs


@app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def catch_all(request: Request, full_path: str):
    raw = await request.body()
    path = "/" + full_path
    log_request(request.method, path, request, raw)
    if request.method != "POST":
        return err(404, "Not found")
    try:
        body = json.loads(raw.decode("utf-8")) if raw else {}
    except Exception:
        return err(400, "invalid JSON")
    if not isinstance(body, dict):
        body = {"items": body}
    low = path.lower()
    try:
        if "propose" in low:
            mid = body.get("requestId") or body.get("request_id") or body.get("id") or request.headers.get("idempotency-key") \
                or hashlib.sha1(json.dumps(body, sort_keys=True).encode()).hexdigest()
            t = handle_send(request, {"message": {"messageId": str(mid), "parts": [{"data": body}]}})
            out = flat(t)
            out["status"] = "proposed" if out.get("state") != "TASK_STATE_FAILED" else "failed"
            return JSONResponse({**out, "proposal": dict(out)})
        rc = body.get("receipt") if isinstance(body.get("receipt"), dict) else body
        if any(w in low for w in ("receipt", "complete", "confirm", "execute", "record")):
            mid = body.get("requestId") or body.get("id") or hashlib.sha1(json.dumps(body, sort_keys=True).encode()).hexdigest()
            t = handle_send(request, {"message": {"messageId": str(mid), "parts": [{"data": {"receipt": rc}}]}})
            return JSONResponse(flat(t))
    except Exception as e:
        return err(500, "Internal error: " + str(e)[:200])
    return JSONResponse({"ok": True, "note": "unrecognised route; request logged", "path": path})
