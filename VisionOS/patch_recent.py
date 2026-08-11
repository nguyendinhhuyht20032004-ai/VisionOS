import sys

def patch_file(path):
    with open(path, 'r') as f:
        content = f.read()

    # Patch vectordb.py
    if path.endswith('vectordb.py'):
        old = '''    def recent(self, limit: int = 20) -> List[dict]:
        """Sự kiện GẦN NHẤT (mới → cũ)."""
        if self.backend == "qdrant" and self._client is not None:
            try:
                start_id = max(0, self._n - limit)
                ids = list(range(start_id, self._n))
                pts = self._client.retrieve(self.collection, ids=ids, with_payload=True) if ids else []
                items = [{"id": p.id, "payload": p.payload or {}} for p in pts]
                return sorted(items, key=lambda x: x["payload"].get("ts", 0), reverse=True)[:limit]
            except Exception:  # noqa: BLE001
                pass
        return [{"id": it["id"], "payload": it["payload"]} for it in self._mem[-limit:]][::-1]'''

        new = '''    def recent(self, limit: int = 20, source: Optional[str] = None) -> List[dict]:
        """Sự kiện GẦN NHẤT (mới → cũ)."""
        if self.backend == "qdrant" and self._client is not None:
            try:
                fetch_limit = limit * 10 if source else limit
                start_id = max(0, self._n - fetch_limit)
                ids = list(range(start_id, self._n))
                pts = self._client.retrieve(self.collection, ids=ids, with_payload=True) if ids else []
                items = [{"id": p.id, "payload": p.payload or {}} for p in pts]
                if source:
                    items = [it for it in items if it["payload"].get("source") == source]
                return sorted(items, key=lambda x: x["payload"].get("ts", 0), reverse=True)[:limit]
            except Exception:  # noqa: BLE001
                pass
        items = [{"id": it["id"], "payload": it["payload"]} for it in self._mem]
        if source:
            items = [it for it in items if it["payload"].get("source") == source]
        return items[-limit:][::-1]'''
        content = content.replace(old, new)

    # Patch app.py
    elif path.endswith('app.py'):
        # Patch api_events
        old_api = '''@app.get("/api/events")
def api_events(limit: int = 20):
    return {"recent_events": _VDB.recent(limit), **_VDB.status()}'''
        new_api = '''@app.get("/api/events")
def api_events(source: Optional[str] = None, limit: int = 20):
    return {"recent_events": _VDB.recent(limit, source=source), **_VDB.status()}'''
        content = content.replace(old_api, new_api)

        # Patch JS
        old_js = '''function loadEvents(){
 fetch('/api/events?limit=50').then(r=>r.json()).then(d=>{'''
        new_js = '''function loadEvents(){
 let src = document.getElementById('source').value;
 fetch('/api/events?limit=50&source=' + encodeURIComponent(src)).then(r=>r.json()).then(d=>{'''
        content = content.replace(old_js, new_js)

    with open(path, 'w') as f:
        f.write(content)

patch_file('recognition/service/vectordb.py')
patch_file('recognition/service/app.py')
