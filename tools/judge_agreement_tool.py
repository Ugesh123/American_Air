"""
Human-vs-judge agreement tool. Samples 50-75 generated replies and lets you
score each one on the SAME rubric the LLM judge uses, blind to the judge's
score (the judge score is never shown while you rate).

Prerequisite: results/generated_replies.jsonl and results/judge_scores.jsonl
must already exist (both are committed in this repo; regenerate them with
generate_replies.py and run_judge.py if you change the pipeline).

Run:
    python3 tools/judge_agreement_tool.py
Then open:
    http://localhost:8421

Saves to data/judge_agreement/human_judge_scores.json as you go.
"""
import json
import os
import random
import http.server
import socketserver
import urllib.parse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GENERATED_PATH = os.path.join(ROOT, 'results', 'generated_replies.jsonl')
JUDGE_SCORES_PATH = os.path.join(ROOT, 'results', 'judge_scores.jsonl')
OUT_DIR = os.path.join(ROOT, 'data', 'judge_agreement')
HUMAN_SCORES_PATH = os.path.join(OUT_DIR, 'human_judge_scores.json')
SAMPLE_PATH = os.path.join(OUT_DIR, 'sample_pair_ids.json')
PORT = 8421
SAMPLE_SIZE = 60
SEED = 42


def build_sample():
    """Deterministically picks SAMPLE_SIZE pair_ids from the generated replies,
    seeded so re-running doesn't change the sample."""
    with open(GENERATED_PATH, encoding='utf-8') as f:
        generated = [json.loads(l) for l in f if json.loads(l).get('reply')]
    rng = random.Random(SEED)
    rng.shuffle(generated)
    sample = generated[:SAMPLE_SIZE]
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(SAMPLE_PATH, 'w', encoding='utf-8') as f:
        json.dump([r['pair_id'] for r in sample], f)
    return sample


def load_labels():
    if os.path.exists(HUMAN_SCORES_PATH):
        return json.load(open(HUMAN_SCORES_PATH, encoding='utf-8'))
    return {}


def save_labels(labels):
    os.makedirs(OUT_DIR, exist_ok=True)
    tmp = HUMAN_SCORES_PATH + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(labels, f, indent=2)
    os.replace(tmp, HUMAN_SCORES_PATH)


PAGE = """<!DOCTYPE html><html><head><meta charset="utf-8"><title>Judge Agreement</title>
<style>
body{font-family:-apple-system,sans-serif;background:#0f1216;color:#e8eaed;max-width:700px;margin:30px auto;padding:0 20px}
.card{background:#171b21;border:1px solid #2a2f38;border-radius:8px;padding:16px;margin-bottom:14px}
textarea,input{width:100%;padding:8px;background:#12151a;border:1px solid #2a2f38;color:#e8eaed;border-radius:5px}
label{display:block;font-size:12px;color:#9aa4b2;margin:10px 0 4px;font-weight:600}
button{padding:10px 18px;border-radius:6px;border:none;cursor:pointer;font-weight:600;margin-right:8px}
#save{background:#3fb950;color:#fff}
#prev{background:#2a2f38;color:#fff}
.progress{color:#9aa4b2;font-size:13px}
</style></head><body>
<h2>Human vs Judge Agreement - <span id="num"></span> / <span id="total"></span></h2>
<div class="progress" id="prog"></div>
<div class="card">
  <div><b>Customer message:</b></div>
  <div id="cust" style="margin:8px 0;padding:8px;background:#12151a;border-radius:4px"></div>
  <div><b>Generated reply (score THIS, blind to the judge's score):</b></div>
  <div id="reply" style="margin:8px 0;padding:8px;background:#12151a;border-radius:4px"></div>
</div>
<div class="card">
  <label>Relevance (1-5)</label><input id="relevance" type="number" min="1" max="5">
  <label>Groundedness (1-5)</label><input id="groundedness" type="number" min="1" max="5">
  <label>Actionability (1-5)</label><input id="actionability" type="number" min="1" max="5">
  <label>Tone/professionalism (1-5)</label><input id="tone" type="number" min="1" max="5">
  <label>Unsupported claim?</label>
  <select id="unsupported"><option value="">--</option><option value="YES">YES</option><option value="NO">NO</option></select>
</div>
<button id="prev" onclick="go(-1)">&larr; Previous</button>
<button id="save" onclick="saveNext()">Save &amp; Next &rarr;</button>
<script>
let DATA=null, cur=0, state={};
async function load(){
  const r = await fetch('/api/data'); DATA = await r.json();
  DATA.items.forEach(it=>state[it.pair_id]=DATA.labels[it.pair_id]||null);
  render();
}
function render(){
  const it = DATA.items[cur];
  document.getElementById('num').innerText=cur+1;
  document.getElementById('total').innerText=DATA.items.length;
  document.getElementById('prog').innerText = Object.values(state).filter(Boolean).length + ' scored so far';
  document.getElementById('cust').innerText = it.customer_text_clean;
  document.getElementById('reply').innerText = it.reply;
  const ex = state[it.pair_id];
  document.getElementById('relevance').value = ex?ex.relevance:'';
  document.getElementById('groundedness').value = ex?ex.groundedness:'';
  document.getElementById('actionability').value = ex?ex.actionability:'';
  document.getElementById('tone').value = ex?ex.tone:'';
  document.getElementById('unsupported').value = ex?ex.unsupported_claim:'';
}
function go(d){ cur=Math.max(0,Math.min(DATA.items.length-1,cur+d)); render(); }
async function saveNext(){
  const it = DATA.items[cur];
  const label = {
    relevance: +document.getElementById('relevance').value,
    groundedness: +document.getElementById('groundedness').value,
    actionability: +document.getElementById('actionability').value,
    tone: +document.getElementById('tone').value,
    unsupported_claim: document.getElementById('unsupported').value,
  };
  state[it.pair_id]=label;
  await fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({pair_id:it.pair_id,label})});
  if(cur<DATA.items.length-1){cur+=1;render();} else {alert('All done!'); render();}
}
load();
</script></body></html>"""


class Handler(http.server.BaseHTTPRequestHandler):
    def _json(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        p = urllib.parse.urlparse(self.path)
        if p.path == '/':
            body = PAGE.encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif p.path == '/api/data':
            sample = json.load(open(SAMPLE_PATH, encoding='utf-8'))
            generated = {json.loads(l)['pair_id']: json.loads(l) for l in open(GENERATED_PATH, encoding='utf-8')}
            items = [generated[pid] for pid in sample]
            self._json({'items': items, 'labels': load_labels()})
        else:
            self.send_response(404); self.end_headers()

    def do_POST(self):
        p = urllib.parse.urlparse(self.path)
        if p.path == '/api/save':
            length = int(self.headers.get('Content-Length', 0))
            body = json.loads(self.rfile.read(length))
            labels = load_labels()
            labels[body['pair_id']] = body['label']
            save_labels(labels)
            self._json({'ok': True})
        else:
            self.send_response(404); self.end_headers()

    def log_message(self, *a):
        pass


if __name__ == '__main__':
    if not os.path.exists(GENERATED_PATH):
        raise SystemExit(f'{GENERATED_PATH} not found - run generate_replies.py first (requires your API key).')
    if not os.path.exists(SAMPLE_PATH):
        build_sample()
    with socketserver.TCPServer(('localhost', PORT), Handler) as httpd:
        print(f'Judge-agreement tool running at http://localhost:{PORT}')
        httpd.serve_forever()
