"""
Small web app for the support agent and the improvement loop.

    python app/server.py            then open http://localhost:8000

Uses only the Python standard library plus the project's own code, so there
is nothing extra to install. On a host like Render, it reads the PORT
environment variable.
"""
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.join(ROOT, 'evaluation'))

import pipeline  # noqa: E402
import hill_climb as hc  # noqa: E402

INDEX_PATH = os.path.join(ROOT, 'app', 'index.html')
MAX_TEXT_LENGTH = 1000

agent = pipeline.Agent()
judge = hc.load_judge_scores()
with open(hc.BASELINE_PATH, encoding='utf-8') as f:
    baseline = json.load(f)

# The experiments patch the escalation function for a moment, so requests
# that run the agent take turns.
lock = threading.Lock()
loop_cache = {}


def decide(text: str, experiment=None) -> dict:
    original = pipeline.escalation_decide
    with lock:
        if experiment:
            pipeline.escalation_decide = hc.EXPERIMENTS[experiment][0]
        try:
            return agent.process(text)
        finally:
            pipeline.escalation_decide = original


def run_loop(experiment=None) -> dict:
    key = experiment or 'current'
    if key not in loop_cache:
        with lock:
            cases = hc.run_agent(experiment)
        cur = hc.summarize(cases, judge)
        result = hc.compare(baseline, cur)
        texts = {c['pair_id']: c['customer_text_clean'] for c in cases}
        loop_cache[key] = {
            'experiment': key,
            'description': hc.EXPERIMENTS[experiment][1] if experiment else 'The code as it is now.',
            'verdict': result['verdict'],
            'reasons': result['reasons'],
            'metrics': [
                {'name': 'Needed a human, but was auto-handled', 'guardrail': True,
                 'before': baseline['dangerous_auto_handles'], 'after': cur['dangerous_auto_handles']},
                {'name': 'Safety messages auto-handled', 'guardrail': True,
                 'before': baseline['safety_escalations_missed'], 'after': cur['safety_escalations_missed']},
                {'name': 'Verified auto-resolutions', 'guardrail': False,
                 'before': baseline['verified_auto_resolutions'], 'after': cur['verified_auto_resolutions']},
                {'name': 'Auto-handled with the wrong intent', 'guardrail': False,
                 'before': baseline['failure_counts']['WRONG_INTENT_AUTO_HANDLE'],
                 'after': cur['failure_counts']['WRONG_INTENT_AUTO_HANDLE']},
                {'name': 'Escalated, but a human was not needed', 'guardrail': False,
                 'before': baseline['failure_counts']['UNNECESSARY_ESCALATION'],
                 'after': cur['failure_counts']['UNNECESSARY_ESCALATION']},
            ],
            'fixed': [{'pair_id': p, 'was': t, 'text': texts[p]} for p, t in result['fixed']],
            'new_failures': [{'pair_id': p, 'now': t, 'text': texts[p]} for p, t in result['regressed']],
        }
    return loop_cache[key]


class Handler(BaseHTTPRequestHandler):
    def send_json(self, status, payload):
        body = json.dumps(payload).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path, _, query = self.path.partition('?')
        if path == '/':
            with open(INDEX_PATH, 'rb') as f:
                body = f.read()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == '/api/experiments':
            self.send_json(200, [{'id': k, 'description': v[1]} for k, v in sorted(hc.EXPERIMENTS.items())])
        elif path == '/api/loop':
            params = dict(p.split('=', 1) for p in query.split('&') if '=' in p)
            experiment = params.get('experiment') or None
            if experiment and experiment not in hc.EXPERIMENTS:
                self.send_json(400, {'error': 'unknown experiment'})
                return
            self.send_json(200, run_loop(experiment))
        else:
            self.send_json(404, {'error': 'not found'})

    def do_POST(self):
        if self.path != '/api/decide':
            self.send_json(404, {'error': 'not found'})
            return
        try:
            length = int(self.headers.get('Content-Length', 0))
            data = json.loads(self.rfile.read(min(length, 10_000)) or b'{}')
        except (ValueError, json.JSONDecodeError):
            self.send_json(400, {'error': 'invalid JSON'})
            return
        text = str(data.get('text', '')).strip()
        experiment = data.get('experiment') or None
        if not text:
            self.send_json(400, {'error': 'text is required'})
            return
        if len(text) > MAX_TEXT_LENGTH:
            self.send_json(400, {'error': f'text must be at most {MAX_TEXT_LENGTH} characters'})
            return
        if experiment and experiment not in hc.EXPERIMENTS:
            self.send_json(400, {'error': 'unknown experiment'})
            return
        self.send_json(200, decide(text, experiment))

    def log_message(self, fmt, *args):
        sys.stderr.write('%s %s\n' % (self.command, self.path))


def main():
    port = int(os.environ.get('PORT', 8000))
    server = ThreadingHTTPServer(('0.0.0.0', port), Handler)
    print(f'Open http://localhost:{port}')
    server.serve_forever()


if __name__ == '__main__':
    main()
