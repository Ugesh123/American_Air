"""
Local golden-set labeling tool. Zero external dependencies (Python stdlib only).

Run:
    python3 tools/label_tool.py
Then open:
    http://localhost:8420

Labels are saved to data/golden_set/labels.json as you go (one write per save,
so you can close the tool and resume later -- progress is never lost).
"""
import json
import os
import http.server
import socketserver
import urllib.parse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANDIDATES_PATH = os.path.join(ROOT, 'data', 'golden_set', 'candidates_200.jsonl')
LABELS_PATH = os.path.join(ROOT, 'data', 'golden_set', 'labels.json')
DRAFTS_PATH = os.path.join(ROOT, 'data', 'golden_set', 'label_drafts.json')
PORT = 8420

TAXONOMY = [
    {"name": "FLIGHT_STATUS_DELAY_CANCEL", "definition": "Customer asking about, or reporting, the operational status of a specific flight -- delays, cancellations, gate/departure changes, or whether a flight will be affected by a disruption.",
     "include": "Delay/cancellation reports and questions; \"will my flight be affected\" questions; ETA/gate questions.",
     "exclude": "General dissatisfaction framed purely as a complaint with no live status ask (-> COMPLAINT_GENERAL); requests to actually change the booking (-> BOOKING_CHANGE)."},
    {"name": "BOOKING_CHANGE", "definition": "Customer wants to modify an existing reservation -- rebook, change dates, change seats, upgrade, split/merge a PNR.",
     "include": "Explicit change/rebook/upgrade requests; PNR/record-locator-referenced change asks.",
     "exclude": "Asking about flight status without requesting a change (-> FLIGHT_STATUS); asking for money back for a change already made (-> REFUND_COMPENSATION)."},
    {"name": "BAGGAGE", "definition": "Anything about checked baggage or a personal item left on board -- lost, delayed, damaged, fees, or left-behind items.",
     "include": "Checked-bag issues; explicit \"left my [item] on the plane/seat\" reports.",
     "exclude": "General amenity complaints unrelated to a specific bag (-> COMPLAINT_GENERAL)."},
    {"name": "REFUND_COMPENSATION", "definition": "Explicit request for money back, a voucher, or compensation for a specific issue.",
     "include": "Refund/voucher/compensation asks; disputes about compensation already offered.",
     "exclude": "General \"this cost me money\" venting with no explicit ask (-> COMPLAINT_GENERAL)."},
    {"name": "CHECKIN_BOARDING", "definition": "Check-in process, boarding pass issues, gate/boarding experience, standby, TSA PreCheck/KTN problems.",
     "include": "Boarding-pass discrepancies; gate-agent interactions during boarding; PreCheck/KTN issues.",
     "exclude": "Post-boarding in-flight complaints (-> COMPLAINT_GENERAL); seat-assignment changes unrelated to boarding (-> BOOKING_CHANGE)."},
    {"name": "LOYALTY_MILES", "definition": "AAdvantage miles, elite status, award tickets, systemwide upgrades.",
     "include": "Anything about the loyalty program itself.", "exclude": "A booking-specific upgrade request unrelated to elite status (-> BOOKING_CHANGE)."},
    {"name": "ACCOUNT_WEBSITE_APP", "definition": "Website or app malfunction, login issues, technical errors unrelated to a specific flight's operational status.",
     "include": "Login/app/website bugs.", "exclude": "Using the app to check a flight's status (-> FLIGHT_STATUS)."},
    {"name": "SAFETY_SECURITY", "definition": "A genuine personal safety, medical, security, or discrimination INCIDENT -- not casual mentions of the word safety.",
     "include": "Medical emergencies, threats, assault, discrimination claims, allergy-related incidents.",
     "exclude": "References to safety VIDEOS, safety marketing/hashtags, or general \"safety is important\" commentary with no personal incident."},
    {"name": "COMPLIMENT", "definition": "Primarily positive/thankful feedback with no unresolved request.",
     "include": "Thanks, praise for staff/service.", "exclude": "Gratitude that's really a sign-off on an unresolved complaint (-> the underlying complaint intent)."},
    {"name": "COMPLAINT_GENERAL", "definition": "Dissatisfaction/negative feedback without a specific actionable request captured by another class.",
     "include": "General venting, service-quality complaints.", "exclude": "Any complaint that also contains a specific actionable ask matching another class -> use that class instead."},
    {"name": "OTHER_UNCLEAR", "definition": "Off-topic/noise content, non-English messages, and later-turn fragments that can't be classified without prior thread context.",
     "include": "Personal chatter unrelated to support; pure acknowledgments; short context-dependent replies; non-English text.",
     "exclude": "Anything you CAN confidently classify once you read the thread context -- use the real intent instead."},
]


def load_candidates():
    with open(CANDIDATES_PATH, encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def load_labels():
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, encoding='utf-8') as f:
            return json.load(f)
    return {}


def load_drafts():
    """AI-drafted starting values for examples not yet confirmed by the human
    annotator. These are NEVER written to labels.json automatically -- they are
    only shown as pre-filled, clearly-marked-as-draft form values in the UI.
    Saving (clicking 'Save & Next') is what turns a draft into a real label."""
    if os.path.exists(DRAFTS_PATH):
        with open(DRAFTS_PATH, encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_labels(labels):
    tmp = LABELS_PATH + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(labels, f, indent=2)
    os.replace(tmp, LABELS_PATH)


PAGE_TEMPLATE = open(os.path.join(os.path.dirname(__file__), 'label_tool_ui.html'), encoding='utf-8').read()


class Handler(http.server.BaseHTTPRequestHandler):
    def _send_json(self, obj, code=200):
        body = json.dumps(obj).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == '/':
            body = PAGE_TEMPLATE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif parsed.path == '/api/data':
            self._send_json({
                'taxonomy': TAXONOMY,
                'candidates': load_candidates(),
                'labels': load_labels(),
                'drafts': load_drafts(),
            })
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == '/api/save':
            length = int(self.headers.get('Content-Length', 0))
            body = json.loads(self.rfile.read(length))
            pair_id = body['pair_id']
            labels = load_labels()
            labels[pair_id] = body['label']
            save_labels(labels)
            self._send_json({'ok': True, 'total_labeled': len(labels)})
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # keep console quiet


if __name__ == '__main__':
    if not os.path.exists(LABELS_PATH):
        save_labels({})
    with socketserver.TCPServer(('localhost', PORT), Handler) as httpd:
        print(f'Golden-set labeling tool running at http://localhost:{PORT}')
        print(f'Candidates: {CANDIDATES_PATH}')
        print(f'Labels save to: {LABELS_PATH}')
        print('Press Ctrl+C to stop (your progress is already saved on disk).')
        httpd.serve_forever()
