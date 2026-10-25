"""
End-to-end pipeline: customer message -> intent classification -> historical
retrieval -> escalation decision -> (optional) grounded reply generation.

Reply generation requires an LLM API key and internet access (see
src/generation/generate_replies.py) and is OFF by default here so this
pipeline can run fully offline for intent/retrieval/escalation evaluation.

Usage:
    python3 src/pipeline.py --text "my flight got delayed and nobody told me"
    python3 src/pipeline.py --golden_set   # run over all 200 golden examples
"""
import argparse
import json
import os
import sys

import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from retrieval.retrieve import TfidfRetriever
from escalation.policy import decide as escalation_decide

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, 'results', 'models')
GOLDEN_CANDIDATES = os.path.join(ROOT, 'data', 'golden_set', 'candidates_200.jsonl')
GOLDEN_LABELS = os.path.join(ROOT, 'data', 'golden_set', 'labels.json')


class Agent:
    def __init__(self):
        self.vec = joblib.load(os.path.join(MODEL_DIR, 'intent_vectorizer.joblib'))
        self.clf = joblib.load(os.path.join(MODEL_DIR, 'intent_classifier.joblib'))
        self.classes = list(self.clf.classes_)
        self.retriever = TfidfRetriever()

    def classify(self, text: str):
        X = self.vec.transform([text])
        probs = self.clf.predict_proba(X)[0]
        idx = probs.argmax()
        return self.classes[idx], float(probs[idx])

    def process(self, text: str, generate_reply: bool = False) -> dict:
        intent, intent_confidence = self.classify(text)
        evidence = self.retriever.query(text, k=5)
        retrieval_confidence = evidence[0].similarity if evidence else 0.0

        esc = escalation_decide(intent, intent_confidence, retrieval_confidence, text)

        reply = None
        if generate_reply:
            from generation.generate_replies import generate_grounded_reply
            reply = generate_grounded_reply(text, intent, evidence)

        return {
            'intent': intent,
            'intent_confidence': round(intent_confidence, 4),
            'reply': reply,
            'decision': esc['decision'],
            'reason': esc['reason'],
            'evidence': [
                {'pair_id': e.pair_id, 'similarity': round(e.similarity, 4),
                 'historical_customer_msg': e.customer_text, 'historical_aa_reply': e.aa_reply_text}
                for e in evidence
            ],
        }


def run_on_golden_set(agent: Agent, generate_reply: bool = False):
    candidates = {json.loads(l)['pair_id']: json.loads(l) for l in open(GOLDEN_CANDIDATES, encoding='utf-8')}
    labels = json.load(open(GOLDEN_LABELS, encoding='utf-8'))
    outputs = []
    for pid, lab in labels.items():
        c = candidates[pid]
        result = agent.process(c['customer_text_clean'], generate_reply=generate_reply)
        outputs.append({
            'pair_id': pid, 'customer_text_clean': c['customer_text_clean'],
            'gold_intent': lab['gold_intent'], 'gold_escalation': lab['gold_escalation'],
            **result,
        })
    return outputs


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--text', default=None)
    ap.add_argument('--golden_set', action='store_true')
    ap.add_argument('--generate_reply', action='store_true',
                     help='Requires OPENROUTER_API_KEY and OPENROUTER_MODEL - see src/generation/generate_replies.py')
    args = ap.parse_args()

    agent = Agent()

    if args.golden_set:
        outputs = run_on_golden_set(agent, generate_reply=args.generate_reply)
        out_path = os.path.join(ROOT, 'results', 'pipeline_outputs_golden_set.json')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(outputs, f, indent=2)
        print(f'Ran pipeline on {len(outputs)} golden-set examples -> {out_path}')
    elif args.text:
        result = agent.process(args.text, generate_reply=args.generate_reply)
        print(json.dumps(result, indent=2))
    else:
        ap.print_help()
