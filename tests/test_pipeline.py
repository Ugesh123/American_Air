"""
Test suite. Uses only Python's built-in unittest (no pytest/extra installs
needed, consistent with the rest of this project's laptop-friendly philosophy).

Run:
    python3 -m unittest discover tests -v
"""
import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from preprocessing.clean import (
    clean_customer_text, clean_agent_text, normalize_for_dedup, is_effectively_empty
)
from escalation.policy import decide as escalation_decide


class TestCleaning(unittest.TestCase):
    def test_html_unescape(self):
        self.assertIn('&', clean_customer_text('@AmericanAir Tom &amp; Jerry'))

    def test_url_becomes_placeholder(self):
        out = clean_customer_text('@AmericanAir see https://t.co/abc123')
        self.assertIn('[URL]', out)
        self.assertNotIn('https://', out)

    def test_brand_mention_stripped(self):
        out = clean_customer_text('@AmericanAir my flight is delayed')
        self.assertNotIn('@AmericanAir', out.lower())

    def test_other_mention_becomes_placeholder(self):
        out = clean_customer_text('@AmericanAir cc @someuser please help')
        self.assertIn('[USER]', out)

    def test_link_only_tweet_is_effectively_empty(self):
        # regression test for the placeholder-counts-as-content bug found in Step 2
        cleaned = clean_customer_text('@AmericanAir https://t.co/xyz')
        self.assertTrue(is_effectively_empty(cleaned))

    def test_real_content_is_not_empty(self):
        cleaned = clean_customer_text('@AmericanAir my flight is delayed again')
        self.assertFalse(is_effectively_empty(cleaned))

    def test_normalize_for_dedup_strips_punctuation(self):
        self.assertEqual(normalize_for_dedup('Thanks!!'), normalize_for_dedup('thanks'))


class TestEscalationPolicy(unittest.TestCase):
    def test_safety_always_escalates(self):
        result = escalation_decide('SAFETY_SECURITY', 0.99, 0.99, 'I was assaulted on the flight')
        self.assertEqual(result['decision'], 'ESCALATE')

    def test_low_intent_confidence_escalates(self):
        result = escalation_decide('BAGGAGE', 0.10, 0.9, 'my bag is missing')
        self.assertEqual(result['decision'], 'ESCALATE')

    def test_low_retrieval_confidence_escalates(self):
        result = escalation_decide('BAGGAGE', 0.9, 0.01, 'my bag is missing')
        self.assertEqual(result['decision'], 'ESCALATE')

    def test_legal_language_forces_escalation_regardless_of_intent(self):
        result = escalation_decide('COMPLIMENT', 0.99, 0.99, 'great flight but I will sue if this happens again')
        self.assertEqual(result['decision'], 'ESCALATE')

    def test_confident_safe_case_auto_handles(self):
        result = escalation_decide('FLIGHT_STATUS_DELAY_CANCEL', 0.95, 0.9, 'is my flight delayed')
        self.assertEqual(result['decision'], 'AUTO_HANDLE')

    def test_every_decision_has_a_reason(self):
        result = escalation_decide('BAGGAGE', 0.9, 0.9, 'where is my bag')
        self.assertTrue(result['reason'])


class TestDataIntegrity(unittest.TestCase):
    """Re-verifies the Step 2 leakage guarantees and Step 3 golden-set schema
    directly against the shipped data files, so a corrupted/edited data file
    would fail this test suite rather than silently propagate."""

    @classmethod
    def setUpClass(cls):
        cls.train = [json.loads(l) for l in open(os.path.join(ROOT, 'data', 'processed', 'train.jsonl'), encoding='utf-8')]
        cls.dev = [json.loads(l) for l in open(os.path.join(ROOT, 'data', 'processed', 'dev.jsonl'), encoding='utf-8')]
        cls.eval_pool = [json.loads(l) for l in open(os.path.join(ROOT, 'data', 'processed', 'eval_pool.jsonl'), encoding='utf-8')]
        cls.candidates = [json.loads(l) for l in open(os.path.join(ROOT, 'data', 'golden_set', 'candidates_200.jsonl'), encoding='utf-8')]
        cls.labels = json.load(open(os.path.join(ROOT, 'data', 'golden_set', 'labels.json'), encoding='utf-8'))

    def test_no_thread_spans_multiple_splits(self):
        thread_splits = {}
        for split_name, records in [('train', self.train), ('dev', self.dev), ('eval', self.eval_pool)]:
            for r in records:
                thread_splits.setdefault(r['thread_id'], set()).add(split_name)
        violations = {t: s for t, s in thread_splits.items() if len(s) > 1}
        self.assertEqual(len(violations), 0, f'threads spanning multiple splits: {violations}')

    def test_golden_set_has_200_unique_candidates(self):
        pair_ids = [c['pair_id'] for c in self.candidates]
        self.assertEqual(len(pair_ids), 200)
        self.assertEqual(len(set(pair_ids)), 200)

    def test_all_candidates_have_labels(self):
        cand_ids = set(c['pair_id'] for c in self.candidates)
        label_ids = set(self.labels.keys())
        self.assertEqual(cand_ids, label_ids)

    def test_all_gold_intents_are_valid(self):
        valid = {'FLIGHT_STATUS_DELAY_CANCEL', 'BOOKING_CHANGE', 'BAGGAGE', 'REFUND_COMPENSATION',
                 'CHECKIN_BOARDING', 'LOYALTY_MILES', 'ACCOUNT_WEBSITE_APP', 'SAFETY_SECURITY',
                 'COMPLIMENT', 'COMPLAINT_GENERAL', 'OTHER_UNCLEAR'}
        for pid, lab in self.labels.items():
            self.assertIn(lab['gold_intent'], valid, f'{pid} has invalid gold_intent {lab["gold_intent"]!r}')

    def test_escalation_is_boolean(self):
        for pid, lab in self.labels.items():
            self.assertIsInstance(lab['gold_escalation'], bool, f'{pid} gold_escalation is not boolean')


class TestPipelineSmoke(unittest.TestCase):
    """Confirms the trained model artifacts load and the pipeline runs end to
    end without error. Skips gracefully if the model hasn't been trained yet
    (so this test file is still useful before Step-6 training has run)."""

    def test_agent_processes_a_message(self):
        model_dir = os.path.join(ROOT, 'results', 'models')
        if not os.path.exists(os.path.join(model_dir, 'intent_classifier.joblib')):
            self.skipTest('classifier not trained yet - run src/intent/train_classifier.py first')
        from pipeline import Agent
        agent = Agent()
        result = agent.process('my flight got delayed and nobody told me anything')
        self.assertIn(result['intent'], ['FLIGHT_STATUS_DELAY_CANCEL', 'OTHER_UNCLEAR',  # any valid class is fine here
                                          'BOOKING_CHANGE', 'BAGGAGE', 'REFUND_COMPENSATION', 'CHECKIN_BOARDING',
                                          'LOYALTY_MILES', 'ACCOUNT_WEBSITE_APP', 'SAFETY_SECURITY',
                                          'COMPLIMENT', 'COMPLAINT_GENERAL'])
        self.assertIn(result['decision'], ['AUTO_HANDLE', 'ESCALATE'])
        self.assertTrue(result['reason'])
        self.assertIsInstance(result['evidence'], list)


if __name__ == '__main__':
    unittest.main()
