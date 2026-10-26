import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'evaluation'))

import hill_climb as hc  # noqa: E402


def case(pid, gold_intent='BAGGAGE', intent='BAGGAGE', gold_esc=False, decision='AUTO_HANDLE'):
    return {'pair_id': pid, 'gold_intent': gold_intent, 'intent': intent,
            'gold_escalation': gold_esc, 'decision': decision}


class TestClassifyCase(unittest.TestCase):
    def test_ok_auto_handle(self):
        self.assertEqual(hc.classify_case(case('a')), 'OK')

    def test_ok_escalation(self):
        self.assertEqual(hc.classify_case(case('a', gold_esc=True, decision='ESCALATE')), 'OK')

    def test_dangerous(self):
        self.assertEqual(hc.classify_case(case('a', gold_esc=True)), 'DANGEROUS_AUTO_HANDLE')

    def test_wrong_intent(self):
        self.assertEqual(hc.classify_case(case('a', intent='LOYALTY')), 'WRONG_INTENT_AUTO_HANDLE')

    def test_unnecessary_escalation(self):
        self.assertEqual(hc.classify_case(case('a', decision='ESCALATE')), 'UNNECESSARY_ESCALATION')


class TestVerdict(unittest.TestCase):
    def run_pair(self, before, after):
        return hc.compare(hc.summarize(before, {}), hc.summarize(after, {}))

    def test_no_change(self):
        cases = [case('a'), case('b', gold_esc=True)]
        self.assertEqual(self.run_pair(cases, cases)['verdict'], 'NO CHANGE')

    def test_fixing_a_dangerous_case_is_improved(self):
        r = self.run_pair([case('a', gold_esc=True)],
                          [case('a', gold_esc=True, decision='ESCALATE')])
        self.assertEqual(r['verdict'], 'IMPROVED')
        self.assertEqual(r['fixed'], [('a', 'DANGEROUS_AUTO_HANDLE')])

    def test_new_dangerous_case_is_blocked(self):
        r = self.run_pair([case('a', gold_esc=True, decision='ESCALATE')],
                          [case('a', gold_esc=True)])
        self.assertEqual(r['verdict'], 'BLOCKED')

    def test_safety_guardrail_blocks_even_if_other_cases_improve(self):
        before = [case('s', gold_intent='SAFETY_SECURITY', intent='SAFETY_SECURITY', decision='ESCALATE'),
                  case('x', gold_esc=True), case('y', gold_esc=True)]
        after = [case('s', gold_intent='SAFETY_SECURITY', intent='SAFETY_SECURITY'),
                 case('x', gold_esc=True, decision='ESCALATE'),
                 case('y', gold_esc=True, decision='ESCALATE')]
        r = self.run_pair(before, after)
        self.assertEqual(r['verdict'], 'BLOCKED')

    def test_large_drop_in_resolutions_is_blocked(self):
        judge = {str(i): {'relevance': 4, 'groundedness': 4, 'actionability': 4, 'tone': 4,
                          'unsupported_claim': 'NO'} for i in range(10)}
        before = [case(str(i)) for i in range(10)]
        after = [case(str(i), decision='ESCALATE') for i in range(10)]
        r = hc.compare(hc.summarize(before, judge), hc.summarize(after, judge))
        self.assertEqual(r['verdict'], 'BLOCKED')


if __name__ == '__main__':
    unittest.main()
