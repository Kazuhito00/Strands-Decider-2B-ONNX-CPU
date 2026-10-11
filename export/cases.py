# -*- coding: utf-8 -*-
"""The requests the oracle (export/make_golden.py) and the gates (verify/) share.

Chosen to hit the branches of the layout: noul / choice / score; several questions about one state; a JSON state;
choice options with and without descriptions and with an em dash in them; noul with custom criteria; non-ASCII text; a
long state; many options; the longest rubric (10 levels); an empty state.
"""

TEAM = {"billing": "charges, refunds, invoices", "technical support": "bugs, outages, errors",
        "sales": "pricing, upgrades, new contracts"}

LONG_STATE = (
    "Ticket #48213, opened Monday. The customer writes: our nightly export has been failing since the update on Friday. "
    "The job times out after 30 minutes, and the log shows repeated 502 responses from the storage gateway. We tried "
    "restarting the worker twice and rotating the credentials. Support replied on Tuesday asking for the job id, which we "
    "sent. No further reply. The finance team needs the report by Thursday morning for the quarterly close, and the CEO "
    "has asked for a status twice already. "
) * 4

CASES = {
    "choice-basic": {
        "state": "Hi, I was charged twice for my Pro subscription this month, please fix.",
        "questions": {"team": {"type": "choice", "instructions": "Which team should handle this?", "criteria": TEAM}}},
    "noul-basic": {
        "state": "Thanks so much for the quick fix yesterday, everything works great now!",
        "questions": {"angry": {"type": "noul", "instructions": "The customer is angry"}}},
    "score-basic": {
        "state": "URGENT: production database is down, all customers affected.",
        "questions": {"urgency": {"type": "score", "instructions": "How urgent is this?",
                                  "criteria": ["low", "medium", "high", "critical"]}}},
    "multi": {
        "state": "This is the third time this week my payouts failed and nobody answers my emails. Unacceptable.",
        "questions": {
            "angry": {"type": "noul", "instructions": "The customer is angry"},
            "team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "payments", "technical": "bugs"}},
            "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "medium", "high"]}}},
    "json-state": {
        "state": {"order": {"id": 1042, "status": "shipped", "items": ["lamp", "desk"]}, "customer": "Aiko", "notes": None},
        "questions": {"intent": {"type": "choice", "instructions": "What does the customer want next?",
                                 "criteria": {"track": "see where the parcel is", "cancel": "cancel the order",
                                              "return": "send the items back"}}}},
    "no-descriptions": {
        "state": "The app crashes every time I open the settings screen on Android 14.",
        "questions": {"kind": {"type": "choice", "instructions": "What kind of report is this?",
                               "criteria": {"bug": "", "feature request": "", "question": "", "complaint": ""}}}},
    "noul-custom": {
        "state": "The deploy finished, all health checks are green and latency is back to normal.",
        "questions": {"incident": {"type": "noul", "instructions": "There is an ongoing incident",
                                   "criteria": {"true": "Something is still broken.", "false": "Everything is healthy."}}}},
    "non-ascii": {
        "state": "Pro プランの料金が今月 2 回引き落とされています。対応をお願いします。 café ☕",
        "questions": {"team": {"type": "choice", "instructions": "どのチームが対応すべきですか？",
                               "criteria": {"請求": "請求・返金", "技術": "不具合・障害", "営業": "価格・契約"}},
                      "angry": {"type": "noul", "instructions": "顧客は怒っている"}}},
    "long-state": {
        "state": LONG_STATE,
        "questions": {"urgent": {"type": "noul", "instructions": "This needs attention today"},
                      "team": {"type": "choice", "instructions": "Which team?",
                               "criteria": {"infra": "storage and gateways", "finance": "reports", "support": "tickets"}}}},
    "score-ten": {
        "state": "The restaurant was fine, the food arrived cold but the staff apologised and gave us dessert.",
        "questions": {"rating": {"type": "score", "instructions": "How would the guest rate the visit?",
                                 "criteria": ["awful", "terrible", "bad", "poor", "so-so", "okay", "fine", "good", "great", "excellent"]}}},
    "many-options": {
        "state": "My package was supposed to arrive on Tuesday and the tracking has not moved since Friday.",
        "questions": {"topic": {"type": "choice", "instructions": "What is this message about?",
                                "criteria": {k: v for k, v in [
                                    ("delivery delay", "the parcel is late"), ("lost parcel", "the parcel is missing"),
                                    ("damaged goods", "the item arrived broken"), ("refund", "money back"),
                                    ("address change", "wrong address"), ("invoice", "billing paperwork"),
                                    ("login", "cannot sign in"), ("praise", "a compliment"), ("spam", "unwanted ads")]}}}},
    "empty-state": {
        "state": "",
        "questions": {"yesno": {"type": "noul", "instructions": "Is Paris the capital of France?"}}},
}
