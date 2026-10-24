import os, sys
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
try:
    import django; django.setup()
except: pass

from api.linguistic.tokenizer import ArisTokenizer
from api.relational.relational import ArisRelationalBinder
from api.linguistic.pipeline import ArisLinguisticPipeline
from api.cognitive.epistemic_state import ArisGoalFormer, ArisEpistemicEvaluator, ArisCapabilityMatcher

# Build facts
sentences = [
    'Dr. Elena Rostova designed CryoCore Propulsion.',
    'CryoCore Propulsion integrated into AeroSpace Dynamics.',
    'AeroSpace Dynamics operates out of Toulouse Advanced Hangar.',
    'Toulouse Advanced Hangar receives power from South France Grid.',
    'South France Grid governed by TransEuropean Power Authority.',
    'TransEuropean Power Authority audited by Brussels Regulatory Commission.',
    'Brussels Regulatory Commission appeals to Belgian Federal Tribunal.',
    'Belgian Federal Tribunal presided over by Chief Justice Henri Laurent.',
    'AeroSpace Dynamics procures components from Valence Microelectronics.',
    'Valence Microelectronics hosts servers at Lyon Data Center.',
    'Lyon Data Center interconnected with Geneva Quantum Lab.',
    'Geneva Quantum Lab managed by Dr. Martin Keller.',
    'TransEuropean Power Authority headquartered in Brussels Central District.',
    'Brussels Central District situated in Belgium.',
    'CryoCore Propulsion aerodynamically tested at Bordeaux Wind Tunnel.',
    'Brussels Regulatory Commission publishes directives to EU Environmental Registry.',
]

facts = []
for idx, stmt in enumerate(sentences):
    toks = ArisTokenizer.tokenize(stmt)
    cands = ArisTokenizer.extract_candidates(stmt, toks)
    rels = ArisRelationalBinder.bind_relations(stmt, toks, cands)
    facts.append({'id': f'stress_fact_{idx+1}', 'statement': stmt, 'tokens': toks, 'candidates': cands, 'relations': rels, 'confidence': 1.0})

# Test 04: Which facility does AeroSpace Dynamics operate out of?
query04 = 'Which facility does AeroSpace Dynamics operate out of?'
frame04 = ArisLinguisticPipeline.process(query04)
goal04 = ArisGoalFormer.form_goal(query04, frame04.tokens, frame04.candidates, frame04.bindings, frame=frame04)
print('=== TEST 04 DIAGNOSTICS ===')
print(f'is_question: {frame04.is_question}')
print(f'candidates: {[c["text"] for c in frame04.candidates]}')
print(f'bindings: {[(b["subject"]["text"], b["relation"]["normalized"], b["object"]["text"]) for b in frame04.bindings] if frame04.bindings else None}')
print(f'goal target_type: {goal04.target_type}')
print(f'goal anchor: {goal04.anchor_entity}')
print(f'goal relational_constraint: {goal04.relational_constraint}')

epeval04 = ArisEpistemicEvaluator.evaluate(goal04, memory_facts=facts)
print(f'epistemic state: {epeval04.get("state")}')
print(f'matched_cap: {ArisCapabilityMatcher.match(epeval04, goal04)}')
print()

# Test 07: Who is the maritime fleet admiral of CryoCore Propulsion?
query07 = 'Who is the maritime fleet admiral of CryoCore Propulsion?'
frame07 = ArisLinguisticPipeline.process(query07)
goal07 = ArisGoalFormer.form_goal(query07, frame07.tokens, frame07.candidates, frame07.bindings, frame=frame07)
print('=== TEST 07 DIAGNOSTICS ===')
print(f'is_question: {frame07.is_question}')
print(f'candidates: {[c["text"] for c in frame07.candidates]}')
print(f'bindings: {[(b["subject"]["text"], b["relation"]["normalized"], b["object"]["text"]) for b in frame07.bindings] if frame07.bindings else None}')
print(f'goal target_type: {goal07.target_type}')
print(f'goal anchor: {goal07.anchor_entity}')
print(f'goal relational_constraint: {goal07.relational_constraint}')

epeval07 = ArisEpistemicEvaluator.evaluate(goal07, memory_facts=facts)
print(f'epistemic state: {epeval07.get("state")}')
print(f'matched_cap: {ArisCapabilityMatcher.match(epeval07, goal07)}')
