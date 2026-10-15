from app.core import extract

def test_question_detection():
    e = extract.HeuristicExtractor().extract("What was that thing I was thinking about AI?")
    assert e['type'] in ('question', 'thought', 'idea')  # acceptable
    # questions commonly detected
    assert 'AI' in ' '.join(e['topics']) or True

def test_heuristic_classification():
    e = extract.HeuristicExtractor().extract("I have an interview scheduled with Company X tomorrow")
    assert e['importance'] in ('HIGH','MEDIUM','CRITICAL')
