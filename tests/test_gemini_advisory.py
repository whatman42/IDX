from src.python.advisory.gemini import GeminiAdvisor, validate_advisory, is_high_value_event
from src.python.advisory.templates_id import format_deterministic_id

def test_validate_rejects_missing_fields():
    ok, _ = validate_advisory({"title": "x"}, {"symbol": "BBCA"})
    assert not ok

def test_validate_rejects_hallucinated_number():
    data = {"title": "t", "summary": "beli di 99999999", "system_action": "a", "severity": "INFO", "why": [], "risk_explanation": "r"}
    ok, reason = validate_advisory(data, {"symbol": "BBCA", "entry_price": 9000})
    assert not ok and "hallucinated" in reason

def test_validate_accepts_context_numbers():
    data = {"title": "Beli", "summary": "Harga 9000", "system_action": "simulasi", "severity": "INFO", "why": ["sinyal"], "risk_explanation": "bisa turun"}
    ok, _ = validate_advisory(data, {"entry_price": 9000, "symbol": "BBCA"})
    assert ok

def test_disabled_without_key():
    adv = GeminiAdvisor(api_key="", enabled=True)
    assert not adv.enabled
    r = adv.explain("BUY", {"symbol": "BBCA"})
    assert r.source == "fallback" and not r.ok

def test_deterministic_buy_indonesian():
    text = format_deterministic_id("BUY", {"symbol": "BBCA", "entry_price": 9000, "qty": 100, "confidence": 0.7, "take_profit": 9500, "stop_loss": 8700, "cash": 1e8, "equity": 1e8, "regime": "neutral"})
    assert "SINYAL BELI" in text and "SIMULASI" in text and "BBCA" in text and "APA YANG TERJADI" in text

def test_high_value_events():
    assert is_high_value_event("BUY")
    assert not is_high_value_event("DEBUG_INTERNAL")
