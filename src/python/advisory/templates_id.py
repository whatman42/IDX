"""Deterministic beginner-friendly Indonesian notification templates (no LLM)."""
from __future__ import annotations
from typing import Any

def _rp(x: Any) -> str:
    try:
        return f"Rp{int(float(x)):,}".replace(",", ".")
    except Exception:
        return str(x)

def format_deterministic_id(event_type: str, p: dict[str, Any]) -> str:
    et = event_type.upper()
    if et == "BUY":
        return (
            f"\U0001F7E2 SINYAL BELI (SIMULASI)\n\n"
            f"Saham: {p.get('symbol', '-')}\n"
            f"Harga simulasi: {_rp(p.get('entry_price'))}\n"
            f"Jumlah lot: {p.get('qty', '-')}\n"
            f"Keyakinan model: {p.get('confidence', p.get('meta_probability', '-'))}\n"
            f"Target: {_rp(p.get('take_profit'))}\n"
            f"Batas risiko: {_rp(p.get('stop_loss'))}\n\n"
            f"APA YANG TERJADI?\nSistem menemukan peluang beli berdasarkan model yang disetujui.\n\n"
            f"KENAPA?\nModel utama dan filter meta memberi sinyal positif (rezim: {p.get('regime', '-')}).\n\n"
            f"APA YANG DILAKUKAN SISTEM?\nPembelian SIMULASI tercatat di portofolio kertas. Bukan order bursa nyata.\n\n"
            f"RISIKONYA APA?\nHarga bisa turun. Batas risiko (stop) sudah dipasang di level simulasi.\n"
            f"Modal kertas: {_rp(p.get('cash'))} | Nilai portofolio: {_rp(p.get('equity'))}"
        )
    if et in ("NO_BUY", "SKIP"):
        return (
            f"\U0001F534 TIDAK DIBELI\n\nSaham: {p.get('symbol', '-')}\n\n"
            f"APA YANG TERJADI?\nSistem memilih tidak membuka posisi.\n\n"
            f"KENAPA?\n{p.get('reason', 'Sinyal tidak memenuhi ambang keyakinan atau risiko.')}\n\n"
            f"APA YANG DILAKUKAN SISTEM?\nTidak ada transaksi simulasi.\n\n"
            f"RISIKONYA APA?\nMenghindari risiko yang tidak sepadan menurut kebijakan saat ini."
        )
    if et == "PORTFOLIO":
        return (
            f"\U0001F4CA RINGKASAN PORTOFOLIO (SIMULASI)\n\n"
            f"Nilai sekarang: {_rp(p.get('equity'))}\nKas: {_rp(p.get('cash'))}\n"
            f"Untung/rugi terealisasi: {_rp(p.get('realized_pnl', p.get('pnl')))}\n"
            f"Drawdown: {p.get('drawdown', '-')}\n\nKondisi: {p.get('status', 'normal')}\n"
            f"Ini hasil simulasi, bukan rekening broker."
        )
    if et == "GOVERNOR":
        return (
            f"\U0001F9E0 PENYESUAIAN SISTEM\n\nAPA YANG TERJADI?\nSistem menyesuaikan cara kerja model.\n\n"
            f"Model aktif: {p.get('active_models', '-')}\nAlasan: {p.get('reason', '-')}\n"
            f"Rezim pasar: {p.get('regime', '-')}\n\n"
            f"APA YANG DILAKUKAN SISTEM?\nParameter strategis diperbarui; batas keamanan tetap berlaku."
        )
    if et == "TRAINING":
        return (
            f"\U0001F9EA HASIL PELATIHAN\n\nKandidat: {p.get('candidate_id', p.get('model_version', '-'))}\n"
            f"Hasil: {p.get('status', p.get('result', '-'))}\nAlasan: {p.get('reason', '-')}\n\n"
            f"Model produksi hanya diganti jika semua uji kelayakan lulus."
        )
    if et in ("SYSTEM_ERROR", "HALT", "DEGRADED_MODE"):
        return (
            f"\u26A0\uFE0F SISTEM BERHENTI SEMENTARA\n\n"
            f"APA YANG TERJADI?\n{p.get('message', p.get('reason', 'Kondisi tidak aman terdeteksi.'))}\n\n"
            f"KENAPA?\n{p.get('reason', 'Validasi data/risiko/rekonciliasi gagal.')}\n\n"
            f"APA DAMPAKNYA?\nTidak ada transaksi baru dilakukan.\n\n"
            f"TINDAKAN:\nSistem menunggu kondisi aman."
        )
    if et in ("STOP_LOSS", "TAKE_PROFIT"):
        return (
            f"\U0001F514 {et.replace('_', ' ')}\n\nSaham: {p.get('symbol', '-')}\n"
            f"Harga: {_rp(p.get('price', p.get('entry_price')))}\nPnL: {_rp(p.get('pnl'))}\n\n"
            f"Posisi simulasi ditutup sesuai aturan risiko."
        )
    return f"IDX {et}\n{str(p)[:500]}"
