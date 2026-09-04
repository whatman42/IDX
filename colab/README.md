# IDX Colab GPU Training Center

## Tujuan

Google Colab dipakai sebagai **lingkungan training / research eksternal** dengan GPU.

**Bukan** production server.

| Peran | Sistem |
|-------|--------|
| Production runtime | GitHub Actions |
| Training GPU | Google Colab |
| Model disetujui | Artifact + SHA256 → production load |
| Operational state | Turso / SQLite |
| Penjelasan | Gemini (opsional, advisory) |

## Requirements

- Akun Google + Colab
- (Opsional) GPU runtime
- (Opsional) Google Drive
- Repo: https://github.com/whatman42/idx

## Cara membuka notebook

1. Buka [Google Colab](https://colab.research.google.com)
2. File → Upload notebook → `colab/IDX_GPU_TRAINING.ipynb`
3. Runtime → Change runtime type → **GPU** (jika tersedia)

## Secrets

Jangan hardcode. Colab Secrets / env:

- `GEMINI_API_KEY` — advisory saja
- `TURSO_*` — opsional (training tidak wajib Turso)

## Menjalankan

```bash
cd /content/idx
python -m colab.colab_train --out-dir artifacts/colab_candidates
```

## Status

| Status | Arti |
|--------|------|
| PASS | Sukses |
| FAIL | Gagal; tidak promote |
| BLOCKED | Resource/kredensial tidak ada |

Synthetic → `MARKET PERFORMANCE = UNVERIFIED`. Production pointer **tidak** diubah default.

## Artifact

`artifacts/colab_candidates/<version>/` + `manifest.json` + SHA256.

## GPU tidak tersedia

CPU fallback otomatis. Laporan: `GPU LIVE = BLOCKED`.
