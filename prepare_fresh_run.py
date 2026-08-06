import shutil
from pathlib import Path

def prepare_fresh_environment():
    print("🧹 [1/2] Membersihkan cache data pasar...")
    
    # 1. Hapus folder-folder cache
    cache_dirs = ["cache", "data/cache", ".cache"]
    for c_dir in cache_dirs:
        dir_path = Path(c_dir)
        if dir_path.exists() and dir_path.is_dir():
            shutil.rmtree(dir_path)
            print(f"   ✔ Folder dihapus: {c_dir}/")

    # 2. Hapus file cache berformat parquet/sqlite di root/subfolder
    for pattern in ["*cache*.parquet", "*cache*.sqlite", "*cache*.db"]:
        for file_path in Path(".").glob(pattern):
            try:
                file_path.unlink()
                print(f"   ✔ File dihapus: {file_path.name}")
            except Exception as e:
                print(f"   ⚠️ Gagal menghapus {file_path.name}: {e}")

    print("\n📄 [2/2] Memeriksa & membuat file log prediksi...")
    log_file = Path("prediksi_idx_log.csv")
    if not log_file.exists():
        # Tulis header standar CSV
        log_file.write_text("timestamp,ticker,prediction_probability,prediction_confidence,signal_status\n", encoding="utf-8")
        print("   ✅ Berkas 'prediksi_idx_log.csv' berhasil dibuat.")
    else:
        print("   ℹ️ Berkas 'prediksi_idx_log.csv' sudah ada.")

    print("\n🎉 Lingkungan bersih! Data fresh akan disedot pada eksekusi berikutnya.")

if __name__ == "__main__":
    prepare_fresh_environment()
