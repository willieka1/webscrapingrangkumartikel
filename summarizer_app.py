# research_agent.py
# Groq API + Tavily MCP — output rapi + ekstraksi data angka
#
# Setup di Google Colab:
# 1. Klik ikon kunci (🔑) di sidebar kiri → "Add new secret"
# 2. GROQ_API_KEY   → isi API key Groq kamu
# 3. TAVILY_API_KEY → isi API key Tavily kamu
# 4. Aktifkan toggle "Notebook access" untuk keduanya

import os
import re
import json
import datetime
import requests
from openai import OpenAI
from google.colab import userdata

# ─────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────

GROQ_API_KEY = userdata.get("GROQ_API_KEY")
TAVILY_API_KEY = userdata.get("TAVILY_API_KEY")

if not GROQ_API_KEY or not TAVILY_API_KEY:
    raise EnvironmentError(
        "API key tidak ditemukan.\n"
        "Pastikan kamu sudah menambahkan GROQ_API_KEY dan TAVILY_API_KEY "
        "di Colab Secrets dan mengaktifkan toggle 'Notebook access'."
    )

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY
)

tavily_tools = [
    {
        "type": "mcp",
        "server_url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={TAVILY_API_KEY}",
        "server_label": "tavily",
        "require_approval": "never"
    }
]

MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


# ─────────────────────────────────────────
# TAVILY — PENCARIAN & EKSTRAK LANGSUNG
# ─────────────────────────────────────────

def tavily_search(query: str, max_results: int = 8) -> list[dict]:
    """
    Cari artikel via Tavily Search API.
    Kembalikan list of { title, url, content }.
    """
    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": TAVILY_API_KEY,
                "query": query,
                "search_depth": "advanced",
                "max_results": max_results,
                "include_answer": False,
                "include_raw_content": False,
            },
            timeout=30
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("results", [])
    except Exception as e:
        print(f"  ✗ Tavily search error: {e}")
        return []

def tavily_extract(urls: list[str]) -> list[dict]:
    """
    Ambil konten penuh dari beberapa URL via Tavily Extract API.
    Kembalikan list of { url, raw_content }.
    """
    if not urls:
        return []
    try:
        resp = requests.post(
            "https://api.tavily.com/extract",
            json={
                "api_key": TAVILY_API_KEY,
                "urls": urls[:3],  # maksimal 3 URL sekaligus
            },
            timeout=30
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("results", [])
    except Exception as e:
        print(f"  ✗ Tavily extract error: {e}")
        return []

def kumpulkan_konteks(topik: str) -> tuple[str, list[str]]:
    """
    Jalankan Tavily search + extract, kembalikan (konteks_teks, daftar_url_valid).
    """
    print("  📡  Mengambil data dari internet via Tavily...")

    hasil_search = tavily_search(topik)
    if not hasil_search:
        print("  ⚠  Tidak ada hasil pencarian dari Tavily.")
        return "", []

    url_list  = [r["url"] for r in hasil_search if r.get("url")]
    url_valid = url_list[:5]

    print(f"  ✔  {len(hasil_search)} artikel ditemukan, mengambil konten dari {len(url_valid)} URL...")

    hasil_extract = tavily_extract(url_valid)

    bagian = []
    bagian.append("=== HASIL PENCARIAN ===")
    for i, r in enumerate(hasil_search, 1):
        judul   = r.get("title", "")
        url     = r.get("url", "")
        snippet = r.get("content", "")
        bagian.append(f"\n[{i}] {judul}\nURL: {url}\n{snippet}")

    if hasil_extract:
        bagian.append("\n\n=== KONTEN LENGKAP ARTIKEL ===")
        for r in hasil_extract:
            url    = r.get("url", "")
            konten = r.get("raw_content", "")[:3000]
            bagian.append(f"\nURL: {url}\n{konten}")

    konteks = "\n".join(bagian)
    return konteks, url_list

def kumpulkan_konteks_url(url: str) -> tuple[str, list[str]]:
    """Khusus mode analisis URL — extract 1 URL + search topik terkait."""
    print("  📡  Mengambil konten dari URL...")

    hasil_extract = tavily_extract([url])
    konten_utama  = ""
    if hasil_extract:
        konten_utama = hasil_extract[0].get("raw_content", "")[:4000]

    topik_dari_url = url.split("/")[-1].replace("-", " ")[:60]
    hasil_search   = tavily_search(topik_dari_url, max_results=5)
    url_tambahan   = [r["url"] for r in hasil_search if r.get("url") and r["url"] != url]

    bagian = [f"=== KONTEN UTAMA ({url}) ===\n{konten_utama}"]
    if hasil_search:
        bagian.append("\n=== ARTIKEL TERKAIT ===")
        for r in hasil_search:
            bagian.append(f"\n[{r.get('title','')}]\nURL: {r.get('url','')}\n{r.get('content','')}")

    konteks   = "\n".join(bagian)
    semua_url = [url] + url_tambahan
    return konteks, semua_url


# ─────────────────────────────────────────
# FORMATTING HELPER
# ─────────────────────────────────────────

LINE  = "─" * 70
DLINE = "═" * 70

def cetak_header(judul: str):
    print(f"\n{DLINE}")
    print(f"  {judul.upper()}")
    print(DLINE)

def justify_line(line: str, width: int) -> str:
    """
    Justifikasi satu baris teks (rata kanan-kiri).
    Jika hanya satu kata atau terlalu pendek, kembalikan apa adanya.
    """
    words = line.split()
    if len(words) <= 1:
        return line
    total_spaces = width - sum(len(w) for w in words)
    if total_spaces <= 0:
        return line
    gaps             = len(words) - 1
    base, extra      = divmod(total_spaces, gaps)
    result = ""
    for i, word in enumerate(words[:-1]):
        result += word + " " * (base + (1 if i < extra else 0))
    result += words[-1]
    return result

def wrap_teks(teks: str, lebar: int = 66, indent: str = "│  ") -> str:
    """
    Wrap teks panjang dengan justifikasi kanan-kiri (rata kedua sisi).
    Baris terakhir tiap paragraf tetap rata kiri (standar tipografi).
    Baris kosong dipertahankan sebagai pemisah paragraf.
    """
    hasil = []
    for baris in teks.strip().splitlines():
        b = baris.strip()
        if not b:
            hasil.append(indent.rstrip())
            continue
        segmen = []
        while len(b) > lebar:
            potong = b.rfind(" ", 0, lebar)
            if potong == -1:
                potong = lebar
            segmen.append(b[:potong])
            b = b[potong:].lstrip()
        segmen.append(b)  # sisa (baris terakhir paragraf)

        for idx, seg in enumerate(segmen):
            # justify semua baris KECUALI yang terakhir
            if idx < len(segmen) - 1:
                hasil.append(f"{indent}{justify_line(seg, lebar)}")
            else:
                hasil.append(f"{indent}{seg}")
    return "\n".join(hasil)

def cetak_section(judul: str, isi: str):
    """Cetak satu section dengan border rapi dan teks justified."""
    batas = f"{'─' * (55 - min(len(judul), 54))}"
    print(f"\n┌─ {judul} {batas}")
    print(wrap_teks(isi))
    print(f"└{LINE}")

def parse_sections(teks: str) -> dict:
    """
    Pecah teks markdown (## JUDUL ... ## JUDUL berikutnya)
    menjadi dict { 'JUDUL': 'isi' }.
    """
    pola    = re.compile(r"^##\s+(.+)$", re.MULTILINE)
    matches = list(pola.finditer(teks))
    sections = {}
    for i, m in enumerate(matches):
        judul = m.group(1).strip()
        mulai = m.end()
        akhir = matches[i + 1].start() if i + 1 < len(matches) else len(teks)
        sections[judul] = teks[mulai:akhir].strip()
    return sections

def format_poin(teks: str) -> str:
    """
    Pastikan setiap baris poin diindentasi rapi.
    Mendukung format '- ...', '* ...', '1. ...', atau baris biasa.
    """
    baris_baru = []
    for baris in teks.splitlines():
        b = baris.strip()
        if not b:
            continue
        if re.match(r"^[-*•]\s+", b):
            b = "  • " + re.sub(r"^[-*•]\s+", "", b)
        elif re.match(r"^\d+\.\s+", b):
            b = "  " + b
        else:
            b = "    " + b
        baris_baru.append(b)
    return "\n".join(baris_baru)

def tampilkan_hasil(teks: str):
    """Cetak hasil dengan formatting rapi per section."""
    sections = parse_sections(teks)

    urutan = [
        "TOPIK", "RINGKASAN", "POIN PENTING",
        "DATA DAN FAKTA", "DATA NUMERIK",
        "SUMBER", "ANALISIS", "KESIMPULAN"
    ]

    for key in urutan:
        if key in sections:
            isi = sections[key]
            if key in ("POIN PENTING", "DATA DAN FAKTA"):
                isi = format_poin(isi)
            cetak_section(key, isi)

    for key, isi in sections.items():
        if key not in urutan:
            cetak_section(key, isi)


# ─────────────────────────────────────────
# EKSTRAKSI DATA ANGKA
# ─────────────────────────────────────────

def ekstrak_angka(teks: str) -> list[dict]:
    """
    Cari semua data numerik di dalam teks dan kembalikan sebagai list dict.
    """
    pola = re.compile(
        r"""
        (?:
            Rp\.?\s*|USD\s*|\$|€|£
        )?
        \d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?
        (?:
            \s*(?:%|persen|juta|miliar|triliun|ribu|
                  km|kg|ton|liter|MW|GW|°C|°F|
                  meter|hektar|ha)\b
        )?
        """,
        re.VERBOSE | re.IGNORECASE
    )

    hasil = []
    for m in pola.finditer(teks):
        nilai = m.group().strip()
        if not nilai or re.fullmatch(r"\d{1,2}", nilai):
            continue
        mulai   = max(0, m.start() - 60)
        akhir   = min(len(teks), m.end() + 60)
        konteks = teks[mulai:akhir].replace("\n", " ").strip()
        if not any(d["nilai"] == nilai for d in hasil):
            hasil.append({"nilai": nilai, "konteks": f"...{konteks}..."})

    return hasil

def tampilkan_data_angka(data: list[dict]):
    if not data:
        print("\n  (Tidak ada data numerik yang ditemukan)")
        return

    cetak_header(f"DATA NUMERIK — {len(data)} item ditemukan")
    for i, item in enumerate(data, 1):
        print(f"\n  [{i:02d}]  Nilai   : {item['nilai']}")
        konteks_wrap = wrap_teks(item['konteks'], lebar=60, indent="          ")
        print(f"        Konteks :\n{konteks_wrap}")

def ekspor_angka_json(data: list[dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  → Data angka diekspor ke: {path}")


# ─────────────────────────────────────────
# HITUNG KATA
# ─────────────────────────────────────────

TARGET_KATA = 1000

def hitung_kata(teks: str) -> int:
    """
    Hitung kata bermakna saja — abaikan angka, simbol, dan marker markdown.
    Cocok untuk bahasa Indonesia/Inggris.
    """
    # hapus baris header markdown (## ...) agar tidak terhitung
    teks_bersih = re.sub(r"^##.*$", "", teks, flags=re.MULTILINE)
    # ambil hanya token kata (min 2 huruf, latin/extended latin)
    kata = re.findall(r"\b[a-zA-Z\u00C0-\u024F]{2,}\b", teks_bersih)
    return len(kata)

def status_kata(jumlah: int, target: int = TARGET_KATA) -> str:
    if jumlah >= target:
        return f"✔  memenuhi target ({target:,} kata)"
    kekurangan = target - jumlah
    return f"⚠  kurang {kekurangan:,} kata dari target {target:,}"


# ─────────────────────────────────────────
# FUNGSI UTAMA: CARI ARTIKEL
# ─────────────────────────────────────────

PROMPT_RISET = """
Berikut adalah data riset yang sudah dikumpulkan dari internet tentang: {topik}

{konteks}

---
Berdasarkan data di atas, tulis laporan lengkap dengan format markdown ini:

## TOPIK
[judul topik]

## RINGKASAN
[ringkasan minimal 200 kata, detail dan mengalir, berdasarkan data di atas]

## POIN PENTING
- [poin 1, jelaskan 2–3 kalimat]
- [poin 2, jelaskan 2–3 kalimat]
(minimal 8 poin)

## DATA DAN FAKTA
- [setiap angka, nama, tanggal, statistik dari data di atas — satu baris satu fakta]
(minimal 10 fakta)

## SUMBER
{url_list}

## ANALISIS
[analisis mendalam minimal 150 kata]

## KESIMPULAN
[kesimpulan minimal 100 kata]

Penting:
- Jawab dalam bahasa Indonesia, total minimal 1000 kata.
- Jangan mengarang fakta — hanya gunakan informasi dari data yang diberikan.
- Pastikan setiap angka/statistik ditulis dengan jelas beserta satuannya.
"""

PROMPT_URL = """
Berikut adalah konten yang diambil dari URL: {url}

{konteks}

---
Berdasarkan konten di atas, tulis laporan lengkap dengan format markdown ini:

## TOPIK
[judul artikel atau topik utama]

## RINGKASAN
[ringkasan minimal 200 kata]

## POIN PENTING
- [poin 1, 2–3 kalimat]
- [poin 2, 2–3 kalimat]
(minimal 8 poin)

## DATA DAN FAKTA
- [setiap angka, nama, tanggal, statistik — satu baris satu fakta]
(minimal 10 fakta)

## SUMBER
{url_list}

## ANALISIS
[analisis mendalam minimal 150 kata]

## KESIMPULAN
[kesimpulan minimal 100 kata]

Penting:
- Jawab dalam bahasa Indonesia, total minimal 1000 kata.
- Jangan mengarang fakta — hanya gunakan informasi dari konten yang diberikan.
"""

SYSTEM_PROMPT = (
    "Kamu asisten riset profesional. "
    "Tulis laporan berdasarkan data yang diberikan saja — jangan mengarang. "
    "Selalu berikan jawaban panjang, detail, dan terstruktur dalam bahasa Indonesia. "
    "Setiap angka dan statistik harus disebutkan dengan jelas beserta satuan dan konteksnya. "
    "Gunakan format markdown ## JUDUL untuk setiap section."
)

def panggil_model(prompt: str) -> str | None:
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt}
            ],
            temperature=0.3,
            max_tokens=4096,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"\n  ✗ Error saat memanggil model: {e}")
        return None

def cari_artikel(topik: str) -> str | None:
    print(f"\n  🔍  Mencari informasi tentang: {topik}")

    konteks, url_list = kumpulkan_konteks(topik)
    if not konteks:
        print("  ⚠  Tidak ada data dari Tavily, hasil mungkin tidak akurat.")

    url_formatted = "\n".join(f"- {u}" for u in url_list) if url_list else "- (tidak ada)"

    print("  🤖  Menyusun laporan...\n")
    return panggil_model(
        PROMPT_RISET.format(topik=topik, konteks=konteks, url_list=url_formatted)
    )

def analisis_url(url: str) -> str | None:
    print(f"\n  🔗  Menganalisis URL: {url}")

    konteks, url_list = kumpulkan_konteks_url(url)
    if not konteks:
        print("  ⚠  Konten tidak berhasil diambil.")

    url_formatted = "\n".join(f"- {u}" for u in url_list) if url_list else f"- {url}"

    print("  🤖  Menyusun laporan...\n")
    return panggil_model(
        PROMPT_URL.format(url=url, konteks=konteks, url_list=url_formatted)
    )


# ─────────────────────────────────────────
# SIMPAN HASIL
# ─────────────────────────────────────────

def simpan_hasil(hasil: str, nama_file: str, data_angka: list[dict] | None = None):
    os.makedirs("hasil", exist_ok=True)
    waktu = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base  = f"hasil/{nama_file}_{waktu}"

    path_txt = f"{base}.txt"
    with open(path_txt, "w", encoding="utf-8") as f:
        f.write(hasil)
    print(f"\n  ✔  Teks disimpan  : {path_txt}")

    if data_angka:
        path_json = f"{base}_angka.json"
        ekspor_angka_json(data_angka, path_json)


# ─────────────────────────────────────────
# PROGRAM UTAMA
# ─────────────────────────────────────────

def main():
    cetak_header("RESEARCH AGENT  —  Groq + Tavily")
    print("\n  Mode yang tersedia:")
    print("  1  →  Cari berdasarkan topik")
    print("  2  →  Analisis dari URL")
    print(f"\n{DLINE}")

    while True:
        pilihan = input("\nPilih [1 / 2] atau ketik 'exit': ").strip().lower()

        if pilihan == "exit":
            print("\n  Sampai jumpa! 👋\n")
            break

        elif pilihan == "1":
            topik = input("  Masukkan topik : ").strip()
            if not topik:
                print("  ✗ Topik tidak boleh kosong."); continue
            hasil = cari_artikel(topik)

        elif pilihan == "2":
            url = input("  Masukkan URL   : ").strip()
            if not url:
                print("  ✗ URL tidak boleh kosong."); continue
            hasil = analisis_url(url)
            topik = "url_analisis"

        else:
            print("  ✗ Pilihan tidak valid."); continue

        if not hasil:
            print("  ✗ Tidak ada hasil yang diterima."); continue

        # tampilkan hasil terformat
        tampilkan_hasil(hasil)

        # ekstrak & tampilkan data angka
        data_angka = ekstrak_angka(hasil)
        tampilkan_data_angka(data_angka)

        # ── statistik ──────────────────────────────
        jumlah_kata = hitung_kata(hasil)
        print(f"\n{LINE}")
        print(f"  Total kata      : {jumlah_kata:,} kata")
        print(f"  Status          : {status_kata(jumlah_kata)}")
        print(f"  Data numerik    : {len(data_angka)} item")
        print(LINE)

        # tawaran simpan
        simpan = input("\n  Simpan hasil? (y/n): ").strip().lower()
        if simpan == "y":
            nama = topik[:30].replace(" ", "_")
            simpan_hasil(hasil, nama, data_angka)


if __name__ == "__main__":
    main()
