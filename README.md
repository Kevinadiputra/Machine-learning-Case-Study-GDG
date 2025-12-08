# Lung Segmentation App

**Nama: Kevin Adiputra**

Proyek ini merupakan implementasi Computer Vision untuk segmentasi medis otomatis pada citra X-Ray dada. Menggunakan algoritma YOLOv8-seg, aplikasi ini dirancang untuk mengenali dan memisahkan area paru-paru (kiri dan kanan), tulang belakang (spinal cord), dan tubuh dengan presisi tinggi.

Proyek ini dikembangkan sebagai bagian dari studi kasus Machine Learning untuk Google Developer Groups (GDG).

## Demo Aplikasi

Aplikasi ini telah di-deploy dan dapat digunakan secara langsung melalui tautan berikut:

**https://machine-learning-case-study-gdg-hapkfgwvzrsh5sx2pmqnip.streamlit.app/**

---

## Fitur Utama

* **Segmentasi Real-time**: Menggunakan model YOLOv8n-seg (Nano) yang efisien untuk inferensi cepat.
* **Deteksi Multi-Kelas**: Mampu melakukan segmentasi pada 4 kelas objek:
    * Body (Tubuh)
    * Cord (Tulang Belakang)
    * Right Lung (Paru Kanan)
    * Left Lung (Paru Kiri)
* **Antarmuka Web**: Dibangun menggunakan Streamlit untuk memudahkan pengunggahan gambar dan visualisasi hasil segmentasi.
* **Metrik Evaluasi**: Menyediakan informasi statistik mengenai akurasi deteksi pada setiap gambar.

## Performa Model

Berdasarkan hasil evaluasi pada dataset validasi, model ini menunjukkan kinerja sebagai berikut:

* **Arsitektur Model**: YOLOv8n-seg
* **mAP@50 (Mask)**: 93.9%
* **mAP@50-95 (Mask)**: 82.7%
* **Kecepatan Inferensi**: ~3.8ms per gambar (GPU T4)

**Rincian Performa per Kelas:**

| Kelas | Precision | Recall | mAP50 |
|-------|-----------|--------|-------|
| All | 0.92 | 0.89 | 0.94 |
| Body | 0.97 | 1.00 | 0.99 |
| Paru Kanan | 0.95 | 0.90 | 0.92 |
| Paru Kiri | 0.83 | 0.90 | 0.93 |

## Instalasi dan Penggunaan Lokal

Ikuti langkah berikut untuk menjalankan aplikasi di komputer lokal:

1.  **Clone Repository**
    ```bash
    git clone [https://github.com/Kevinadiputra/Machine-learning-Case-Study-GDG.git](https://github.com/Kevinadiputra/Machine-learning-Case-Study-GDG.git)
    cd Machine-learning-Case-Study-GDG/lung-segmentation-app
    ```

2.  **Instalasi Dependensi**
    Disarankan menggunakan Python 3.9 atau versi yang lebih baru.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Menjalankan Aplikasi**
    ```bash
    streamlit run src/app.py
    ```
    Aplikasi akan berjalan di `http://localhost:8501`.

## Struktur Direktori

* `src/`: Berisi kode sumber utama aplikasi (app.py) dan logika inferensi.
* `models/`: Direktori penyimpanan bobot model (best.pt).
* `assets/`: Direktori untuk gambar sampel atau aset statis lainnya.
* `requirements.txt`: Daftar pustaka Python yang diperlukan.

## Penafian (Disclaimer)

Aplikasi ini dikembangkan semata-mata untuk tujuan penelitian dan edukasi dalam bidang Machine Learning. Hasil prediksi model ini tidak ditujukan sebagai pengganti diagnosis medis profesional. Penggunaan untuk keputusan klinis harus melalui verifikasi tenaga medis ahli.

---