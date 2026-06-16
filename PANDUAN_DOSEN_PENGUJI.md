# 🎓 PANDUAN UTAMA & PENJELASAN LENGKAP SISTEM: SIREN MASTER AI

Dokumentasi komprehensif ini dirancang khusus untuk menjelaskan seluruh aspek sistem **Siren Master AI** secara mendalam. Panduan ini mencakup latar belakang proyek, urgensi penggunaan kecerdasan buatan, spesifikasi perangkat keras (hardware), arsitektur end-to-end, detail pemrosesan sinyal digital (DSP), arsitektur model deep learning (TinyML), proses kuantisasi, hingga logika dasbor.

---

## 📂 1. LATAR BELAKANG, TUJUAN, & URGENSI SISTEM

### Latar Belakang Masalah
Keterlambatan kendaraan darurat (seperti ambulans, pemadam kebakaran, dan mobil polisi) di jalan raya akibat kemacetan lalu lintas merupakan masalah kritis yang dapat mengancam keselamatan jiwa. Di persimpangan jalan, pengemudi sering kali lambat menyadari kehadiran kendaraan darurat karena isolasi suara di dalam kabin mobil modern atau volume musik yang tinggi.

### Mengapa Menggunakan AI (Kecerdasan Buatan)?
Metode konvensional untuk mendeteksi sirine biasanya menggunakan deteksi amplitudo (keras-lemah suara) atau penyaringan frekuensi statis (*bandpass filter*). Namun, metode klasik ini sangat rentan gagal di jalan raya karena:
1. **Kebisingan Jalan Raya (Ambient Noise):** Klakson kendaraan, deru mesin bus/truk, suara angin, dan hujan memiliki amplitudo tinggi yang dapat memicu alarm palsu (*false positive*).
2. **Karakteristik Sirine yang Dinamis:** Frekuensi sirine terus berubah-ubah secara periodik (meliuk naik-turun seperti pola *wail* dan *yelp*), sehingga tidak dapat dideteksi dengan batas frekuensi statis tunggal.

Siren Master AI mengatasi kelemahan ini dengan menerapkan **Jaringan Saraf Tiruan (Convolutional Neural Network - CNN)**. AI dilatih menggunakan ribuan sampel suara untuk mengenali *pola visual* dari spektrogram suara sirine. AI tidak hanya melihat frekuensi sesaat, melainkan mengenali pola perubahan frekuensi dan waktu secara simultan, sehingga sangat kokoh terhadap gangguan kebisingan jalanan biasa.

### Tujuan Proyek
* **Klasifikasi Real-Time:** Mengklasifikasikan suara secara real-time ke dalam 4 kategori: **Ambulance, Firetruck, Police,** dan **Normal**.
* **Implementasi TinyML (Edge AI):** Memasukkan model AI yang kompleks ke dalam mikrokontroler murah dan hemat daya (ESP32) agar sistem dapat dipasang di lampu lalu lintas atau kendaraan tanpa ketergantungan pada koneksi internet atau server eksternal.
* **Akurasi & Latensi Optimal:** Menghasilkan prediksi yang akurat (di atas 90%) dengan delay pengambilan keputusan yang sangat rendah (di bawah 1 detik).

---

## 🎛️ 2. SPESIFIKASI PERANGKAT KERAS (HARDWARE SYSTEM)

Sistem Siren Master AI dirancang untuk berjalan pada dua platform: dasbor laptop (untuk pemantauan visual terperinci) dan mikrokontroler *edge* (untuk implementasi mandiri di lapangan). Berikut adalah rincian komponen hardware yang digunakan:

```
                  +-----------------------------------+
                  |      Suara Sirine (Fisik)         |
                  +-----------------+-----------------+
                                    |
                                    v
                  +-----------------+-----------------+
                  |      Mikrofon (INMP441 / Analog)  |
                  +-----------------+-----------------+
                                    |
                                    v
                  +-----------------+-----------------+
                  |   Mikrokontroler ESP32-WROOM-32E  |
                  +--------+-----------------+--------+
                           |                 |
            +--------------v-------+  +------v---------------+
            |  Tampilan LCD / OLED |  | Indikator Buzzer/LED |
            +----------------------+  +----------------------+
```

### 1. Mikrokontroler: ESP32-WROOM-32E
* **Processor:** Dual-core Xtensa 32-bit LX6 dengan kecepatan clock hingga 240 MHz. Mikrokontroler ini dipilih karena memiliki unit pemrosesan matematika yang cepat (FPU) untuk mempercepat perhitungan FFT dan inferensi AI.
* **Memori RAM:** 520 KB SRAM internal. Keterbatasan memori ini merupakan tantangan utama dalam TinyML, sehingga model AI harus didesain sangat efisien agar muat dalam alokasi RAM dinamis (*Tensor Arena*) sebesar ~85 KB.
* **Memori Flash:** 4 MB Flash untuk menyimpan program (*firmware*) dan bobot biner model AI.

### 2. Sensor Suara (Mikrofon)
* **Mikrofon Digital I2S (INMP441):** Mikrofon omnidirectional berkualitas tinggi yang menghasilkan data audio digital 24-bit langsung melalui bus I2S. Keuntungannya adalah sinyal bebas dari noise interferensi elektromagnetik dan tidak memerlukan proses konversi analog-ke-digital (ADC) eksternal.
* **Mikrofon Analog dengan AGC (MAX9814):** Digunakan sebagai alternatif. Sensor ini dilengkapi dengan *Auto Gain Control* (AGC) untuk memperkuat suara sirine yang jauh tanpa memotong suara sirine dekat (*clipping*). Output analog dibaca melalui pin ADC ESP32 pada frekuensi sampling 8.000 Hz.

### 3. Antarmuka Output (Output Interface)
* **Layar LCD I2C 16x2 / OLED SSD1306:** Menampilkan hasil klasifikasi suara AI secara real-time beserta persentase tingkat kepercayaan (*confidence score*).
* **Buzzer Aktif & LED Indikator:** Memberikan peringatan auditori dan visual instan. LED akan menyala dengan warna berbeda berdasarkan jenis sirine (misalnya: LED Merah untuk Pemadam Kebakaran, LED Biru untuk Polisi, LED Kuning untuk Ambulans, dan LED Hijau untuk Normal).

---

## 🔗 3. ARSITEKTUR SISTEM END-TO-END

Proses deteksi sirine bekerja melalui alur pemrosesan data yang sistematis sebagai berikut:

```
[Suara Sekitar] 
      ➡️ [Pengambilan Sampel Audio (8 kHz, 16-bit, Mono)]
      ➡️ [Pemotongan Frame Temporal (128 sampel/step)]
      ➡️ [Preprocessing DSP (LPF, DC Removal, Hamming Window)]
      ➡️ [Transformasi FFT & Filterbank Mel (40 Koefisien)]
      ➡️ [Normalisasi Z-Score Spektrogram]
      ➡️ [Inferensi Model AI (2D CNN INT8)]
      ➡️ [Logika Keputusan (Temporal Voting / Smart Detect)]
      ➡️ [Output Action (LCD Display, LED, Buzzer, Dasbor Laptop)]
```

---

## 📊 4. DATASET & PROSES PELATIHAN MODEL

Kunci utama dari keandalan AI adalah kualitas data pelatihan. Model Siren Master AI dilatih menggunakan dataset audio yang dikurasi secara ketat.

### Distribusi Data
Dataset terdiri dari sekitar 15.000 sampel suara yang dibagi rata ke dalam 4 kelas untuk menghindari ketimpangan kelas (*class imbalance*):
1. **Ambulance:** Rekaman sirine ambulans dari berbagai negara dengan berbagai tipe modulasi (wail, yelp).
2. **Firetruck:** Rekaman sirine truk pemadam kebakaran yang umumnya didominasi oleh klakson pneumatik bernada rendah (*air horn*) dan sirine mekanis cepat.
3. **Police:** Rekaman sirine polisi patroli dengan pola yelp cepat dan phaser.
4. **Normal:** Rekaman kontrol suara lingkungan nyata, seperti kebisingan lalu lintas kota, suara hujan, angin, mesin kendaraan, klakson pendek, musik radio, dan suara orang berbicara.

### Teknik Augmentasi Data (SpecAugment)
Untuk mencegah model menghafal data latihan (*overfitting*) dan agar kuat saat digunakan pada kondisi mikrofon yang berbeda, diterapkan teknik augmentasi data langsung pada spektrogram:
* **Frequency Masking:** Menghapus pita frekuensi acak pada spektrogram untuk mensimulasikan mikrofon yang kurang sensitif pada frekuensi tertentu.
* **Time Masking:** Menghapus potongan waktu acak pada spektrogram untuk melatih AI mendeteksi sirine meskipun suara sempat terputus-putus atau tertutup suara klakson kendaraan lain.
* Melalui augmentasi ini, jumlah data latihan yang efektif berlipat ganda hingga **78.594 sampel**.

---

## 🎛️ 5. PIPELINE PEMROSESAN SINYAL DIGITAL (DSP)

Sebelum spektrogram suara dimasukkan ke dalam model AI, sinyal suara dari mikrofon harus melalui serangkaian proses pre-processing agar fitur suara dapat diekstraksi secara optimal.

### Detail Parameter DSP
* **Sample Rate ($f_s$):** 8.000 Hz.
* **Durasi Input:** 4.0 detik (Total 32.000 sampel audio per analisis).
* **Frame Size (FFT Size):** 256 sampel (32 ms per frame).
* **Hop Length (Shift):** 128 sampel (16 ms tumpang tindih / *overlap*).
* **Dimensi Output Spektrogram:** 249 frame waktu × 40 koefisien Mel.

### 7 Tahap Pemrosesan Fitur Audio
1. **3-tap Moving Average LPF (Low Pass Filter)**
   Menghaluskan gelombang suara mentah menggunakan rumus:
   $$y[n] = \frac{x[n-1] + x[n] + x[n+1]}{3}$$
   Proses ini meredam kebisingan frekuensi tinggi (*high-frequency hiss*) dari sirkuit mikrofon analog.
2. **DC Offset Removal**
   Menghilangkan pergeseran tegangan searah pada sinyal analog dengan mengurangi rata-rata nilai sinyal dalam satu frame:
   $$x_{clean}[i] = x[i] - \mu_{frame}$$
   Ini memastikan gelombang suara berosilasi sempurna di sekitar titik nol.
3. **Hamming Windowing**
   Mengalikan sinyal frame dengan fungsi Hamming Window untuk memperhalus transisi sinyal di ujung frame. Hal ini mencegah munculnya frekuensi palsu akibat pemotongan sinyal waktu secara mendadak (*spectral leakage*).
4. **Fast Fourier Transform (FFT) & Power Spectrum**
   Mengubah sinyal audio dari domain waktu menjadi domain frekuensi untuk menghitung distribusi energi suara pada setiap frekuensi.
5. **Mel-Filterbank (40 Koefisien Mel)**
   Memetakan rentang frekuensi linier hasil FFT ke skala Mel. Skala Mel meniru cara kerja pendengaran manusia yang lebih sensitif terhadap perbedaan nada rendah dibanding nada tinggi.
6. **Log-Scale**
   Menerapkan fungsi logaritma desibel $\log(\text{energi} + 10^{-9})$ karena telinga manusia menangkap perubahan keras-lemah suara secara logaritmik, bukan linier.
7. **Z-score Normalization**
   Menormalisasi nilai spektrogram Mel akhir menggunakan nilai rata-rata (`MEL_MEAN`) dan deviasi standar (`MEL_STD`) global dari dataset pelatihan:
   $$\text{Output} = \frac{\text{Value} - \text{MEL\_MEAN}}{\text{MEL\_STD}}$$
   Ini memastikan rentang nilai input yang diterima AI selalu konsisten, tidak terpengaruh oleh perbedaan sensitivitas volume mikrofon.

---

## 🕸️ 6. ARSITEKTUR JARINGAN SARAF 2D CNN RINGAN

Model Siren Master AI menggunakan arsitektur **Lightweight 2D Convolutional Neural Network (2D CNN)**. Kami memperlakukan spektrogram Mel berukuran `(249, 40)` sebagai gambar hitam-putih berdimensi dua, di mana sumbu X adalah Waktu dan sumbu Y adalah Frekuensi Mel.

```
Input: (249, 40, 1)
   |
   v
[Conv2D (16 filter, Kernel 3x3, Strides 2x2)]  <-- Mengurangi dimensi spasial 50% untuk hemat RAM
   |
   v
[Batch Normalization & MaxPool (2x2)]          <-- Stabilisasi & reduksi dimensi lanjut
   |
   v
[SeparableConv2D (32 filter, Kernel 3x3)]      <-- Memisahkan operasi spatial & channel (Hemat 70% parameter)
   |
   v
[Batch Normalization & MaxPool (2x2)]
   |
   v
[SeparableConv2D (48 filter, Kernel 3x3)]      <-- Ekstraksi fitur tingkat tinggi
   |
   v
[Global Average Pooling 2D (GAP)]              <-- Mengubah matriks fitur menjadi vektor 48-dimensi (Tanpa Flatten)
   |
   v
[Dense (64 neuron) + Dropout (40%)]            <-- Layer klasifikasi penuh
   |
   v
[Dense (4 neuron, Softmax Output)]             <-- Menghasilkan probabilitas kelas (Ambulance, Fire, Police, Normal)
```

### Mengapa Desain Ini Sangat Efisien?
* **Strided Convolution:** Melakukan reduksi ukuran data sejak awal menggunakan *strides=(2,2)* sehingga memori untuk menyimpan hasil aktivasi (*RAM activation*) menyusut drastis.
* **Depthwise Separable Convolutions:** Mengurangi jumlah perhitungan matematis dan memori bobot secara drastis dibanding konvolusi standar.
* **Global Average Pooling (GAP):** Menggantikan fungsi *Flatten* (yang biasanya menghasilkan ribuan koneksi padat parameter). GAP hanya menghitung rata-rata tiap channel sehingga ukuran file model tetap di bawah 20 KB.
* **Total Parameter:** Hanya sekitar **6.500 parameter** (sangat kecil dibanding model audio standar yang biasanya memiliki ratusan ribu parameter).

---

## ⚡ 7. OPTIMASI & KUANTISASI TINYML (INT8)

Agar model AI dapat ditanam langsung pada chip ESP32, model harus dikonversi dari format TensorFlow Keras (.h5) ke format C++ array menggunakan teknik **Post-Training Quantization (PTQ)**.

### Proses Kuantisasi INT8
1. **Perubahan Tipe Data:** Mengonversi bobot jaringan saraf dan nilai aktivasi yang semula berupa bilangan pecahan desimal presisi tinggi (**Float32**, 32-bit) menjadi bilangan bulat (**Int8**, 8-bit).
2. **Kompak & Cepat:** Proses ini memangkas ukuran biner model sebesar **4 kali lipat** (dari ~80 KB menjadi **19,12 KB**). Operasi matematika bilangan bulat (*integer math*) berjalan jauh lebih cepat pada processor mikrokontroler dibanding pecahan desimal.
3. **Representative Dataset:** Menggunakan subset kecil dari dataset asli saat konversi untuk menentukan rentang pembulatan nilai numerik secara dinamis. Hasilnya, penurunan akurasi akibat kuantisasi sangat minimal (**di bawah 0,1%**).
4. **Binerisasi C++ (`model.h`):** File biner model `.tflite` diterjemahkan menjadi array heksadesimal C++:
   ```cpp
   const unsigned char siren_model_data[] = { 0x1c, 0x00, 0x00, ... };
   const unsigned int siren_model_data_len = 19584;
   ```
   Array ini di-flash ke dalam memori program (*Flash*) ESP32, dan dibaca oleh pustaka **TensorFlow Lite for Microcontrollers**.

---

## 🛡️ 8. LOGIKA OPERASIONAL DASBOR & PENGUJIAN

Dasbor laptop dirancang agar interaktif, responsif, dan stabil menghadapi gangguan di lapangan.

### Logika Penyaringan Gangguan (Smart Detect)
Di jalan raya, suara klakson mobil, batuk manusia, atau ketukan pintu dapat memiliki kemiripan spektral sesaat dengan sirine, yang dapat memicu deteksi salah (*false trigger*).
* Dasbor mengevaluasi suara per 100 ms (frekuensi pengujian 10 Hz).
* Sistem menerapkan **Temporal Voting**: AI harus memprediksi kategori sirine yang sama (misalnya *Police*) secara konsisten selama **6 frame berturut-turut** (~0,6 detik) sebelum dasbor resmi menyatakan bahwa sirine terdeteksi. Ini menyaring gangguan transient berdurasi pendek.

### Fitur Penahan Prediksi (Prediction Hold / Freeze State)
Saat sirine terdeteksi, dasbor akan membekukan visualisasi layar (*Frozen State*) dan mengunci hasil prediksi beserta probabilitasnya.
* **Kemudahan Evaluasi:** Memberikan waktu yang cukup bagi dosen penguji untuk membaca hasil tebakan AI dengan tenang, bahkan setelah suara sirine dimatikan.
* **Verifikasi Manual:** Pengguna dapat menekan tombol **`Y`** (jika prediksi benar) atau tombol **`N`** / **`T`** (jika prediksi salah) pada keyboard. Jawaban ini dicatat langsung oleh sistem ke dalam tabel statistik performa sebelum dasbor mencair kembali untuk mendengarkan (*listening*).

### Rendering Bebas Kedipan (Flicker-Free CMD)
Dasbor menggunakan ANSI escape sequence `\033[H` untuk mengembalikan posisi kursor penulisan langsung ke pojok kiri atas layar (menimpa teks lama secara real-time) alih-alih menggunakan perintah pembersihan layar total `os.system("cls")`. Hasilnya, visualisasi dasbor berjalan sangat mulus tanpa ada efek berkedip (*flicker-free*) pada layar Command Prompt (CMD).

---

## 💡 9. PERTANYAAN TEKNIS KRITIS DOSEN & STRATEGI MENJAWAB

Berikut adalah beberapa pertanyaan sulit yang sering diajukan oleh dosen penguji beserta panduan cara menjawabnya dengan percaya diri:

### 💬 Pertanyaan 1: "Mengapa kamu menggunakan Sample Rate 8.000 Hz? Bukankah audio standar itu 16.000 Hz atau 44.100 Hz?"
> **Jawaban:** 
> "Kami menggunakan sample rate 8.000 Hz karena rentang frekuensi dominan sirine kendaraan darurat berada di bawah 4.000 Hz. Berdasarkan Teorema Nyquist, sample rate 8 kHz sudah cukup untuk merekonstruksi sinyal hingga frekuensi 4.000 Hz tanpa kehilangan informasi penting (*aliasing*). Menggunakan sample rate lebih tinggi seperti 16.000 Hz hanya akan melipatgandakan ukuran data spektrogram dan membuang RAM mikrofon/ESP32 secara percuma tanpa meningkatkan performa klasifikasi."

### 💬 Pertanyaan 2: "Bagaimana model AI sekecil 19 KB (6.500 parameter) bisa memiliki akurasi tinggi? Bukankah biasanya CNN butuh jutaan parameter?"
> **Jawaban:** 
> "Akurasi tinggi dicapai karena kami menyederhanakan tugas model. Model hanya perlu mengklasifikasikan 4 kategori suara spesifik yang telah diproses menjadi representasi spektrogram Mel 2D yang padat informasi fitur. Selain itu, kami menerapkan teknik optimasi modern seperti *Depthwise Separable Convolutions* yang memisahkan pemfilteran spasial dan channel untuk memangkas parameter hingga 80%, serta *Global Average Pooling* untuk menggantikan layer *Flatten* yang boros parameter. Terakhir, proses kuantisasi INT8 memangkas ukuran biner model hingga 4x lipat tanpa degradasi performa."

### 💬 Pertanyaan 3: "Bagaimana cara sistem membedakan suara sirine ambulans asli dengan suara musik lagu di jalan raya?"
> **Jawaban:** 
> "Kami melatih model menggunakan kelas kontrol **NORMAL** yang berisi berbagai kebisingan jalan raya, termasuk musik radio, klakson pendek, suara mesin, dan percakapan. Selain itu, kami menerapkan sistem penyaringan keputusan **Smart Detect (Voting Temporal)**. Suara musik atau klakson acak mungkin hanya memicu prediksi sirine selama 1 atau 2 frame (100-200 ms). Karena sistem mensyaratkan deteksi konsisten selama minimal 6 frame berturut-turut (600 ms), gangguan suara acak tersebut akan langsung tereliminasi dan diabaikan."
