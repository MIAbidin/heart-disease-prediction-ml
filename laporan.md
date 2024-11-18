# Laporan Proyek Machine Learning - Muhammad Irfan Abidin

## Daftar Isi

- [Domain Proyek](#domain-proyek)
- [Business Understanding](#business-understanding)
- [Data Understanding](#data-understanding)
- [Data Preparation](#data-preparation)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Referensi](#referensi)

## Domain Proyek

![picture 3](https://i.imgur.com/kqSOLBt.jpeg)  


Penyakit  jantung  adalah  sebuah  kondisi  yang  menyebabkan  jantung  tidak  dapat  melaksanakan tugasnya dengan baik. Hal ini disebabkan matinya sebagian otot jantung yang disebabkan karena penyempitan arteri koroner. .<sup>[[1]](https://doi.org/10.56705/IJODAS.V3I2.35)</sup>

Penyakit jantung merupakan salah satu penyebab utama kematian di seluruh dunia. Menurut laporan dari World Health Organization (WHO), penyakit kardiovaskular bertanggung jawab atas sekitar 17,9 juta kematian setiap tahunnya. Di tengah meningkatnya prevalensi faktor risiko seperti hipertensi, diabetes, merokok, dan obesitas, deteksi dini penyakit jantung menjadi sangat penting untuk mengurangi angka kematian.<sup>[[2]](https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds))</sup>

Teknologi machine learning memberikan peluang untuk membantu mendeteksi penyakit jantung secara lebih cepat dan akurat melalui analisis data pasien. Dengan model prediktif yang tepat, diagnosis awal dapat ditingkatkan, sehingga memungkinkan tindakan pencegahan yang lebih efektif.


## Business Understanding

### Problem Statements

- Bagaimana memanfaatkan data pasien untuk memprediksi risiko penyakit jantung?
- Algoritma machine learning mana yang paling efektif untuk prediksi penyakit jantung berdasarkan fitur data klinis?

### Goals

- Membangun model prediktif yang dapat mengklasifikasikan apakah seorang pasien memiliki risiko penyakit jantung.
- Mencapai akurasi di atas 85% pada data uji untuk membantu tenaga medis dalam keputusan klinis.

### Solution statements


#### 1. Menggunakan Beberapa Algoritma Machine Learning untuk Memprediksi Penyakit Jantung
Beberapa algoritma machine learning akan diuji untuk menentukan pendekatan terbaik dalam memprediksi penyakit jantung. Algoritma yang digunakan meliputi:  
- **Logistic Regression**: Digunakan sebagai model dasar untuk memahami hubungan linier antara fitur dan target.  
- **K-Nearest Neighbors (KNN)**: Mengidentifikasi pola berdasarkan kedekatan fitur dengan titik data lain.  
- **Random Forest**: Menggunakan teknik ensemble learning untuk meningkatkan akurasi dan mengurangi overfitting.  
- **AdaBoost**: Meningkatkan kinerja model dasar dengan memberi bobot lebih pada data yang sulit diprediksi.  
- **Gradient Boosting**: Memperbaiki akurasi dengan memfokuskan pada kesalahan model sebelumnya.  

#### 2. Evaluasi Berdasarkan Metrik yang Relevan
Setiap model akan dievaluasi menggunakan metrik berikut:  
- **Akurasi**: Mengukur persentase prediksi yang benar.  
- **Precision**: Menilai seberapa baik model dalam mengidentifikasi kasus positif secara akurat.  
- **Recall**: Mengukur kemampuan model dalam menangkap semua kasus positif.  
- **F1 Score**: Kombinasi dari precision dan recall untuk memberikan gambaran menyeluruh tentang kinerja model.  

#### 3. Memilih Model Terbaik Berdasarkan Hasil Evaluasi
Model terbaik akan dipilih berdasarkan metrik evaluasi, termasuk akurasi, precision, recall, dan F1 score tertinggi. Model terpilih diharapkan mampu memberikan prediksi yang lebih akurat dan dapat diandalkan dalam mendiagnosis penyakit jantung.


## Data Understanding

### Informasi Dataset

| Jenis          | Keterangan                                                                 |
|----------------|----------------------------------------------------------------------------|
| **Title**      | Heart Disease Prediction                                                   |
| **Source**     | [Kaggle](https://www.kaggle.com/datasets/rashadrmammadov/heart-disease-prediction) |
| **License**    | CC0: Public Domain                                                        |
| **Usability**  | 10.00                                                                     |
| **Jumlah Data**| 1,000 sampel, 16 fitur                                                    |

### Deskripsi Variabel

| Nama Fitur                | Tipe          | Deskripsi                                                                  |
|---------------------------|---------------|----------------------------------------------------------------------------|
| **Age**                  | Numerik       | Usia pasien dalam tahun                                                    |
| **Gender**               | Kategorikal   | Jenis kelamin (`Male`/`Female`)                                            |
| **Cholesterol**          | Numerik       | Level kolesterol (mg/dL)                                                   |
| **Blood Pressure**       | Numerik       | Tekanan darah sistolik (mmHg)                                              |
| **Heart Rate**           | Numerik       | Detak jantung (bpm)                                                        |
| **Smoking**              | Kategorikal   | Status merokok (`Never`/`Former`/`Current`)                                |
| **Alcohol Intake**       | Kategorikal   | Konsumsi alkohol (`None`/`Moderate`/`Heavy`)                               |
| **Exercise Hours**       | Numerik       | Jumlah jam olahraga per minggu                                             |
| **Family History**       | Kategorikal   | Riwayat penyakit jantung dalam keluarga (`Yes`/`No`)                       |
| **Diabetes**             | Kategorikal   | Status diabetes (`Yes`/`No`)                                               |
| **Obesity**              | Kategorikal   | Status obesitas (`Yes`/`No`)                                               |
| **Stress Level**         | Numerik       | Tingkat stres (skala 1-10)                                                 |
| **Blood Sugar**          | Numerik       | Kadar gula darah (mg/dL)                                                   |
| **Exercise Induced Angina** | Kategorikal | Angina selama latihan (`Yes`/`No`)                                         |
| **Chest Pain Type**      | Kategorikal   | Jenis nyeri dada (`Typical Angina`/`Atypical Angina`/`Non-anginal Pain`/`Asymptomatic`) |
| **Heart Disease**        | Numerik (Target) | 1 = Memiliki penyakit jantung, 0 = Tidak                                  |

---

### Exploratory Data Analysis (EDA)

#### 1. Analisis Univariate
![Univariate Analysis](https://i.imgur.com/hvTkSbE.png)

- Analisis univariate menunjukkan bahwa fitur seperti **Age**, **Cholesterol**, dan **Blood Pressure** berpotensi memiliki hubungan dengan penyakit jantung.
- Distribusi target memperlihatkan ketidakseimbangan data, di mana lebih banyak pasien yang tidak memiliki penyakit jantung (nilai `0`). Oleh karena itu, diperlukan teknik penyeimbangan data seperti oversampling atau undersampling untuk meningkatkan akurasi model pada kelas minoritas (nilai `1`).

#### 2. Analisis Multivariate
![Multivariate Analysis](https://i.imgur.com/F9NzCmc.png)

- Beberapa fitur kategorikal seperti **Gender**, **Alcohol Intake**, **Family History**, **Diabetes**, **Obesity**, dan **Exercise Induced Angina** menunjukkan pengaruh yang relatif rendah terhadap penyakit jantung, karena distribusi risiko antar kategori cenderung serupa.
- **Smoking** memiliki pengaruh sedang terhadap risiko penyakit jantung, di mana perokok aktif lebih rentan terkena penyakit jantung.
- **Chest Pain Type** merupakan fitur yang paling signifikan, dengan variasi risiko yang jelas antar kategori nyeri dada terhadap penyakit jantung.

#### Kesimpulan EDA
- Fitur kategorikal seperti **Smoking** dan **Chest Pain Type** lebih relevan untuk prediksi penyakit jantung dibandingkan fitur kategorikal lainnya.
- Ketidakseimbangan pada target (`Heart Disease`) perlu diatasi untuk memastikan performa model yang lebih baik pada kelas minoritas.


## Data Preparation

Pada bagian ini, menjelaskan tahap-tahap yang dilakukan untuk mempersiapkan data sebelum digunakan untuk pelatihan model. Data preparation sangat penting karena data yang bersih dan terstruktur akan meningkatkan kinerja model machine learning. Berikut adalah tahapan yang dilakukan dalam proyek ini:

### 1. Menangani Nilai yang Hilang (Missing Values)  
Pertama, dilakukan pemeriksaan terhadap data untuk mendeteksi nilai yang hilang menggunakan `isnull().sum()`.
Jika ditemukan nilai yang hilang pada kolom tertentu, maka langkah yang dilakukan adalah menghapus baris yang mengandung nilai hilang tersebut menggunakan `dropna()`. Menghindari adanya data yang hilang yang dapat menyebabkan model menghasilkan hasil yang tidak akurat.


### 2.  Menghapus Fitur dengan Korelasi Rendah  
Setelah melakukan analisis korelasi antar fitur numerik, fitur yang memiliki korelasi rendah dapat dihapus untuk mengurangi kompleksitas model. Pada kasus ini, fitur dengan korelasi rendah seperti `Exercise Hours`, `Stress Level`, dan `Blood Sugar` dihapus. Mengurangi kompleksitas model dan meningkatkan akurasi model dengan mengeliminasi fitur yang tidak relevan.


### 3. One-Hot Encoding untuk Fitur Kategorikal  
Fitur kategorikal yang memiliki tipe data objek diubah menjadi variabel dummy (binary) dengan menggunakan one-hot encoding. Ini dilakukan dengan menggunakan fungsi `pd.get_dummies()`. One-hot encoding mengubah setiap kategori menjadi kolom terpisah yang berisi nilai 0 atau 1. Proses ini menghilangkan kategori referensi pertama untuk menghindari multicollinearity. Mengubah data kategorikal menjadi bentuk numerik agar dapat diproses oleh algoritma machine learning. 


### 4. Split Data ke dalam Set Pelatihan dan Pengujian  
Data dibagi menjadi dua set: satu untuk pelatihan (`X_train` dan `y_train`) dan satu untuk pengujian (`X_test` dan `y_test`) menggunakan `train_test_split()`. Proporsi pembagian adalah 80% untuk pelatihan dan 20% untuk pengujian. Memisahkan data agar model dapat dilatih pada data pelatihan dan dievaluasi pada data pengujian yang tidak terlihat sebelumnya.

### 5. Menangani Ketidakseimbangan Kelas dengan SMOTE  
Dataset ini memiliki ketidakseimbangan kelas, di mana jumlah sampel pada kelas minoritas (Heart Disease) lebih sedikit dibandingkan dengan kelas mayoritas. Untuk mengatasi masalah ini, menggunakan SMOTE (Synthetic Minority Over-sampling Technique). SMOTE bekerja dengan cara menghasilkan contoh sintetis dari kelas minoritas untuk menyeimbangkan jumlah sampel antara kelas mayoritas dan kelas minoritas.Mengurangi bias model terhadap kelas mayoritas dan meningkatkan performa model dalam memprediksi kelas minoritas.

### 6. Normalisasi Fitur Numerik  
Fitur numerik distandarisasi menggunakan StandardScaler untuk memastikan bahwa data memiliki skala yang seragam. Hal ini membantu model dalam mempelajari data dengan lebih efisien dan memperbaiki kinerja model. StandardScaler mengubah data menjadi distribusi dengan mean 0 dan standar deviasi 1, yang mengurangi bias terhadap fitur dengan skala yang lebih besar.

## Modeling

Pada tahap ini, beberapa algoritma machine learning digunakan untuk memodelkan data. Setiap model memiliki kelebihan, kekurangan, dan karakteristik unik yang mempengaruhi performa prediksi. Model yang digunakan adalah sebagai berikut:

### 1. Logistic Regression
**Logistic Regression** adalah algoritma klasifikasi linear yang memprediksi probabilitas sebuah data masuk ke salah satu dari dua kelas.

- **Parameter:**
  - `max_iter`: Jumlah iterasi maksimum untuk memastikan konvergensi (diatur ke 1000 dalam proyek ini).
  - `random_state`: Menjamin hasil replikasi.
  
- **Kelebihan:**
  - Sederhana dan efisien untuk dataset linier.
  - Mudah diinterpretasikan.
  
- **Kekurangan:**
  - Kurang efektif untuk data yang tidak linier.

---

### 2. K-Nearest Neighbors (KNN)
**KNN** adalah algoritma non-parametrik yang menggunakan jarak antar titik untuk menentukan kelas data baru berdasarkan *k* tetangga terdekatnya.

- **Parameter:**
  - `n_neighbors`: Jumlah tetangga yang dipertimbangkan (diatur ke 10).
  
- **Kelebihan:**
  - Tidak memerlukan asumsi distribusi data.
  
- **Kekurangan:**
  - Sensitif terhadap outlier.
  - Waktu prediksi lambat pada dataset besar karena menghitung jarak untuk setiap sampel.

---

### 3. Random Forest
**Random Forest** adalah algoritma berbasis ensemble yang menggabungkan beberapa pohon keputusan (*decision tree*) untuk meningkatkan akurasi.

- **Parameter:**
  - `n_estimators`: Jumlah pohon (diatur ke 50).
  - `max_depth`: Kedalaman maksimum pohon (diatur ke 16).
  
- **Kelebihan:**
  - Tahan terhadap overfitting karena menggabungkan prediksi dari banyak pohon.
  - Efektif untuk data dengan banyak fitur.
  
- **Kekurangan:**
  - Bisa lambat pada dataset sangat besar.

---

### 4. AdaBoost
**AdaBoost** adalah algoritma boosting yang memperbaiki kesalahan prediksi model sebelumnya dengan menambahkan bobot lebih pada sampel yang sulit.

- **Parameter:**
  - `learning_rate`: Kecepatan pembaruan bobot (diatur ke 0.05).
  
- **Kelebihan:**
  - Meningkatkan akurasi dengan iterasi bertahap.
  
- **Kekurangan:**
  - Kurang efektif untuk data yang bising.

---

### 5. Gradient Boosting
**Gradient Boosting** adalah algoritma boosting yang mengoptimalkan fungsi kehilangan (*loss function*) melalui iterasi bertahap.

- **Parameter:**
  - `n_estimators`: Jumlah iterasi boosting (diatur ke 100).
  - `learning_rate`: Kecepatan pembaruan (diatur ke 0.1).
  - `max_depth`: Kedalaman maksimum tiap pohon (diatur ke 3).
  
- **Kelebihan:**
  - Fleksibel dan memiliki performa tinggi.
  
- **Kekurangan:**
  - Memerlukan tuning parameter yang lebih kompleks.


### Pemilihan Model
Setiap algoritma dilatih menggunakan *training set* dan diuji pada *test set* untuk mendapatkan metrik akurasi, precision, recall, dan F1 Score.

Berdasarkan hasil evaluasi, model ensemble (Random Forest, AdaBoost, Gradient Boosting) memberikan performa sempurna dengan semua metrik mencapai 100%. Dari ketiga model tersebut, **Gradient Boosting** dipilih sebagai model terbaik karena fleksibilitas dan kemampuannya dalam menangani data kompleks.


## Evaluation

Dalam tahapan ini mengevaluasi performa model klasifikasi untuk mendeteksi risiko penyakit jantung menggunakan empat metrik evaluasi utama: **Akurasi, Precision, Recall,** dan **F1 Score**. Pemilihan metrik ini didasarkan pada kebutuhan untuk menyeimbangkan antara prediksi benar dan kesalahan dalam klasifikasi risiko kesehatan. Berikut penjelasan mengenai setiap metrik, formula yang digunakan, serta hasil evaluasi model berdasarkan metrik tersebut.


### Metrik Evaluasi

### 1. Akurasi
**Formula:**

![picture 5](https://i.imgur.com/VxOI97W.png)  


- **Jumlah Prediksi Benar**: Jumlah sampel yang diprediksi dengan benar oleh model.
- **Total Data**: Jumlah total sampel dalam dataset.

**Cara Kerja:**
Akurasi mengukur persentase prediksi yang benar dari total data yang tersedia. Misalnya, jika model berhasil memprediksi dengan benar 90 dari 100 sampel, maka akurasi adalah 90%. Metrik ini cocok digunakan jika distribusi kelas dalam dataset seimbang.

**Kelebihan:**
- Mudah dipahami dan dihitung.
- Memberikan gambaran umum mengenai performa model.

**Kelemahan:**
- Tidak cocok untuk dataset yang tidak seimbang karena metrik ini tidak mempertimbangkan distribusi kelas. Jika satu kelas mendominasi, model bisa mendapatkan akurasi tinggi hanya dengan memprediksi kelas mayoritas.

---

### 2. Precision
**Formula:**

![picture 6](https://i.imgur.com/ZFTpUmI.png)  


- **True Positives (TP)**: Jumlah sampel positif yang diprediksi benar oleh model.
- **False Positives (FP)**: Jumlah sampel negatif yang diprediksi sebagai positif oleh model.

**Cara Kerja:**
Precision mengukur seberapa banyak prediksi kelas positif yang benar dari total prediksi positif. Sebagai contoh, jika model memprediksi 100 kasus sebagai positif dan 80 di antaranya benar, maka precision adalah 80%. Metrik ini penting dalam situasi di mana kesalahan prediksi positif (false positives) harus diminimalkan, seperti dalam diagnosis penyakit serius.

**Kelebihan:**
- Berguna untuk menghindari false positives, yang sangat penting dalam konteks medis.
  
**Kelemahan:**
- Precision tidak memperhitungkan kasus false negatives, sehingga kurang relevan jika model perlu mendeteksi semua kasus positif.

---

### 3. Recall
**Formula:**

![picture 7](https://i.imgur.com/zd1mCWV.png)  


- **True Positives (TP)**: Jumlah sampel positif yang diprediksi benar oleh model.
- **False Negatives (FN)**: Jumlah sampel positif yang diprediksi sebagai negatif oleh model.

**Cara Kerja:**
Recall mengukur kemampuan model dalam mendeteksi semua instance kelas positif. Misalnya, jika ada 100 kasus positif sebenarnya dan model berhasil mendeteksi 85, maka recall adalah 85%. Metrik ini sangat penting jika mengurangi kesalahan negatif palsu (false negatives) menjadi prioritas, seperti dalam mendeteksi penyakit yang berpotensi fatal.

**Kelebihan:**
- Ideal untuk kasus di mana semua kelas positif harus dideteksi (minimizing false negatives).
  
**Kelemahan:**
- Tidak mempertimbangkan false positives, sehingga model dengan recall tinggi bisa saja memiliki precision yang rendah.

---

### 4. F1 Score
**Formula:**

![picture 8](https://i.imgur.com/eZNXZR7.png)  

- **Precision**: Seberapa banyak dari prediksi positif yang benar.
- **Recall**: Seberapa baik model mendeteksi semua kasus positif.

**Cara Kerja:**
F1 Score adalah rata-rata harmonik antara precision dan recall. Nilai F1 Score tinggi menunjukkan keseimbangan yang baik antara precision dan recall. Sebagai contoh, jika precision adalah 0.9 dan recall adalah 0.8, maka F1 Score akan menjadi 0.847.

**Kelebihan:**
- Berguna jika terdapat trade-off antara precision dan recall, terutama pada dataset yang tidak seimbang.
  
**Kelemahan:**
- Tidak memberikan wawasan yang terpisah antara precision dan recall, sehingga terkadang memerlukan analisis lebih lanjut.

---

### Hasil Evaluasi Model

| **Model**               | **Akurasi (Train)** | **Akurasi (Test)** | **Precision** | **Recall** | **F1 Score** |
|-------------------------|---------------------|--------------------|---------------|------------|--------------|
| Logistic Regression     | 87.60%              | 87.10%             | 78.20%        | 89.60%     | 83.50%       |
| K-Nearest Neighbors (KNN) | 92.70%            | 92.40%             | 88.00%        | 91.70%     | 89.80%       |
| Random Forest           | 100.00%             | 100.00%            | 100.00%       | 100.00%    | 100.00%      |
| AdaBoost                | 100.00%             | 100.00%            | 100.00%       | 100.00%    | 100.00%      |
| Gradient Boosting       | 100.00%             | 100.00%            | 100.00%       | 100.00%    | 100.00%      |

### Analisis Hasil
1. **Logistic Regression**:
   - Akurasi pada test set mencapai 87%.
   - Precision dan recall menunjukkan performa yang seimbang, namun hasilnya lebih rendah dibandingkan model lain karena Logistic Regression cenderung kurang optimal untuk data yang tidak linier.

2. **K-Nearest Neighbors (KNN)**:
   - Akurasi pada test set mencapai 92%.
   - Precision yang lebih rendah (**78%**) menunjukkan model ini cenderung menghasilkan lebih banyak prediksi positif yang salah. Recall cukup tinggi (**89%**), yang berarti model ini cukup sensitif dalam mendeteksi kasus positif.

3. **Random Forest, AdaBoost, Gradient Boosting**:
   - Semua metrik (akurasi, precision, recall, F1 Score) mencapai 100%.
   - Model ini memberikan performa sempurna, baik pada training set maupun test set, menunjukkan kemampuan yang sangat baik dalam menangani data yang kompleks.

### Kesimpulan
**Gradient Boosting** dipilih sebagai model terbaik karena memberikan hasil evaluasi sempurna. Selain akurasi tinggi, model ini lebih fleksibel dalam menangani data yang kompleks.. Evaluasi menggunakan precision dan recall sangat relevan dalam konteks medis, di mana kesalahan prediksi dapat memiliki konsekuensi serius. Gradient Boosting memberikan hasil terbaik untuk kebutuhan ini dengan precision dan recall yang sempurna.

Dengan menggunakan kombinasi keempat metrik ini, performa model dapat dinilai lebih komprehensif sesuai dengan kebutuhan proyek, khususnya dalam kasus medis yang membutuhkan deteksi risiko kesehatan secara akurat.


## Referensi

[1] Pradana, D., Alghifari, M. L., Juna, M. F., & Palaguna, D. (2022). Klasifikasi Penyakit Jantung Menggunakan Metode Artificial Neural Network. Indonesian Journal of Data and Science, 3(2), 55–60. From [tautan](https://doi.org/10.56705/IJODAS.V3I2.35). Retrieved November 16, 2024

[2] Cardiovascular diseases (CVDs). (n.d.). From [tautan](https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)). Retrieved November 16, 2024

[3] Heart Disease prediction. (n.d.). From [tautan](https://www.kaggle.com/datasets/rashadrmammadov/heart-disease-prediction). Retrieved November 16, 2024