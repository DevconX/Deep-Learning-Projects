# Bahasa-Entities-Recognition
Use deep learning models, SVM and Multinomial to classify entities

## Entities supported
```text
PRN - person, group of people, believes, etc
LOC - location
NORP - Military, police, government, Parties, etc
ORG - Organization, company
LAW - related law document, etc
ART - art of work, special names, etc
EVENT - event happen, etc
FAC - facility, hospitals, clinics, etc
TIME - date, day, time, etc
```

## Models used

1. LSTM + CRF + chars embeddings + Dynamic Bidirectional
2. LSTM + CRF + chars embeddings + Static Bidirectional
3. LSTM + chars sequence + Static Bidirectional
4. Multinomial Naive Bayes on BOW & TFIDF char-wise
5. SVM on BOW & TFIDF char-wise

## Results

LSTM + CRF + chars embeddings + Dynamic Bidirectional
```text
precision    recall  f1-score   support

        PAD       0.98      0.98      0.98      8063
        FAC       0.99      1.00      0.99    185006
        ORG       0.99      0.95      0.97     23771
      EVENT       0.97      0.99      0.98      2600
        PRN       0.96      0.91      0.93      6080
        LOC       0.98      0.95      0.97      2120
          O       0.96      0.92      0.94      2120
        LAW       0.98      0.93      0.95     10280
       NORP       0.96      0.99      0.97      1280
       TIME       0.99      0.94      0.97      1860
        ART       0.59      0.50      0.54        20

avg / total       0.99      0.99      0.99    243200
```

LSTM + CRF + chars embeddings + Static Bidirectional
```text
precision    recall  f1-score   support

      EVENT       0.98      0.96      0.97      8063
        ORG       0.99      1.00      0.99    185006
        FAC       0.98      0.96      0.97     23771
        DOC       0.95      0.99      0.97      2600
        LAW       0.97      0.86      0.91      6080
       NORP       0.96      0.96      0.96      2120
        ART       0.98      0.92      0.95      2120
        PAD       0.98      0.92      0.95     10280
       TIME       0.97      0.88      0.93      1280
        PRN       0.98      0.96      0.97      1860
        LOC       1.00      0.65      0.79        20

avg / total       0.98      0.98      0.98    243200
```

Multinomial Naive Bayes on BOW & TFIDF char-wise
```text
             precision    recall  f1-score   support

       NORP       0.68      0.29      0.41       405
        FAC       0.85      0.97      0.91      9275
        ART       0.64      0.52      0.57      1190
        ORG       0.95      0.15      0.25       130
       TIME       0.42      0.32      0.36       304
        PAD       0.38      0.11      0.17       106
        LAW       0.55      0.10      0.17       106
        PRN       0.64      0.28      0.39       514
        DOC       0.67      0.03      0.06        64
        LOC       0.90      0.10      0.17        93
      EVENT       0.00      0.00      0.00         1

avg / total       0.80      0.82      0.79     12188
```

SVM on BOW & TFIDF char-wise
```text
             precision    recall  f1-score   support

        FAC       0.74      0.61      0.67       405
        PRN       0.89      0.99      0.94      9275
        ART       0.86      0.72      0.78      1190
        DOC       0.96      0.65      0.78       130
       NORP       0.76      0.26      0.38       304
          O       0.96      0.41      0.57       106
        PAD       0.89      0.24      0.37       106
        LAW       0.78      0.46      0.57       514
      EVENT       0.88      0.36      0.51        64
       TIME       0.88      0.47      0.62        93
        ORG       0.00      0.00      0.00         1

avg / total       0.88      0.88      0.87     12188
```

## Predicted output

LSTM + CRF + chars embeddings + Dynamic Bidirectional
```text
KUALA LOC
LUMPUR: LOC
Sempena O
sambutan O
Aidilfitri O
minggu TIME
depan, TIME
Perdana PRN
Menteri PRN
Tun PRN
Dr PRN
Mahathir PRN
Mohamad PRN
dan O
Menteri PRN
Pengangkutan PRN
Anthony PRN
Loke PRN
Siew PRN
Fook PRN
menitipkan EVENT
pesanan EVENT
khas EVENT
kepada EVENT
orang EVENT
ramai EVENT
yang EVENT
mahu EVENT
pulang EVENT
ke EVENT
kampung EVENT
halaman EVENT
masing-masing. EVENT
Dalam EVENT
video EVENT
pendek EVENT
terbitan EVENT
Jabatan EVENT
Keselamatan EVENT
Jalan EVENT
Raya EVENT
(JKJR) EVENT
itu, EVENT
Dr EVENT
Mahathir EVENT
menasihati EVENT
mereka EVENT
supaya EVENT
berhenti EVENT
berehat EVENT
dan EVENT
tidur EVENT
sebentar EVENT
sekiranya EVENT
mengantuk EVENT
ketika EVENT
memandu. EVENT
```

LSTM + CRF + chars embeddings + Static Bidirectional
```text
KUALA LOC
LUMPUR: LOC
Sempena O
sambutan O
Aidilfitri TIME
minggu TIME
depan, TIME
Perdana PRN
Menteri PRN
Tun PRN
Dr PRN
Mahathir PRN
Mohamad PRN
dan O
Menteri PRN
Pengangkutan PRN
Anthony PRN
Loke PRN
Siew PRN
Fook PRN
menitipkan PRN
pesanan PRN
khas PRN
kepada O
orang O
ramai O
yang O
mahu O
pulang O
ke O
kampung LOC
halaman O
masing-masing. O
Dalam O
video O
pendek O
terbitan O
Jabatan O
Keselamatan O
Jalan O
Raya ART
(JKJR) O
itu, O
Dr PRN
Mahathir PRN
menasihati O
mereka O
supaya O
berhenti O
berehat O
dan O
tidur O
sebentar O
sekiranya O
mengantuk O
ketika O
memandu. O
```

LSTM + chars sequence + Static Bidirectional
```text
KUALA LOC
LUMPUR: LOC
Sempena O
sambutan O
Aidilfitri EVENT
minggu O
depan, O
Perdana PRN
Menteri PRN
Tun PRN
Dr PRN
Mahathir PRN
Mohamad PRN
dan O
Menteri PRN
Pengangkutan NORP
Anthony PRN
Loke PRN
Siew PRN
Fook O
menitipkan O
pesanan O
khas NORP
kepada O
orang O
ramai O
yang O
mahu O
pulang O
ke O
kampung LOC
halaman O
masing-masing. O
Dalam O
video O
pendek O
terbitan O
Jabatan NORP
Keselamatan O
Jalan LOC
Raya O
(JKJR) O
itu, O
Dr PRN
Mahathir PRN
menasihati O
mereka O
supaya O
berhenti O
berehat O
dan O
tidur O
sebentar O
sekiranya O
mengantuk O
ketika O
memandu. O
```
