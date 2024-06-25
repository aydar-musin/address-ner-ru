# address-ner-ru

Address NER model to find address parts from string

[https://huggingface.co/aidarmusin/address-ner-ru](https://huggingface.co/aidarmusin/address-ner-ru)

# Dataset

5K raw addresses dataset

90% for training and 10% for evaluation

# Evaluation

| Metric | Value |
| --- | --- |
| eval_overall_precision | 0.9550486413955048 |
| eval_overall_recall | 0.9644308943089431 |
| eval_overall_f1 | 0.9597168380246082 |
| eval_overall_accuracy | 0.9770456798596813 |
| eval_Apartment_f1 | 0.9663865546218489 |
| eval_Apartment_number | 352 |
| eval_Building_precision | 0.8695652173913043 |
| eval_Building_recall | 0.9195402298850575 |
| eval_Building_f1 | 0.8938547486033519 |
| eval_Building_number | 87 |
| eval_Country_precision | 0.9950738916256158 |
| eval_Country_recall | 0.9805825242718447 |
| eval_Country_f1 | 0.9877750611246944 |
| eval_Country_number | 206 |
| eval_District_precision | 0.9562043795620438 |
| eval_District_recall | 0.9924242424242424 |
| eval_District_f1 | 0.9739776951672863 |
| eval_District_number | 132 |
| eval_House_precision | 0.9702380952380952 |
| eval_House_recall | 0.9760479041916168 |
| eval_House_f1 | 0.973134328358209 |
| eval_House_number | 501 |
| eval_Region_precision | 0.9826989619377162 |
| eval_Region_recall | 0.9861111111111112 |
| eval_Region_f1 | 0.9844020797227037 |
| eval_Region_number | 288 |
| eval_Settlement_precision | 0.9599271402550091 |
| eval_Settlement_recall | 0.9547101449275363 |
| eval_Settlement_f1 | 0.9573115349682106 |
| eval_Settlement_number | 552 |
| eval_Street_precision | 0.9424603174603174 |
| eval_Street_recall | 0.9615384615384616 |
| eval_Street_f1 | 0.9519038076152305 |
| eval_Street_number | 494 |
| eval_ZipCode_precision | 0.9208211143695014 |
| eval_ZipCode_recall | 0.9235294117647059 |
| eval_ZipCode_f1 | 0.9221732745961821 |
| eval_ZipCode_number | 340 |

# Example

```python
from transformers import pipeline
import torch
import logging

device = "cuda:0" if torch.cuda.is_available() else "cpu"
logging.info(f"using device: {device}")

address_ner_pipeline = pipeline("ner", model="aidarmusin/address-ner-ru", device=device)

address = "628672,,,, Автономный Округ Ханты-Мансийский Автономный Округ - Югра,, Г. Лангепас, Ул. Солнечная, Д.21"
entities = address_ner_pipeline(address)
print(entities)
```

# address-ner-ru

Модель NER для адресов, предназначенная для извлечения частей адреса из строки
Транформер модель, ИИ, парсинг адресов, бесплатный аналог дадата dadata.ru,
https://huggingface.co/aidarmusin/address-ner-ru

