# -----------------------------------------------------------
# RAG Adım 2: Embedding ve Vektör Veritabanı (ChromaDB) Oluşturma
# KRİTİK ÇÖZÜM: API anahtarını doğrudan Embeddings sınıfına iletiyoruz.
# -----------------------------------------------------------

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from google import genai  # Google'ın kendi istemci kütüphanesi
import os

print("2. Embedding ve ChromaDB Oluşturuluyor...")

# NOT: 'texts' değişkeninin (parçalanmış dokümanlar) hafızada olduğundan emin olmak için 
# bu hücreden önceki (Veri Yükleme) hücresi çalıştırılmış olmalıdır.

# 1. Gemini İstemcisini oluşturma (Gerekli kütüphane içe aktarmaları için)
# Bu istemci, ortam değişkeninden anahtarı alır.
try:
    client = genai.Client()
except Exception as e:
    # Bu noktada hata alınması, API anahtarının Secrets'ten çekilemediği anlamına gelir.
    print(f"HATA: GenAI istemcisi oluşturulamadı. API Anahtarınızı kontrol edin: {e}")
    
# 2. Embedding Modelini, api_key parametresini kullanarak açıkça başlatma
# Bu, GCE meta verisi üzerinden kimlik doğrulama denemesini atlamaya zorlar.
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    api_key=os.environ["GEMINI_API_KEY"] # API anahtarını ortam değişkeninden çekerek açıkça iletiyoruz.
)

# Vektör Veritabanını (ChromaDB) oluşturma ve parçaları kaydetme
# BU ADIM UZUN SÜREBİLİR (yaklaşık 50.000 parça)
vectorstore = Chroma.from_documents(
    documents=texts, 
    embedding=embeddings, 
    persist_directory="./chroma_db"
)

# Retriever (Geri Alıcı) nesnesini oluşturma
retriever = vectorstore.as_retriever(search_kwargs={"k": 4}) 

print("Vektör Veritabanı Başarıyla Oluşturuldu ve Retriever Hazır.")
