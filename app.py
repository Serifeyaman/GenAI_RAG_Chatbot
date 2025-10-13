# ==============================================================================
# PROJE BAĞIMLILIKLARI
# Tüm kütüphaneler requirements.txt dosyasında listelenmiştir.
# API anahtarı, GEMINI_API_KEY ortam değişkeni ile tanımlanmalıdır (Lokalde .env, Colab'de Secrets).
# ==============================================================================

import os
from datasets import load_dataset
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from google import genai 

# NOT: Bu kod, API anahtarının (GEMINI_API_KEY) Python ortamında tanımlı olduğunu varsayar.

# -----------------------------------------------------------
# RAG Adım 1: Veri Yükleme ve Parçalama
# Teknik Açıklama: Hugging Face 'datasets' kütüphanesi ile veri yüklenir, 
# kota limitleri nedeniyle 2000 dokümanla sınırlandırılır ve LangChain Document formatına dönüştürülür.
# -----------------------------------------------------------

def load_and_split_data(dataset_name="ChavyvAkvar/Next.js-Dataset-Converted", subset_size=2000):
    """Veri setini yükler, sınırlar, işler ve parçalara ayırır."""
    print("1. Next.js Veri Seti Yükleniyor...")
    hf_dataset = load_dataset(dataset_name, split="train")

    # Kota limitleri için veri setini sınırla
    hf_dataset_subset = hf_dataset.select(range(subset_size)) 
    print(f"Veri seti boyutu {subset_size} dokümanla sınırlandırıldı.")

    documents = []
    for item in hf_dataset_subset: 
        # Karmaşık yapıdaki 'messages' altındaki 'content' ayıklanır.
        if 'messages' in item and item['messages'] and isinstance(item['messages'], list) and isinstance(item['messages'][0], dict):
            content = item['messages'][0].get('content')
            if content:
                documents.append(
                    Document(
                        page_content=content,
                        metadata={"source": dataset_name, "index": str(len(documents))}
                    )
                )

    print(f"Yüklenen Toplam Doküman Sayısı (Kısıtlanmış): {len(documents)}")

    # Parçalama (Chunking) işlemi
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)
    print(f"Oluşturulan Toplam Metin Parçası Sayısı: {len(texts)}")
    return texts

# -----------------------------------------------------------
# RAG Adım 2: Embedding ve Vektör Veritabanı (ChromaDB) Oluşturma
# Teknik Açıklama: Metin parçaları Gemini Embedding ile vektörleştirilir ve ChromaDB'ye kaydedilir.
# -----------------------------------------------------------

def create_vector_store(texts):
    """Metin parçalarını vektörleştirir ve ChromaDB'ye kaydeder."""
    print("2. Embedding ve ChromaDB Oluşturuluyor...")
    
    # Embedding Modelini, api_key ve Toplu İşlem Boyutu ile başlatma (Kota yönetimi için)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        api_key=os.environ["GEMINI_API_KEY"],
        batch_size=100 
    )

    # Vektör Veritabanını oluşturma
    vectorstore = Chroma.from_documents(
        documents=texts, 
        embedding=embeddings, 
        persist_directory="./chroma_db"
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4}) 
    print("Vektör Veritabanı Başarıyla Oluşturuldu ve Retriever Hazır.")
    return retriever

# -----------------------------------------------------------
# RAG Adım 3: Generation (Üretim) Zincirini Kurma ve Test
# Teknik Açıklama: Gemini LLM, retriever ile birleştirilerek RAG zinciri oluşturulur ve test edilir.
# -----------------------------------------------------------

def run_qa_chain(retriever):
    """RAG zincirini kurar ve test sorgusunu çalıştırır."""
    print("3. RAG Zinciri Kuruluyor...")
    
    # Generation Modeli (Gemini-2.5-flash)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0.0,
        api_key=os.environ["GEMINI_API_KEY"] 
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=retriever
    )

    # Test Sorgusu
    query = "Next.js'de veri çekmenin (data fetching) üç ana yolu nedir? Cevabı tamamen Türkçe ve maddeler halinde ver."

    print(f"\n4. Test Sorgusu Başlatılıyor...")
    print(f"Soru: {query}")
    
    result = qa_chain.invoke({"query": query})

    print("\n--- CHATBOT CEVABI (Türkçe) ---")
    print(result['result'])
    print("---------------------------------")


# ==============================================================================
# ANA ÇALIŞMA FONKSİYONU
# ==============================================================================

if __name__ == "__main__":
    if not os.getenv("GEMINI_API_KEY"):
        raise ValueError("Lütfen GEMINI_API_KEY ortam değişkenini tanımlayın.")
    
    # 1. Veri Yükleme ve Parçalama
    texts = load_and_split_data()

    # 2. Embedding ve Vektör Veritabanı Oluşturma
    # DİKKAT: Kota yenilenmeden bu adım hata verecektir!
    retriever = create_vector_store(texts)

    # 3. RAG Zincirini Kurma ve Test
    run_qa_chain(retriever)
