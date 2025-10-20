import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from datasets import load_dataset
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

# --- RAG Adım 1: Veri Yükleme ve Parçalama ---
# Bu kısım, Streamlit uygulaması ilk başlatıldığında çalışacak ve veriyi yükleyip parçalayacaktır.
# Veri yükleme ve parçalama işlemini önbelleğe alarak (cache) uygulamanın her yeniden yüklenmesinde tekrar çalışmasını engelliyoruz.
@st.cache_resource
def load_and_split_data(subset_size=2000):
    st.write("1. Next.js Veri Seti Yükleniyor (Hugging Face kütüphanesi ile)...")
    hf_dataset = load_dataset("ChavyvAkvar/Next.js-Dataset-Converted", split="train")
    hf_dataset_subset = hf_dataset.select(range(subset_size))
    st.write(f"Veri seti boyutu {subset_size} dokümanla sınırlandırıldı.")

    documents = []
    for item in hf_dataset_subset:
        if 'messages' in item and item['messages'] and isinstance(item['messages'], list) and isinstance(item['messages'][0], dict):
            content = item['messages'][0].get('content')
            if content:
                documents.append(
                    Document(
                        page_content=content,
                        metadata={"source": "Next.js-Dataset", "index": str(len(documents))}
                    )
                )

    st.write(f"Yüklenen Toplam Doküman Sayısı (Kısıtlanmış): {len(documents)}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)
    st.write(f"Oluşturulan Toplam Metin Parçası Sayısı: {len(texts)}")
    return texts

# Veriyi yükle ve parçala
texts = load_and_split_data(subset_size=2000)

# --- RAG Adım 2: Embedding ve Vektör Veritabanı (ChromaDB) Oluşturma ---
# Embedding modelini ve vektör veritabanını önbelleğe alıyoruz.
@st.cache_resource
def create_vectorstore(_texts):
    st.write("2. Embedding (Yerel Model) ve ChromaDB Oluşturuluyor...")
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=_texts,
        embedding=embeddings,
        persist_directory="./chroma_db" # Yerel olarak kaydedilecek
    )
    st.write("Vektör Veritabanı Başarıyla Oluşturuldu ve Retriever Hazır.")
    return vectorstore

# Vektör veritabanını oluştur
vectorstore = create_vectorstore(texts)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# --- RAG Adım 3: Dil Modeli (Gemini) ve RAG Zinciri Kurulumu ---
# LLM ve RAG zincirini önbelleğe alıyoruz.
@st.cache_resource
def setup_rag_chain(_retriever):
    st.write("3. Dil Modeli (Gemini) ve RAG Zinciri Kuruluyor...")
    # API Anahtarını ortam değişkeninden al
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
         st.error("HATA: GEMINI_API_KEY ortam değişkeni tanımlanmamış.")
         st.stop() # Uygulamayı durdur

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7, google_api_key=api_key)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=_retriever,
        return_source_documents=True
    )
    st.write("Dil Modeli ve RAG Zinciri Başarıyla Kuruldu.")
    return qa_chain

# RAG zincirini kur
qa_chain = setup_rag_chain(retriever)

# --- RAG Adım 4: Sorgu Çalıştırma (Streamlit Arayüzü Üzerinden) ---
st.title("Next.js RAG Sorgulama Uygulaması")
st.info("Next.js veri seti üzerinde RAG (Retrieval Augmented Generation) kullanarak sorularınıza yanıt bulun.")

query = st.text_input("Lütfen Next.js hakkında bir soru sorun:", key="rag_query_input")

if query:
    with st.spinner("Yanıt aranıyor..."):
        try:
            response = qa_chain.invoke({"query": query})
            st.subheader("Yanıt:")
            st.write(response.get("result", "Yanıt alınamadı."))

            if "source_documents" in response and response["source_documents"]:
                st.subheader("Kaynak Dokümanlar:")
                for i, doc in enumerate(response["source_documents"]):
                    st.write(f"**Kaynak {i+1}:**")
                    st.write(f"- Kaynak: {doc.metadata.get('source', 'Bilinmiyor')}")
                    st.write(f"  İçerik: {doc.page_content[:300]}...") # İlk 300 karakteri göster
            else:
                st.info("Bu sorgu için kaynak doküman bulunamadı.")

        except Exception as e:
            st.error(f"Sorgu çalıştırılırken bir hata oluştu: {e}")

st.markdown("---")
st.markdown("Bu uygulama bir RAG (Retrieval Augmented Generation) örneğidir.")