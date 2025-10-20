import streamlit as st
import os
from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
from typing import Optional, List, Any
import google.generativeai as genai
import warnings
warnings.filterwarnings('ignore')

# Sayfa yapılandırması
st.set_page_config(
    page_title="Next.js RAG Asistanı - Gemini 2.0",
    page_icon="🤖",
    layout="wide"
)

# Gemini LLM Wrapper
class GeminiLLM(LLM):
    model_name: str = "gemini-2.0-flash-exp"
    temperature: float = 0.7
    
    @property
    def _llm_type(self) -> str:
        return "gemini"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        try:
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=2048,
                )
            )
            return response.text
        except Exception as e:
            return f"Hata oluştu: {str(e)}"
    
    @property
    def _identifying_params(self) -> dict:
        return {"model_name": self.model_name, "temperature": self.temperature}

# Önbellek fonksiyonları
@st.cache_resource
def load_and_prepare_data():
    """Veri setini yükle ve hazırla"""
    with st.spinner("Veri seti yükleniyor..."):
        # Hugging Face veri setini yükle (ilk 100 döküman)
        dataset = load_dataset("ChavyvAkvar/Next.js-Dataset-Converted", split="train[:10]")
        
        # Dökümanları hazırla
        documents = []
        for item in dataset:
            # Veri setindeki text alanını kullan
            text_content = item.get('text', '') or item.get('content', '') or str(item)
            
            if text_content and len(text_content.strip()) > 0:
                doc = Document(
                    page_content=text_content,
                    metadata={"source": "Next.js Dataset"}
                )
                documents.append(doc)
        
        st.success(f"✅ {len(documents)} döküman yüklendi!")
        return documents

@st.cache_resource
def create_vector_store(_documents):
    """Vector store oluştur"""
    with st.spinner("Embeddings oluşturuluyor ve vektör veritabanı hazırlanıyor..."):
        # Text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len
        )
        
        # Metinleri parçala
        texts = text_splitter.split_documents(_documents)
        st.info(f"📄 {len(texts)} metin parçası oluşturuldu")
        
        # Embeddings modeli
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # FAISS vector store
        vectorstore = FAISS.from_documents(texts, embeddings)
        
        st.success("✅ Vektör veritabanı hazır!")
        return vectorstore

def get_qa_chain(vectorstore, gemini_api_key, temperature):
    """QA chain oluştur"""
    try:
        # Gemini API yapılandır
        genai.configure(api_key=gemini_api_key)
        
        # Gemini LLM
        llm = GeminiLLM(temperature=temperature)
        
        # Prompt template
        prompt_template = """Aşağıdaki bağlam bilgisini kullanarak soruyu yanıtla. Eğer cevabı bağlamda bulamazsan, bilmiyorum de ve tahminde bulunma.

Bağlam:
{context}

Soru: {question}

Detaylı Cevap (Türkçe):"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # RetrievalQA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        return qa_chain
    except Exception as e:
        st.error(f"QA Chain oluşturulurken hata: {str(e)}")
        return None

def main():
    st.title("🤖 Next.js RAG Asistanı - Gemini 2.0")
    st.markdown("*Google Gemini 2.0 Flash ile güçlendirilmiştir*")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Ayarlar")
        
        # Gemini API Key
        gemini_api_key = st.text_input(
            "Google Gemini API Key",
            type="password",
            help="Google AI Studio'dan ücretsiz API key alabilirsiniz: https://aistudio.google.com/apikey"
        )
        
        # Temperature ayarı
        temperature = st.slider(
            "Temperature (Yaratıcılık)",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Düşük değer: Daha deterministik, Yüksek değer: Daha yaratıcı"
        )
        
        st.markdown("---")
        st.markdown("### 📊 Model Bilgisi")
        st.info("**Model:** Gemini 2.0 Flash Exp\n\n**Özellikler:**\n- Hızlı yanıt\n- Türkçe desteği\n- Gelişmiş anlama")
        
        st.markdown("### 🔗 Kaynaklar")
        st.markdown("- [Google AI Studio](https://aistudio.google.com)")
        st.markdown("- [Gemini API Docs](https://ai.google.dev)")
        st.markdown("- [Next.js Dataset](https://huggingface.co/datasets/ChavyvAkvar/Next.js-Dataset-Converted)")
    
    # Ana içerik
    if not gemini_api_key:
        st.warning("⚠️ Lütfen Google Gemini API Key'inizi yan panelden girin.")
        st.info("👉 Ücretsiz API key almak için: https://aistudio.google.com/apikey")
        
        # API key alma rehberi
        with st.expander("📖 API Key Nasıl Alınır?"):
            st.markdown("""
            1. https://aistudio.google.com/apikey adresine gidin
            2. Google hesabınızla giriş yapın
            3. "Create API Key" butonuna tıklayın
            4. Oluşturulan API key'i kopyalayın
            5. Sol paneldeki alana yapıştırın
            """)
        return
    
    # Veri setini yükle
    try:
        documents = load_and_prepare_data()
        
        # Vector store oluştur
        vectorstore = create_vector_store(documents)
        
        # QA chain oluştur
        qa_chain = get_qa_chain(vectorstore, gemini_api_key, temperature)
        
        if qa_chain is None:
            return
        
        st.markdown("---")
        st.success("✅ Sistem hazır! Gemini 2.0 ile sorularınızı sorabilirsiniz.")
        
        # Örnek sorular
        with st.expander("💡 Örnek Sorular"):
            st.markdown("""
            - Next.js'de server-side rendering nasıl yapılır?
            - App Router ve Pages Router arasındaki farklar nelerdir?
            - Next.js'de API route'ları nasıl oluşturulur?
            - Static Site Generation (SSG) nedir?
            - Next.js'de middleware nasıl kullanılır?
            """)
        
        # Soru-cevap bölümü
        st.markdown("### 💬 Soru Sorun")
        
        question = st.text_area(
            "Sorunuz:",
            placeholder="Örnek: Next.js'de server-side rendering nasıl yapılır?",
            height=100
        )
        
        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            ask_button = st.button("🔍 Ara", use_container_width=True, type="primary")
        with col2:
            clear_button = st.button("🗑️ Temizle", use_container_width=True)
        
        if clear_button:
            st.rerun()
        
        if ask_button and question:
            with st.spinner("🤔 Gemini 2.0 düşünüyor..."):
                try:
                    result = qa_chain({"query": question})
                    
                    # Cevabı göster
                    st.markdown("### 📝 Cevap")
                    st.markdown(f"""
                    <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #4CAF50;">
                    {result['result']}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Kaynak dökümanları göster
                    if 'source_documents' in result and result['source_documents']:
                        st.markdown("### 📚 Kaynak Dökümanlar")
                        for i, doc in enumerate(result['source_documents'], 1):
                            with st.expander(f"📄 Kaynak {i}"):
                                st.text(doc.page_content[:800] + "..." if len(doc.page_content) > 800 else doc.page_content)
                
                except Exception as e:
                    st.error(f"❌ Hata oluştu: {str(e)}")
                    st.info("💡 API key'inizi kontrol edin veya birkaç saniye bekleyip tekrar deneyin.")
        
        elif ask_button and not question:
            st.warning("⚠️ Lütfen bir soru girin.")
    
    except Exception as e:
        st.error(f"❌ Uygulama başlatılırken hata: {str(e)}")
        st.info("Lütfen requirements.txt dosyasındaki tüm paketlerin kurulu olduğundan emin olun.")

if __name__ == "__main__":
    main()