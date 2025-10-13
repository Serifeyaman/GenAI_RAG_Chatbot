# GenAI RAG Chatbot
Generative AI RAG Chatbot

## Projenin Amacı

Bu projenin temel amacı, RAG (Retrieval Augmented Generation) mimarisini kullanarak belirli bir teknik dokümantasyon (Next.js) üzerinde çalışan, güncel ve bağlamsal olarak doğru cevaplar üretebilen bir chatbot geliştirmektir.
Chatbot, kullanıcıların doğal dilde sorduğu sorulara, modelin eğitildiği veriler yerine dışarıdan aldığı bağlama dayanarak yanıt verecektir.

## Veri Seti Hakkında

* **Veri Seti:** [ChavyvAkvar/Next.js-Dataset-Converted](https://huggingface.co/datasets/ChavyvAkvar/Next.js-Dataset-Converted/viewer/default/train?row=0&views%5B%5D=train) (Hugging Face).
* **İçerik:** Next.js web geliştirme framework'üne ait teknik dökümantasyon ve Soru-Cevap metinlerini içermektedir.
* **Hazırlık:** Kota limitleri (429 Quota Exceeded) nedeniyle, orijinal 49.954 dokümanlık veri seti, RAG pipeline'ını test etmek amacıyla **ilk 2000 dokümanla** sınırlandırılmıştır. Metin içeriği, veri setinin karmaşık yapısından dolayı özel bir kodlama ile 'messages' altındaki 'content' sütunundan manuel olarak ayıklanmıştır.
* **Dil:** İngilizce. Chatbot, LLM'e verilen özel komutlar (Prompt Engineering) sayesinde İngilizce bağlamı kullanarak Türkçe cevaplar üretebilmektedir.

## Kullanılan Yöntemler

Proje, temel bir **RAG (Retrieval Augmented Generation) Mimarisi** üzerine kurulmuştur.

* **RAG Çatısı:** LangChain.
* **Generation (LLM):** Google Gemini 2.5 Flash API (Türkçe cevap üretimi için prompt engineering kullanılmıştır).
* **Embedding (Vektörleştirme):** Google/Gemini Embedding Model (`models/embedding-001`).
* **Vektör Veritabanı:** ChromaDB (Lokal kalıcı depolama).
* **Veri İşleme:** Hugging Face `datasets` kütüphanesi ve `RecursiveCharacterTextSplitter` kullanılarak dokümanlar 1000 karakterlik parçalara ayrılmıştır.

## Elde Edilen Sonuçlar (Özet)

* **Başarı:** RAG pipeline'ı başarılı bir şekilde kurulmuş ve çalışır durumdadır.
* **Çözüm:** Gemini API kota limitleri, veri setinin küçültülmesi ve `batch_size=100` ayarı ile yönetilmiş, kimlik doğrulama sorunları `api_key`'in açıkça iletilmesiyle aşılmıştır.

## Web Linkiniz
