from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class Treater():
    def __init__(self, file):
        self.file = file

    def load_documents(self):
        """
        Retorna lista de Document objects com metadados de página e fonte.
        PyMuPDF preserva melhor a estrutura de PDFs com formatação complexa.
        """
        loader = PyMuPDFLoader(self.file)
        return loader.load()

    def split_chunks(self, chunk_size=1000, overlap=150):
        """
        Divide os documentos em chunks respeitando fronteiras semânticas.
        Ordem de separadores: parágrafo → linha → frase → palavra.
        Retorna lista de Document objects com metadados preservados.
        """
        docs = self.load_documents()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(docs)
        return chunks
