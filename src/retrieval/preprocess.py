import unicodedata
import re
from dotenv import load_dotenv
from typing import List

load_dotenv()

_STOPWORDS_PT = {
        'a', 'ao', 'aos', 'aquela', 'aquelas', 'aquele', 'aqueles', 'aquilo',
        'as', 'ate', 'com', 'como', 'da', 'das', 'de', 'dela', 'delas', 'dele',
        'deles', 'depois', 'do', 'dos', 'e', 'ela', 'elas', 'ele', 'eles', 'em',
        'entre', 'era', 'essa', 'essas', 'esse', 'esses', 'esta', 'estas', 'este',
        'estes', 'eu', 'foi', 'for', 'foram', 'ha', 'isso', 'isto', 'ja', 'lhe',
        'lhes', 'mais', 'mas', 'me', 'mesmo', 'meu', 'meus', 'minha', 'minhas',
        'muito', 'na', 'nas', 'nao', 'nos', 'no', 'numa', 'o', 'os', 'ou',
        'para', 'pela', 'pelas', 'pelo', 'pelos', 'por', 'qual', 'quando', 'que',
        'se', 'seu', 'seus', 'sua', 'suas', 'tambem', 'te', 'tem', 'tendo',
        'tera', 'tinha', 'tudo', 'um', 'uma', 'umas', 'uns', 'voce', 'voces',
        'ser', 'sido', 'sendo', 'sao', 'sera', 'serao', 'esta', 'estao', 'estou',
}

def preprocess_pt(text: str) -> List[str]:
        '''
        Pré-processamento para português:
        1. Normaliza NFD e remove diacríticos (acentos)
        2. Converte para minúsculas
        3. Remove pontuação e caracteres não-alfa-numéricos
        4. Remove stopwords e tokens muito curtos
        Usado tanto pelo BM25Retriever quanto na query de busca.
        '''
        text = unicodedata.normalize('NFD', text)
        text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        tokens = text.split()
        return [t for t in tokens if t not in _STOPWORDS_PT and len(t) > 2]
