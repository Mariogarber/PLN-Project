from torch.utils.data import Dataset
import pandas as pd
from typing import List

class ToxicTokenDataset(Dataset):
    """
    Dataset para clasificación de tokens en múltiples idiomas.
    Espera varios archivos .parquet con columnas:
        - 'text': palabra (string)
        - 'label': True/False si la palabra es tóxica
    Cada fila es una palabra independiente.
    """

    def __init__(self, parquet_paths: List[str]):
        super().__init__()
        # Cargar y mezclar todos los parquets
        dfs = [pd.read_parquet(p) for p in parquet_paths]
        self.data = pd.concat(dfs, ignore_index=True)
        self.data = self.data.sample(frac=1).reset_index(drop=True)

        # Asegurar tipos correctos
        self.data["text"] = self.data["text"].astype(str)
        self.data["label"] = self.data["label"].astype(int)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "text": self.data.iloc[idx]["text"],
            "label": self.data.iloc[idx]["label"]
        }

class ToxicNonToxicDataset(Dataset):
    """
    Dataset para tareas de detoxificación de texto.
    Espera varios archivos .parquet con columnas:
        - 'toxic_sentences': texto tóxico original (string)
        - 'neutral_sentences': versión detoxificada del texto (string)
    Cada fila es un par de oraciones (tóxica -> neutral).
    """

    def __init__(self, parquet_paths: List[str]):
        super().__init__()
        # Cargar y mezclar todos los parquets
        dfs = []
        for p in parquet_paths:
            df = pd.read_parquet(p)
            lang = p.split('_')[-1].split('.parquet')[0]
            df['language'] = lang
            dfs.append(df)
        self.data = pd.concat(dfs, ignore_index=True)
        
        # Limpiar datos: remover filas con valores nulos
        self.data = self.data.dropna(subset=["toxic_sentence", "neutral_sentence"])
        self.data = self.data.reset_index(drop=True)

        # Limpiar espacios en blanco
        self.data["toxic_sentence"] = self.data["toxic_sentence"].str.strip()
        self.data["neutral_sentence"] = self.data["neutral_sentence"].str.strip()

        # Mezclar datos
        self.data = self.data.sample(frac=1).reset_index(drop=True)

        # Asegurar tipos correctos
        self.data["toxic_sentence"] = self.data["toxic_sentence"].astype(str)
        self.data["neutral_sentence"] = self.data["neutral_sentence"].astype(str)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "toxic_text": self.data.iloc[idx]["toxic_sentence"],
            "neutral_text": self.data.iloc[idx]["neutral_sentence"],
            "language": self.data.iloc[idx]["language"]
        }