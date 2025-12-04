import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List

import matplotlib.pyplot as plt
import seaborn as sns

# Métricas externas
from bert_score import score as bertscore
from sklearn.metrics import f1_score
from sacrebleu.metrics import BLEU, CHRF

from sentence_transformers import SentenceTransformer, util


class Evaluator:
    """
    Evaluator class for detoxification models.

    Takes as input a dataframe with columns:
        - toxic
        - gold_neutral
        - predicted_neutral
        - (optional) language
        - (optional) difficulty

    Computes:
        - Semantic similarity (BERTScore + SBERT)
        - Structural similarity (BLEU/chrF)
        - Copy-rate
        - Length ratios
        - Toxicity estimation (lexicon-based)
        - Drift scores
        - Grouped metrics by language/difficulty
        - Visualizations
    """

    def __init__(self, toxic_lexicons: Optional[Dict[str, List[str]]] = None, device="cpu"):
        """
        toxic_lexicons: dict mapping language -> list of toxic words
        """
        self.toxic_lexicons = toxic_lexicons or {}
        self.device = device

        # SBERT multilingual model
        self.sbert = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v2",
                                         device=device)

        self.bleu_metric = BLEU()
        self.chrf_metric = CHRF()

    # ======================================================
    # ---------------- BASIC UTILITIES ---------------------
    # ======================================================

    def _compute_copy_rate(self, df):
        return np.mean(df["toxic"] == df["predicted_neutral"])

    def _toxicity_count(self, text: str, lang: str = None):
        """
        Lexicon-based toxicity estimate.
        Counts toxic words in the predicted string.
        """
        if lang not in self.toxic_lexicons:
            # If no lexicon available for the language → zero (safe fallback)
            return 0

        lex = self.toxic_lexicons[lang]
        toks = text.lower().split()
        return sum(t in lex for t in toks)

    # ======================================================
    # ---------------- SEMANTIC METRICS --------------------
    # ======================================================

    def compute_bertscore(self, df):
        """
        Compute BERTScore F1 between prediction and gold.
        """
        preds = df["predicted_neutral"].to_list()
        golds = df["gold_neutral"].to_list()

        P, R, F1 = bertscore(preds, golds, lang="multilingual", device=self.device)
        return float(F1.mean())

    def compute_sbert_similarity(self, df):
        """
        SBERT cosine similarity between prediction and gold.
        """
        preds = df["predicted_neutral"].to_list()
        golds = df["gold_neutral"].to_list()

        emb_pred = self.sbert.encode(preds, convert_to_tensor=True)
        emb_gold = self.sbert.encode(golds, convert_to_tensor=True)

        sims = util.pytorch_cos_sim(emb_pred, emb_gold)
        return float(sims.diag().mean())

    # ======================================================
    # ---------------- STRUCTURAL METRICS ------------------
    # ======================================================

    def compute_bleu(self, df):
        preds = [p for p in df["predicted_neutral"]]
        golds = [[g] for g in df["gold_neutral"]]
        return self.bleu_metric.corpus_score(preds, golds).score

    def compute_chrf(self, df):
        preds = [p for p in df["predicted_neutral"]]
        golds = [[g] for g in df["gold_neutral"]]
        return self.chrf_metric.corpus_score(preds, golds).score

    def compute_length_ratio(self, df):
        return np.mean(df["predicted_neutral"].str.split().str.len() /
                       df["gold_neutral"].str.split().str.len())

    # ======================================================
    # ---------------- DETOX METRICS -----------------------
    # ======================================================

    def compute_residual_toxicity(self, df):
        """
        Lexicon-based toxicity count normalized by length.
        """
        scores = []
        for idx, row in df.iterrows():
            lang = row["language"] if "language" in df.columns else None
            tox = self._toxicity_count(row["predicted_neutral"], lang=lang)
            length = max(1, len(row["predicted_neutral"].split()))
            scores.append(tox / length)
        return float(np.mean(scores))

    # ======================================================
    # ---------------- DRIFT METRICS -----------------------
    # ======================================================

    def compute_semantic_drift(self, df):
        """
        SBERT similarity between toxic input and prediction.
        Too low → model rewrites too aggressively.
        Too high → model copies too much.
        """
        tox = df["toxic"].to_list()
        preds = df["predicted_neutral"].to_list()

        emb_tox = self.sbert.encode(tox, convert_to_tensor=True)
        emb_pred = self.sbert.encode(preds, convert_to_tensor=True)

        sims = util.pytorch_cos_sim(emb_tox, emb_pred)
        return float(sims.diag().mean())

    # ======================================================
    # ---------------- GLOBAL EVALUATION -------------------
    # ======================================================

    def evaluate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Returns a dictionary with all global metrics.
        """
        results = {}

        results["copy_rate"] = self._compute_copy_rate(df)
        results["bertscore_f1"] = self.compute_bertscore(df)
        results["sbert_similarity"] = self.compute_sbert_similarity(df)
        results["bleu"] = self.compute_bleu(df)
        results["chrf"] = self.compute_chrf(df)
        results["length_ratio"] = self.compute_length_ratio(df)
        results["residual_toxicity"] = self.compute_residual_toxicity(df)
        results["semantic_drift"] = self.compute_semantic_drift(df)

        return results

    # ======================================================
    # ---------------- GROUPED EVALUATION ------------------
    # ======================================================

    def evaluate_by(self, df, group_col: str):
        """
        Evaluate metrics grouped by language, difficulty, etc.
        """
        if group_col not in df.columns:
            raise ValueError(f"Column {group_col} not found in dataframe.")

        groups = {}

        for key, subdf in df.groupby(group_col):
            groups[key] = self.evaluate(subdf)

        return groups

    # ======================================================
    # ---------------- VISUALIZATIONS ----------------------
    # ======================================================

    def plot_copy_rate(self, df):
        plt.figure(figsize=(6,4))
        sns.histplot((df["predicted_neutral"] == df["toxic"]).astype(int))
        plt.title("Copy Rate Distribution")
        plt.xlabel("Copy (1=yes)")
        plt.tight_layout()

    def plot_length_ratio(self, df):
        ratios = df["predicted_neutral"].str.split().str.len() / df["gold_neutral"].str.split().str.len()
        plt.figure(figsize=(6,4))
        sns.histplot(ratios, bins=30)
        plt.title("Length Ratio Distribution")
        plt.xlabel("Prediction Length / Gold Length")
        plt.tight_layout()

    def plot_semantic_vs_toxicity(self, df):
        """
        Useful scatter to analyze tradeoffs.
        """
        sims = self.sbert.encode(df["predicted_neutral"].tolist(), convert_to_tensor=True)
        golds = self.sbert.encode(df["gold_neutral"].tolist(), convert_to_tensor=True)
        cosims = util.pytorch_cos_sim(sims, golds).diag().cpu().numpy()

        tox = [self._toxicity_count(t) for t in df["predicted_neutral"]]

        plt.figure(figsize=(6,5))
        sns.scatterplot(x=cosims, y=tox)
        plt.xlabel("Semantic similarity (SBERT)")
        plt.ylabel("Toxicity count")
        plt.title("Semantic Preservation vs Residual Toxicity")
        plt.tight_layout()

    # ======================================================
    # ---------------- REPORT EXPORT -----------------------
    # ======================================================

    def generate_report(self, df, output_path="evaluation_report.md", plots_dir="evaluation_plots"):
        """
        Generate a full markdown report including:
        - global metrics
        - saved visualizations (png)
        - paths embedded in the markdown file
        """

        # Create directory for plots
        import os
        os.makedirs(plots_dir, exist_ok=True)

        # 1. Compute metrics
        metrics = self.evaluate(df)

        # 2. Generate and save plots
        plot_files = {}

        # ---- Copy Rate Histogram ----
        plt.figure(figsize=(6,4))
        sns.histplot((df["predicted_neutral"] == df["toxic"]).astype(int), bins=2)
        plt.title("Copy Rate Distribution")
        plt.xlabel("Copy (1 = identical to input)")
        plt.tight_layout()
        copy_plot_path = os.path.join(plots_dir, "copy_rate.png")
        plt.savefig(copy_plot_path)
        plt.close()
        plot_files["copy_rate"] = copy_plot_path

        # ---- Length Ratio Histogram ----
        ratios = df["predicted_neutral"].str.split().str.len() / df["gold_neutral"].str.split().str.len()
        plt.figure(figsize=(6,4))
        sns.histplot(ratios, bins=30)
        plt.title("Length Ratio Distribution")
        plt.xlabel("Prediction Length / Gold Length")
        plt.tight_layout()
        length_plot_path = os.path.join(plots_dir, "length_ratio.png")
        plt.savefig(length_plot_path)
        plt.close()
        plot_files["length_ratio"] = length_plot_path

        # ---- Semantic vs Toxicity scatter ----
        # compute SBERT sims
        sims = self.sbert.encode(df["predicted_neutral"].tolist(), convert_to_tensor=True)
        golds = self.sbert.encode(df["gold_neutral"].tolist(), convert_to_tensor=True)
        cosims = util.pytorch_cos_sim(sims, golds).diag().cpu().numpy()

        tox = []
        for idx, row in df.iterrows():
            lang = row["language"] if "language" in df.columns else None
            tox.append(self._toxicity_count(row["predicted_neutral"], lang))

        plt.figure(figsize=(6,5))
        sns.scatterplot(x=cosims, y=tox)
        plt.xlabel("Semantic similarity (SBERT)")
        plt.ylabel("Toxicity count")
        plt.title("Semantic Preservation vs Residual Toxicity")
        plt.tight_layout()
        sem_tox_path = os.path.join(plots_dir, "semantic_vs_toxicity.png")
        plt.savefig(sem_tox_path)
        plt.close()
        plot_files["semantic_vs_toxicity"] = sem_tox_path

        # 3. Build markdown
        md = "# Evaluation Report\n\n"
        md += "## Global Metrics\n\n"
        for k, v in metrics.items():
            md += f"- **{k}**: {v:.4f}\n"

        md += "\n---\n## Visualizations\n\n"

        md += "### Copy Rate Histogram\n"
        md += f"![Copy Rate]({plot_files['copy_rate']})\n\n"

        md += "### Length Ratio Distribution\n"
        md += f"![Length Ratio]({plot_files['length_ratio']})\n\n"

        md += "### Semantic Similarity vs Residual Toxicity\n"
        md += f"![Semantic vs Toxicity]({plot_files['semantic_vs_toxicity']})\n\n"

        # 4. Save markdown file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md)

        print(f"Report saved to {output_path}")
        print(f"Plots saved to {plots_dir}/")

        return metrics
