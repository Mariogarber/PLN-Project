import os
from typing import Dict, Any, Optional, List

import pandas as pd
import numpy as np

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

    Input dataframe expected columns:
        - toxic              (str)
        - gold_neutral       (str)
        - predicted_neutral  (str)
        - (optional) language
        - (optional) difficulty

    Computes:
        - Semantic similarity (BERTScore + SBERT)
        - Structural similarity (BLEU / chrF)
        - Copy-rate
        - Length ratios
        - Toxicity estimation (lexicon-based)
        - Toxicity Δ (input → prediction)
        - Semantic drift (toxic → prediction)
        - Grouped metrics by language / difficulty
        - Visualizations (global + per-language)
        - HTML report
    """

    def __init__(
        self,
        toxic_lexicons: Optional[Dict[str, List[str]]] = None,
        device: str = "cpu",
    ):
        """
        toxic_lexicons: dict mapping language -> list of toxic words
        """
        self.toxic_lexicons = toxic_lexicons or {}
        self.device = device

        # SBERT multilingual model
        self.sbert = SentenceTransformer(
            "sentence-transformers/distiluse-base-multilingual-cased-v2",
            device=device,
        )

        self.bleu_metric = BLEU()
        self.chrf_metric = CHRF()

    # ======================================================
    # ---------------- BASIC UTILITIES ---------------------
    # ======================================================

    def _compute_copy_rate(self, df: pd.DataFrame) -> float:
        return float(np.mean(df["toxic"] == df["predicted_neutral"]))

    def _toxicity_count(self, text: str, lang: Optional[str] = None) -> int:
        """
        Lexicon-based toxicity estimate.
        Counts toxic words in the given string (case-insensitive).
        """
        if not text:
            return 0

        if lang not in self.toxic_lexicons:
            # If no lexicon available for the language → zero (safe fallback)
            return 0

        lex = set(self.toxic_lexicons[lang])
        toks = text.lower().split()
        return sum(t in lex for t in toks)

    # ======================================================
    # ---------------- SEMANTIC METRICS --------------------
    # ======================================================

    def compute_bertscore(self, df: pd.DataFrame) -> float:
        """
        Compute BERTScore F1 between prediction and gold.
        """
        preds = df["predicted_neutral"].tolist()
        golds = df["gold_neutral"].tolist()

        P, R, F1 = bertscore(preds, golds, lang="multilingual", device=self.device)
        return float(F1.mean())

    def compute_sbert_similarity(self, df: pd.DataFrame) -> float:
        """
        SBERT cosine similarity between prediction and gold.
        """
        preds = df["predicted_neutral"].tolist()
        golds = df["gold_neutral"].tolist()

        emb_pred = self.sbert.encode(preds, convert_to_tensor=True)
        emb_gold = self.sbert.encode(golds, convert_to_tensor=True)

        sims = util.pytorch_cos_sim(emb_pred, emb_gold)
        return float(sims.diag().mean())

    # ======================================================
    # ---------------- STRUCTURAL METRICS ------------------
    # ======================================================

    def compute_bleu(self, df: pd.DataFrame) -> float:
        preds = [p for p in df["predicted_neutral"]]
        golds = [[g] for g in df["gold_neutral"]]
        return float(self.bleu_metric.corpus_score(preds, golds).score)

    def compute_chrf(self, df: pd.DataFrame) -> float:
        preds = [p for p in df["predicted_neutral"]]
        golds = [[g] for g in df["gold_neutral"]]
        return float(self.chrf_metric.corpus_score(preds, golds).score)

    def compute_length_ratio(self, df: pd.DataFrame) -> float:
        pred_len = df["predicted_neutral"].str.split().str.len()
        gold_len = df["gold_neutral"].str.split().str.len().replace(0, np.nan)
        ratios = pred_len / gold_len
        return float(ratios.replace([np.inf, -np.inf], np.nan).dropna().mean())

    # ======================================================
    # ---------------- DETOX METRICS -----------------------
    # ======================================================

    def compute_residual_toxicity(self, df: pd.DataFrame) -> float:
        """
        Lexicon-based toxicity count normalized by length.
        """
        scores = []
        for _, row in df.iterrows():
            lang = row["language"] if "language" in df.columns else None
            tox = self._toxicity_count(row["predicted_neutral"], lang=lang)
            length = max(1, len(row["predicted_neutral"].split()))
            scores.append(tox / length)
        return float(np.mean(scores))

    def compute_toxicity_delta(self, df: pd.DataFrame) -> float:
        """
        Toxicity(input) - Toxicity(prediction), averaged over dataset.
        Positive values → detoxification on average.
        """
        deltas = []
        for _, row in df.iterrows():
            lang = row["language"] if "language" in df.columns else None
            tox_in = self._toxicity_count(row["toxic"], lang=lang)
            tox_out = self._toxicity_count(row["predicted_neutral"], lang=lang)
            deltas.append(tox_in - tox_out)
        return float(np.mean(deltas))

    # ======================================================
    # ---------------- DRIFT METRICS -----------------------
    # ======================================================

    def compute_semantic_drift(self, df: pd.DataFrame) -> float:
        """
        SBERT similarity between toxic input and prediction.
        Too low → model rewrites too aggressively.
        Too high → model copies too much.
        """
        tox = df["toxic"].tolist()
        preds = df["predicted_neutral"].tolist()

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
        results: Dict[str, Any] = {}

        results["copy_rate"] = self._compute_copy_rate(df)
        results["bertscore_f1"] = self.compute_bertscore(df)
        results["sbert_similarity"] = self.compute_sbert_similarity(df)
        results["bleu"] = self.compute_bleu(df)
        results["chrf"] = self.compute_chrf(df)
        results["length_ratio"] = self.compute_length_ratio(df)
        results["residual_toxicity"] = self.compute_residual_toxicity(df)
        results["toxicity_delta"] = self.compute_toxicity_delta(df)
        results["semantic_drift"] = self.compute_semantic_drift(df)

        return results

    # ======================================================
    # ---------------- GROUPED EVALUATION ------------------
    # ======================================================

    def evaluate_by(self, df: pd.DataFrame, group_col: str) -> Dict[Any, Dict[str, Any]]:
        """
        Evaluate metrics grouped by language, difficulty, etc.
        """
        if group_col not in df.columns:
            raise ValueError(f"Column {group_col} not found in dataframe.")

        groups: Dict[Any, Dict[str, Any]] = {}
        for key, subdf in df.groupby(group_col):
            groups[key] = self.evaluate(subdf)

        return groups

    # ======================================================
    # ---------------- VISUALIZATION HELPERS ---------------
    # ======================================================

    def _plot_copy_rate(self, df: pd.DataFrame, save_path: str):
        plt.figure(figsize=(5, 4))
        sns.histplot((df["predicted_neutral"] == df["toxic"]).astype(int), bins=2)
        plt.title("Copy Rate Distribution")
        plt.xlabel("Copy (1 = identical to input)")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def _plot_length_ratio(self, df: pd.DataFrame, save_path: str):
        pred_len = df["predicted_neutral"].str.split().str.len()
        gold_len = df["gold_neutral"].str.split().str.len().replace(0, np.nan)
        ratios = pred_len / gold_len
        ratios = ratios.replace([np.inf, -np.inf], np.nan).dropna()

        plt.figure(figsize=(5, 4))
        sns.histplot(ratios, bins=30)
        plt.title("Length Ratio Distribution")
        plt.xlabel("Prediction Length / Gold Length")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def _plot_semantic_vs_toxicity(self, df: pd.DataFrame, save_path: str):
        preds = df["predicted_neutral"].tolist()
        golds = df["gold_neutral"].tolist()

        emb_pred = self.sbert.encode(preds, convert_to_tensor=True)
        emb_gold = self.sbert.encode(golds, convert_to_tensor=True)
        cosims = util.pytorch_cos_sim(emb_pred, emb_gold).diag().cpu().numpy()

        tox_counts = []
        for _, row in df.iterrows():
            lang = row["language"] if "language" in df.columns else None
            tox_counts.append(self._toxicity_count(row["predicted_neutral"], lang))

        plt.figure(figsize=(5, 4))
        sns.scatterplot(x=cosims, y=tox_counts)
        plt.xlabel("Semantic similarity (SBERT)")
        plt.ylabel("Toxicity count")
        plt.title("Semantic Preservation vs Residual Toxicity")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def _plot_metric_by_group(
        self,
        metrics_by_group: Dict[Any, Dict[str, Any]],
        metric_name: str,
        ylabel: str,
        title: str,
        save_path: str,
    ):
        groups = list(metrics_by_group.keys())
        values = [metrics_by_group[g][metric_name] for g in groups]

        plt.figure(figsize=(6, 4))
        sns.barplot(x=groups, y=values)
        plt.ylabel(ylabel)
        plt.xlabel("Group")
        plt.title(title)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    # ======================================================
    # ---------------- REPORT EXPORT (HTML) ----------------
    # ======================================================

    def generate_report(
        self,
        df: pd.DataFrame,
        output_path: str = "evaluation_report.html",
        plots_dir: str = "evaluation_plots",
        include_by_language: bool = True,
        include_by_difficulty: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate a full HTML report including:
            - Global metrics
            - Metrics grouped by language / difficulty (if available)
            - Global visualizations
            - Visualizations by language (metrics barplots)

        Returns:
            Dictionary with global metrics.
        """

        os.makedirs(plots_dir, exist_ok=True)

        # ------------------ METRICS -----------------------
        global_metrics = self.evaluate(df)

        metrics_by_language = {}
        metrics_by_difficulty = {}

        if include_by_language and "language" in df.columns:
            metrics_by_language = self.evaluate_by(df, "language")

        if include_by_difficulty and "difficulty" in df.columns:
            metrics_by_difficulty = self.evaluate_by(df, "difficulty")

        # ------------------ PLOTS (GLOBAL) ----------------
        plot_files = {}

        copy_plot = os.path.join(plots_dir, "copy_rate.png")
        self._plot_copy_rate(df, copy_plot)
        plot_files["copy_rate"] = copy_plot

        length_plot = os.path.join(plots_dir, "length_ratio.png")
        self._plot_length_ratio(df, length_plot)
        plot_files["length_ratio"] = length_plot

        sem_tox_plot = os.path.join(plots_dir, "semantic_vs_toxicity.png")
        self._plot_semantic_vs_toxicity(df, sem_tox_plot)
        plot_files["semantic_vs_toxicity"] = sem_tox_plot

        # ------------------ PLOTS (BY LANGUAGE) -----------
        lang_plot_files: Dict[str, str] = {}
        if metrics_by_language:
            for metric in ["copy_rate", "residual_toxicity", "toxicity_delta", "semantic_drift"]:
                path = os.path.join(plots_dir, f"by_language_{metric}.png")
                self._plot_metric_by_group(
                    metrics_by_language,
                    metric_name=metric,
                    ylabel=metric,
                    title=f"{metric} by language",
                    save_path=path,
                )
                lang_plot_files[metric] = path

        # ------------------ PLOTS (BY DIFFICULTY) ---------
        diff_plot_files: Dict[str, str] = {}
        if metrics_by_difficulty:
            for metric in ["copy_rate", "residual_toxicity", "toxicity_delta", "semantic_drift"]:
                path = os.path.join(plots_dir, f"by_difficulty_{metric}.png")
                self._plot_metric_by_group(
                    metrics_by_difficulty,
                    metric_name=metric,
                    ylabel=metric,
                    title=f"{metric} by difficulty",
                    save_path=path,
                )
                diff_plot_files[metric] = path

        # ------------------ HTML REPORT -------------------
        def metrics_dict_to_html_table(title: str, metrics: Dict[str, Any]) -> str:
            rows = ""
            for k, v in metrics.items():
                if isinstance(v, (float, int)):
                    cell = f"{v:.4f}"
                else:
                    cell = str(v)
                rows += f"<tr><td>{k}</td><td>{cell}</td></tr>"

            return f"""
            <h2>{title}</h2>
            <table border="1" cellpadding="4" cellspacing="0">
              <thead><tr><th>Metric</th><th>Value</th></tr></thead>
              <tbody>{rows}</tbody>
            </table>
            """

        def grouped_metrics_to_html(title: str, grouped: Dict[Any, Dict[str, Any]]) -> str:
            if not grouped:
                return ""

            metric_names = list(next(iter(grouped.values())).keys())

            # Cabecera
            header_cells = "".join(f"<th>{m}</th>" for m in metric_names)
            header = f"<tr><th>Group</th>{header_cells}</tr>"

            # Filas
            body_rows = ""
            for g, vals in grouped.items():
                row_cells = ""
                for m in metric_names:
                    val = vals[m]
                    if isinstance(val, (float, int)):
                        cell = f"{val:.4f}"
                    else:
                        cell = str(val)
                    row_cells += f"<td>{cell}</td>"
                body_rows += f"<tr><td>{g}</td>{row_cells}</tr>"

            return f"""
            <h2>{title}</h2>
            <table border="1" cellpadding="4" cellspacing="0">
              <thead>{header}</thead>
              <tbody>{body_rows}</tbody>
            </table>
            """


        html = [
            "<html><head><meta charset='utf-8'><title>Detox Evaluation Report</title>",
            "<style>body{font-family:Arial, sans-serif; margin:20px;} "
            "table{border-collapse:collapse; margin-bottom:20px;} "
            "th,td{padding:4px 8px;} h1,h2,h3{margin-top:24px;}</style>",
            "</head><body>",
            "<h1>Detox Evaluation Report</h1>",
        ]

        # Global
        html.append(metrics_dict_to_html_table("Global metrics", global_metrics))

        # By language
        if metrics_by_language:
            html.append(grouped_metrics_to_html("Metrics by language", metrics_by_language))

        # By difficulty
        if metrics_by_difficulty:
            html.append(grouped_metrics_to_html("Metrics by difficulty", metrics_by_difficulty))

        # Global plots
        html.append("<h2>Global Visualizations</h2>")
        html.append(f"<h3>Copy rate distribution</h3><img src='{plot_files['copy_rate']}' width='400'>")
        html.append(f"<h3>Length ratio distribution</h3><img src='{plot_files['length_ratio']}' width='400'>")
        html.append(
            f"<h3>Semantic similarity vs residual toxicity</h3>"
            f"<img src='{plot_files['semantic_vs_toxicity']}' width='400'>"
        )

        # By language plots
        if lang_plot_files:
            html.append("<h2>Visualizations by language</h2>")
            for metric, path in lang_plot_files.items():
                html.append(f"<h3>{metric} by language</h3><img src='{path}' width='500'>")

        # By difficulty plots
        if diff_plot_files:
            html.append("<h2>Visualizations by difficulty</h2>")
            for metric, path in diff_plot_files.items():
                html.append(f"<h3>{metric} by difficulty</h3><img src='{path}' width='500'>")

        html.append("</body></html>")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(html))

        print(f"[Evaluator] HTML report saved to: {output_path}")
        print(f"[Evaluator] Plots saved under: {plots_dir}/")

        return global_metrics

import os
import io
import base64
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from typing import Optional, Dict, Any

from .metrics.similarity import SimilarityMeasurement
from .metrics.toxicity import ToxicityMeasurement
from .metrics.fluency.xcomet import CometFluency


class OfficialEvaluator:
    """
    Official evaluator for detoxification systems.
    Computes SIM, STA, XCOMET and final score J = SIM * STA * XCOMET.
    Produces an HTML report with plots and aggregated statistics.
    """

    def __init__(
        self,
        similarity_module: Optional[SimilarityMeasurement] = None,
        toxicity_module: Optional[ToxicityMeasurement] = None,
        fluency_module: Optional[CometFluency] = None,
    ):
        self.sim = similarity_module or SimilarityMeasurement()
        self.tox = toxicity_module or ToxicityMeasurement()
        self.fluency = fluency_module or CometFluency()

    # --------------------------------------------------------
    # Utility: convert figure to base64 HTML <img>
    # --------------------------------------------------------
    @staticmethod
    def _fig_to_html(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)
        return f'<img src="data:image/png;base64,{img_b64}" width="600"/>'

    # --------------------------------------------------------
    # Compute official metrics
    # --------------------------------------------------------
    def compute_metrics(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()

        toxic   = df["toxic_sentence"].tolist()
        pred    = df["generated_sentence"].tolist()
        gold    = df["neutral_sentence"].tolist()

        # SIM -----------------------
        sim_scores = self.sim.evaluate_similarity(
            original_texts=toxic,
            rewritten_texts=pred,
            reference_texts=gold,
        )

        # STA -----------------------
        sta_scores = self.tox.compare_toxicity(
            original_texts=toxic,
            rewritten_texts=pred,
            reference_texts=gold,
        )

        # XCOMET --------------------
        comet_input = [
            {"src": t, "mt": p, "ref": r}
            for t, p, r in zip(toxic, pred, gold)
        ]
        xcomet_scores = self.fluency.get_scores(
            input_data=comet_input,
            batch_size=256,
        )

        # Store results
        df["SIM"] = sim_scores
        df["STA"] = sta_scores
        df["XCOMET"] = xcomet_scores
        df["J"] = df["SIM"] * df["STA"] * df["XCOMET"]

        return df

    # --------------------------------------------------------
    # Plot utilities
    # --------------------------------------------------------
    def _plot_hist(self, series, title):
        fig = plt.figure(figsize=(6, 4))
        plt.hist(series, bins=30, alpha=0.7, color="steelblue")
        plt.title(title)
        return fig

    def _plot_scatter(self, x, y, title, xlabel, ylabel):
        fig = plt.figure(figsize=(6, 4))
        plt.scatter(x, y, alpha=0.4)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        return fig

    def _plot_box(self, df, column, group):
        fig = plt.figure(figsize=(7, 4))
        df.boxplot(column=column, by=group, grid=False)
        plt.title(f"{column} by {group}")
        plt.suptitle("")
        return fig

    # --------------------------------------------------------
    # Report generator
    # --------------------------------------------------------
    def generate_report(self, df: pd.DataFrame, output_path: str):

        html = [
            "<html><head><meta charset='utf-8'>",
            "<title>Official Detox Evaluation Report</title></head><body>",
            "<h1>Official Detox System Evaluation</h1>",
        ]

        # Global summary
        html.append("<h2>Global Metric Summary</h2>")
        html.append(df.describe().T.to_html())

        # Language summary
        if "language" in df.columns:
            summary_lang = df.groupby("language")[["SIM","STA","XCOMET","J"]].mean()
            html.append("<h2>Scores by Language</h2>")
            html.append(summary_lang.to_html())

        # Histograms
        html.append("<h2>Distributions</h2>")
        for col in ["SIM", "STA", "XCOMET", "J"]:
            html.append(f"<h3>{col} Distribution</h3>")
            html.append(self._fig_to_html(self._plot_hist(df[col], f"{col} Histogram")))

        # Scatterplots
        html.append("<h2>Metric Interactions</h2>")
        html.append("<h3>SIM vs STA</h3>")
        html.append(self._fig_to_html(
            self._plot_scatter(df["SIM"], df["STA"], "SIM vs STA", "SIM", "STA")
        ))

        html.append("<h3>SIM vs XCOMET</h3>")
        html.append(self._fig_to_html(
            self._plot_scatter(df["SIM"], df["XCOMET"], "SIM vs XCOMET", "SIM", "XCOMET")
        ))

        html.append("<h3>STA vs XCOMET</h3>")
        html.append(self._fig_to_html(
            self._plot_scatter(df["STA"], df["XCOMET"], "STA vs XCOMET", "STA", "XCOMET")
        ))

        # Boxplots by language
        if "language" in df.columns:
            html.append("<h2>Per-language breakdown</h2>")
            for col in ["SIM", "STA", "XCOMET", "J"]:
                html.append(f"<h3>{col} by language</h3>")
                html.append(self._fig_to_html(
                    self._plot_box(df, col, "language")
                ))

        html.append("</body></html>")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(html))

        print(f"✓ Official report saved to {output_path}")
