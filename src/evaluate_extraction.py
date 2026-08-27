"""Score both extractors against the hand-labeled headlines.

Scoring is at the headline level: did the extractor detect a relationship in
this headline, and if so, is the relation type right? Firm-level matching is
not used because the labels name companies as written while the
rules extractor can only emit universe tickers, the two are not comparable
at that granularity, and the gap is itself one of the results.

Two panels, because the 45 labels come from two samples that answer
different questions:
  random 30  — headlines drawn at random from multi-firm articles. All 30 are
               labeled 'none', so this measures FALSE POSITIVES on noise.
  keyword 15 — headlines pre-filtered for relation keywords, 7 of which are
               real. This measures whether real relations are found.
"""

import re

import pandas as pd

from data import ROOT
from extraction_rules import (classify, find_firm_mentions, is_junk,
                              load_name_map, pair_firms)
from resolve import load_llm_edges

LABEL_FILES = [ROOT / "labels" / "headline_sample_30.csv",
               ROOT / "labels" / "headline_sample_keyword15.csv"]


def load_labels() -> pd.DataFrame:
    labels = pd.concat([pd.read_csv(f) for f in LABEL_FILES], ignore_index=True)
    labels["label"] = labels["label"].astype(str).str.strip()
    labels["has_relation"] = labels["label"].str.lower() != "none"
    labels["true_type"] = labels["label"].str.split(":").str[0].where(
        labels["has_relation"])
    labels["panel"] = ["random 30"] * 30 + ["keyword 15"] * 15
    return labels


def rules_prediction(headline: str) -> str | None:
    """Relation type the rules extractor would emit, or None."""
    if not isinstance(headline, str) or is_junk(headline):
        return None
    relation = classify(headline)
    if relation is None:
        return None
    pairs = pair_firms(headline, find_firm_mentions(headline, load_name_map()))
    return relation if pairs else None


def llm_predictions() -> dict[str, str]:
    """headline -> relation type found by the LLM extractor."""
    edges = load_llm_edges()
    return dict(zip(edges["headline"], edges["relation"]))


def score(labels: pd.DataFrame, predicted: pd.Series) -> dict:
    """Detection precision/recall/F1 plus type accuracy on true positives."""
    truth = labels["has_relation"]
    found = predicted.notna()
    tp = int((truth & found).sum())
    fp = int((~truth & found).sum())
    fn = int((truth & ~found).sum())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if tp and precision + recall else 0.0)
    both = truth & found
    type_ok = int((labels.loc[both, "true_type"] == predicted[both]).sum())
    return {"found": int(found.sum()), "TP": tp, "FP": fp, "FN": fn,
            "precision": precision, "recall": recall, "f1": f1,
            "type correct": f"{type_ok}/{tp}" if tp else "-"}


def main():
    labels = load_labels()
    llm_map = llm_predictions()

    preds = {
        "rules": labels["title"].apply(rules_prediction),
        "llm": labels["title"].map(llm_map),
    }

    rows = []
    for name, pred in preds.items():
        for panel in ["random 30", "keyword 15", "all 45"]:
            mask = (labels["panel"] == panel) if panel != "all 45" else slice(None)
            rows.append({"extractor": name, "panel": panel,
                         **score(labels[mask], pred[mask])})
    table = pd.DataFrame(rows)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    print("\nheadlines where the two extractors disagree:")
    disagree = labels[preds["rules"].notna() != preds["llm"].notna()]
    for i, row in disagree.iterrows():
        r, l = preds["rules"][i], preds["llm"][i]
        print(f"  truth={row['label'][:28]:28s} rules={str(r):12s} llm={str(l):12s}"
              f" | {row['title'][:70]}")

    table.to_csv(ROOT / "reports" / "extraction_scores.csv", index=False)
    print(f"\nsaved to reports/extraction_scores.csv")


if __name__ == "__main__":
    main()
