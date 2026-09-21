"""
================================================================================
 事件管道 — 舆情事件聚类、时间线对齐
 公告文本 → Embedding → UMAP降维 → HDBSCAN聚类 → LLM标签 → 事件簇
================================================================================
"""
import json
import os
import re
from datetime import datetime
from collections import defaultdict
from typing import Optional

import pandas as pd
import numpy as np


# ============================================================================
# 事件簇数据结构
# ============================================================================

class EventCluster:
    """一个事件簇（一组语义相似的公告/新闻）"""

    def __init__(self, cluster_id: int, label: str = ""):
        self.cluster_id = cluster_id
        self.label = label                          # LLM 生成的标签
        self.keywords: list[str] = []               # TF-IDF top 关键词
        self.doc_indices: list[int] = []            # 属于该簇的文档索引
        self.doc_titles: list[str] = []             # 代表性文档标题
        self.doc_dates: list[datetime] = []         # 文档日期
        self.stock_codes: set[str] = set()          # 涉及的股票代码
        self.size: int = 0
        self.time_span: Optional[tuple[datetime, datetime]] = None

    def to_dict(self) -> dict:
        return {
            "cluster_id": self.cluster_id,
            "label": self.label,
            "keywords": self.keywords,
            "size": self.size,
            "sample_titles": self.doc_titles[:5],
            "time_span": [
                self.time_span[0].isoformat() if self.time_span and self.time_span[0] else None,
                self.time_span[1].isoformat() if self.time_span and self.time_span[1] else None,
            ],
            "stock_count": len(self.stock_codes),
            "top_stocks": list(self.stock_codes)[:10],
        }


# ============================================================================
# 特征提取
# ============================================================================

def extract_text_features(df: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    """
    从公告/研报 DataFrame 中提取文本特征
    优先使用 LLM API embedding，回退到 TF-IDF
    """
    texts = []
    for _, row in df.iterrows():
        # 优先用 title，可以拼接摘要
        title = str(row.get('n_info_title', row.get('title', '')))
        texts.append(title)

    # 尝试 LLM embedding
    try:
        from llm_client import embed
        print(f"[EventPipeline] 使用 LLM API 生成 {len(texts)} 个文本的 embedding...")
        vectors = embed(texts)
        return texts, np.array(vectors)
    except Exception as e:
        print(f"[EventPipeline] LLM embedding 不可用 ({e})，回退到 TF-IDF")
        return _tfidf_embed(texts)


def _tfidf_embed(texts: list[str], max_features: int = 256) -> tuple[list[str], np.ndarray]:
    """TF-IDF 回退方案"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        analyzer='char_wb',  # 字符级，对中文友好
    )
    vectors = vectorizer.toarray()
    return texts, vectors


# ============================================================================
# 聚类
# ============================================================================

def cluster_events(
    vectors: np.ndarray,
    texts: list[str],
    min_cluster_size: int = 3,
    metric: str = "cosine",
) -> list[EventCluster]:
    """
    用 HDBSCAN 做聚类；如果不可用回退到简单相似度聚类
    """
    try:
        import hdbscan
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric=metric,
            cluster_selection_method='eom',
        )
        labels = clusterer.fit_predict(vectors)
        print(f"[EventPipeline] HDBSCAN: 发现 {len(set(labels)) - (1 if -1 in labels else 0)} 个簇, "
              f"噪声点: {sum(1 for l in labels if l == -1)}")
    except ImportError:
        print("[EventPipeline] HDBSCAN 不可用，使用 UMAP + 简单聚类")
        labels = _fallback_cluster(vectors, min_cluster_size)

    # 按簇分组
    clusters = defaultdict(list)
    for idx, label in enumerate(labels):
        clusters[int(label)].append(idx)

    # 构建 EventCluster 对象
    result = []
    for label, indices in clusters.items():
        if label == -1:  # 噪声
            continue
        ec = EventCluster(cluster_id=label)
        ec.doc_indices = indices
        ec.size = len(indices)
        ec.doc_titles = [texts[i] for i in indices]
        result.append(ec)

    return result


def _fallback_cluster(vectors: np.ndarray, min_size: int = 3) -> np.ndarray:
    """回退聚类方案：基于余弦相似度的简单层次聚类"""
    from sklearn.metrics.pairwise import cosine_similarity

    n = len(vectors)
    sim = cosine_similarity(vectors)
    labels = np.full(n, -1, dtype=int)
    visited = np.zeros(n, dtype=bool)

    cluster_id = 0
    for i in range(n):
        if visited[i]:
            continue
        # 找与 i 相似度 > 0.7 的点
        neighbors = [i]
        for j in range(i + 1, n):
            if not visited[j] and sim[i, j] > 0.7:
                neighbors.append(j)
        if len(neighbors) >= min_size:
            for idx in neighbors:
                labels[idx] = cluster_id
                visited[idx] = True
            cluster_id += 1

    return labels


# ============================================================================
# 关键词提取
# ============================================================================

def extract_keywords(texts: list[str], top_k: int = 10) -> list[str]:
    """用 TF-IDF 提取关键词"""
    from sklearn.feature_extraction.text import TfidfVectorizer

    if len(texts) < 2:
        return []

    vectorizer = TfidfVectorizer(
        max_features=100,
        ngram_range=(1, 2),
        analyzer='char_wb',
        max_df=0.8,
        min_df=2,
    )
    try:
        tfidf = vectorizer.fit_transform(texts)
        scores = np.asarray(tfidf.sum(axis=0)).flatten()
        indices = scores.argsort()[-top_k:][::-1]
        feature_names = vectorizer.get_feature_names_out()
        return [feature_names[i] for i in indices if scores[i] > 0]
    except Exception:
        return []


# ============================================================================
# LLM 标签生成
# ============================================================================

def generate_cluster_labels(
    clusters: list[EventCluster],
    use_llm: bool = True,
) -> list[EventCluster]:
    """
    为每个事件簇生成人类可读的标签
    如果 use_llm=True，用 LLM 生成；否则只用关键词
    """
    for ec in clusters:
        # 先用 TF-IDF 提取关键词
        ec.keywords = extract_keywords(ec.doc_titles, top_k=8)

        if use_llm and len(ec.doc_titles) >= 2:
            try:
                from llm_client import chat
                sample_texts = ec.doc_titles[:5]
                prompt = (
                    "你是一个金融事件分析专家。以下是一组语义相似的上市公司公告标题，"
                    "请为这组事件生成一个简洁的标签（不超过12个字），描述它们共同的主题。\n\n"
                    "公告标题：\n"
                )
                for i, t in enumerate(sample_texts, 1):
                    prompt += f"{i}. {t}\n"
                prompt += "\n关键词参考：" + "、".join(ec.keywords[:5])
                prompt += "\n\n只输出标签文本，不要额外解释。"

                response = chat([
                    {"role": "system", "content": "你是一个金融文本分析专家。只输出简短标签。"},
                    {"role": "user", "content": prompt},
                ], temperature=0.3, max_tokens=50)

                ec.label = response.strip().strip('"''「」')
            except Exception as e:
                ec.label = " / ".join(ec.keywords[:3]) if ec.keywords else f"事件簇{ec.cluster_id}"

        if not ec.label:
            ec.label = " / ".join(ec.keywords[:3]) if ec.keywords else f"事件簇{ec.cluster_id}"

    return clusters


# ============================================================================
# 时间线构建
# ============================================================================

class EventTimeline:
    """
    单只股票或一组股票的事件时间线
    对齐股权变更事件和舆情事件
    """

    def __init__(self, stock_code: str):
        self.stock_code = stock_code
        self.events: list[dict] = []  # [{date, type, description, cluster_id, ...}]

    def add_equity_event(self, date: datetime, description: str, metadata: dict = None):
        self.events.append({
            "date": date,
            "type": "股权变更",
            "description": description,
            "metadata": metadata or {},
        })

    def add_news_event(self, date: datetime, title: str, cluster_id: int, cluster_label: str):
        self.events.append({
            "date": date,
            "type": "舆情事件",
            "description": title,
            "cluster_id": cluster_id,
            "cluster_label": cluster_label,
        })

    def build(self) -> list[dict]:
        """按时间排序并返回完整时间线"""
        self.events.sort(key=lambda x: x["date"] if x["date"] else datetime.min)
        return self.events

    def to_text(self) -> str:
        """生成 LLM 可读的时间线文本"""
        lines = [f"# {self.stock_code} 事件时间线", ""]
        for evt in self.build():
            date_str = evt["date"].strftime("%Y-%m-%d") if evt["date"] else "未知日期"
            type_tag = f"[{evt['type']}]"
            desc = evt["description"]
            if evt.get("cluster_label"):
                desc += f" (事件簇: {evt['cluster_label']})"
            lines.append(f"{date_str} {type_tag} {desc}")
        return "\n".join(lines)


def build_timeline_for_stock(
    stock_code: str,
    announcements_df: pd.DataFrame,
    clusters: list[EventCluster],
    equity_events: list[dict] = None,
) -> EventTimeline:
    """
    为一只股票构建对齐的时间线
    """
    timeline = EventTimeline(stock_code)

    # 1. 股权变更事件
    if equity_events:
        for evt in equity_events:
            timeline.add_equity_event(
                evt.get("date", datetime.min),
                evt.get("description", ""),
                evt.get("metadata", {}),
            )

    # 2. 相关公告（按股票代码匹配）
    stock_announcements = announcements_df[
        announcements_df['s_info_windcode'].astype(str).str.contains(
            stock_code.replace('.SH', '').replace('.SZ', '').replace('.BJ', ''),
            na=False
        )
    ]

    # 为每条公告找到所属的簇
    for _, row in stock_announcements.iterrows():
        title = str(row.get('n_info_title', ''))
        date = row.get('ann_date', None)
        if isinstance(date, str):
            try:
                date = pd.to_datetime(date)
            except:
                date = None

        # 查找该标题属于哪个簇
        found_cluster = None
        for ec in clusters:
            if title in ec.doc_titles:
                found_cluster = ec
                break

        timeline.add_news_event(
            date if date and not pd.isna(date) else datetime.min,
            title,
            found_cluster.cluster_id if found_cluster else -1,
            found_cluster.label if found_cluster else "",
        )

    return timeline


# ============================================================================
# 完整管道
# ============================================================================

class EventPipeline:
    """
    完整的事件处理管道
    """

    def __init__(self):
        self.clusters: list[EventCluster] = []
        self.announcements_df: Optional[pd.DataFrame] = None
        self.timelines: dict[str, EventTimeline] = {}

    def fit(
        self,
        announcements_df: pd.DataFrame,
        use_llm_labels: bool = True,
        min_cluster_size: int = 3,
    ) -> list[EventCluster]:
        """
        完整管道：文本特征 → embedding → 聚类 → LLM 标签

        Returns:
            标注好的事件簇列表
        """
        print(f"[EventPipeline] 开始处理 {len(announcements_df)} 条公告...")
        self.announcements_df = announcements_df

        # 1. 提取特征
        texts, vectors = extract_text_features(announcements_df)

        # 2. 聚类
        self.clusters = cluster_events(vectors, texts, min_cluster_size=min_cluster_size)

        # 3. 填充时间、股票等元信息
        for ec in self.clusters:
            dates = []
            for idx in ec.doc_indices:
                row = announcements_df.iloc[idx]
                # 日期
                date_val = row.get('ann_date', None)
                if date_val and not pd.isna(date_val):
                    if isinstance(date_val, str):
                        try:
                            date_val = pd.to_datetime(date_val)
                        except:
                            date_val = None
                    if date_val:
                        dates.append(date_val.to_pydatetime() if hasattr(date_val, 'to_pydatetime') else date_val)
                # 股票代码
                stock = str(row.get('s_info_windcode', ''))
                if stock and stock != 'nan':
                    ec.stock_codes.add(stock)

            ec.doc_dates = dates
            if dates:
                ec.time_span = (min(dates), max(dates))

        # 4. LLM 标签
        self.clusters = generate_cluster_labels(self.clusters, use_llm=use_llm_labels)

        print(f"[EventPipeline] 处理完成: {len(self.clusters)} 个事件簇")
        for ec in sorted(self.clusters, key=lambda x: -x.size)[:5]:
            print(f"  簇{ec.cluster_id}: [{ec.label}] ({ec.size}条, {len(ec.stock_codes)}只股票)")

        return self.clusters

    def get_timeline(self, stock_code: str, equity_events: list[dict] = None) -> EventTimeline:
        """获取单只股票的事件时间线"""
        if self.announcements_df is None:
            raise ValueError("请先调用 fit() 处理公告数据")

        timeline = build_timeline_for_stock(
            stock_code, self.announcements_df, self.clusters, equity_events
        )
        self.timelines[stock_code] = timeline
        return timeline

    def save(self, path: str):
        """保存聚类结果"""
        os.makedirs(path, exist_ok=True)
        data = {
            "clusters": [c.to_dict() for c in self.clusters],
            "cluster_count": len(self.clusters),
        }
        with open(os.path.join(path, "event_clusters.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[EventPipeline] 已保存 {len(self.clusters)} 个簇到 {path}")

    @classmethod
    def load(cls, path: str) -> "EventPipeline":
        """加载聚类结果"""
        obj = cls()
        with open(os.path.join(path, "event_clusters.json"), "r", encoding="utf-8") as f:
            data = json.load(f)
        # 简化版：只恢复基本结构
        for cd in data.get("clusters", []):
            ec = EventCluster(cd["cluster_id"], cd.get("label", ""))
            ec.keywords = cd.get("keywords", [])
            ec.size = cd.get("size", 0)
            ec.doc_titles = cd.get("sample_titles", [])
            obj.clusters.append(ec)
        print(f"[EventPipeline] 已加载 {len(obj.clusters)} 个簇")
        return obj


# ============================================================================
# 快速测试
# ============================================================================

if __name__ == "__main__":
    print("=== EventPipeline 测试 ===\n")

    # 模拟公告数据
    mock_data = pd.DataFrame({
        's_info_windcode': ['600519.SH', '600519.SH', '000858.SZ', '000858.SZ',
                            '601318.SH', '601318.SH', '600036.SH', '600036.SH'],
        'n_info_title': [
            '贵州茅台:关于收到中国证监会行政处罚决定书的公告',
            '贵州茅台:关于公司及相关人员收到证监局警示函的公告',
            '五粮液:关于最近五年被证券监管部门和交易所处罚或采取监管措施的公告',
            '五粮液:关于收到四川证监局行政监管措施决定书的公告',
            '中国平安:2023年度报告',
            '中国平安:2023年度利润分配方案公告',
            '招商银行:关于副行长辞职的公告',
            '招商银行:关于聘任新任行长的公告',
        ],
        'ann_date': pd.to_datetime(['2023-06-15', '2023-08-20', '2023-05-10',
                                     '2023-07-01', '2024-03-15', '2024-03-15',
                                     '2023-09-01', '2023-09-15']),
    })

    pipeline = EventPipeline()
    clusters = pipeline.fit(mock_data, use_llm_labels=False)

    print("\n=== 聚类结果 ===")
    for ec in sorted(clusters, key=lambda x: -x.size):
        print(f"  簇{ec.cluster_id}: [{ec.label}]")
        print(f"    大小: {ec.size}, 关键词: {ec.keywords[:5]}")
        print(f"    样本: {ec.doc_titles[:3]}")
        print(f"    股票: {list(ec.stock_codes)}")
        print()

    # 测试时间线
    timeline = pipeline.get_timeline("600519")
    print("=== 600519 时间线 ===")
    print(timeline.to_text())
