"""
Adapters for converting various QA datasets to MuSiQue-like JSONL format.
Each adapter inherits from BaseDatasetAdapter.
"""

from .base_adapter import BaseDatasetAdapter
from .pubmedqa_adapter import PubMedQAAdapter
from .belebele_adapter import BelebeleAdapter
from .bioasq_adapter import BioASQAdapter
from .popqa_adapter import PopQAAdapter
from .nq_rear_adapter import NQRearAdapter
from .narrativeqa_adapter import NarrativeQAAdapter
from .hotpotqa_adapter import HotpotQAAdapter
from .twowiki_adapter import TwoWikiMultihopQAAdapter
from .beir_adapter import (
    NFCorpusAdapter, SciFactAdapter, QuoraAdapter,
    TRECCOVIDAdapter, DBPediaAdapter,
)
from ._stubs import (
    SciDQAAdapter,
    DropAdapter,
    MultiMedQAAdapter,
    DocFinQAAdapter,
    MedReadMeAdapter,
    CflueAdapter,
)


ADAPTER_REGISTRY = {
    "pubmedqa": PubMedQAAdapter,
    "scidqa": SciDQAAdapter,
    "drop": DropAdapter,
    "belebele": BelebeleAdapter,
    "bioasq": BioASQAdapter,
    "multimedqa": MultiMedQAAdapter,
    "docfinqa": DocFinQAAdapter,
    "medreadme": MedReadMeAdapter,
    "cflue": CflueAdapter,
    "popqa": PopQAAdapter,
    "nq_rear": NQRearAdapter,
    "narrativeqa": NarrativeQAAdapter,
    "hotpotqa": HotpotQAAdapter,
    "2wikimultihopqa": TwoWikiMultihopQAAdapter,
    "nfcorpus": NFCorpusAdapter,
    "scifact": SciFactAdapter,
    "quora": QuoraAdapter,
    "trec-covid": TRECCOVIDAdapter,
    "trec_covid": TRECCOVIDAdapter,
    "dbpedia-entity": DBPediaAdapter,
    "dbpedia": DBPediaAdapter,
    "sf_custom": lambda **kwargs: type('SFCustomAdapter', (object,), {
        'dataset_name': 'sf_custom',
        'display_name': 'SF Custom Corpus',
        'default_subset': 'test',
        'get_recommended_params': lambda self: {
            'grid_size': 64, 'spreading_steps': 1, 'top_percent': 0.10,
            'weighting': 'idf', 'smoothing_sigma': 1.5, 'morton': True,
            'min_word_length': 3, 'min_freq': 1, 'keep_verbs': True, 'top_k': 10,
            'tsne_perplexity': 50, 'tsne_iter': 1000,
        },
    })(),
}


def get_adapter(name: str, **kwargs) -> BaseDatasetAdapter:
    """Factory: instantiate the right adapter by dataset name."""
    if name not in ADAPTER_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {list(ADAPTER_REGISTRY)}"
        )
    return ADAPTER_REGISTRY[name](**kwargs)


__all__ = [
    "BaseDatasetAdapter",
    "get_adapter",
    "ADAPTER_REGISTRY",
    "PubMedQAAdapter",
    "SciDQAAdapter",
    "DropAdapter",
    "BelebeleAdapter",
    "BioASQAdapter",
    "MultiMedQAAdapter",
    "DocFinQAAdapter",
    "MedReadMeAdapter",
    "CflueAdapter",
    "PopQAAdapter",
    "NQRearAdapter",
    "NarrativeQAAdapter",
    "HotpotQAAdapter",
    "TwoWikiMultihopQAAdapter",
    "NFCorpusAdapter",
    "SciFactAdapter",
    "QuoraAdapter",
    "TRECCOVIDAdapter",
    "DBPediaAdapter",
]
