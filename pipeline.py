from typing import List, Tuple, Dict
import pandas as pd
from modules.base import DuplicationModule
from modules.crossencoder import CrossEncoderModule


class Pipeline:

    def __init__(self, modules: List[DuplicationModule]):
        self.modules = modules

    def run(self, df: pd.DataFrame) -> List[Tuple[int, int, float]]:
        pairs = None

        for module in self.modules:
            print(f"{'='*60}")

            if pairs is None:
                pairs = module.compute_pairs(df)
                print(f"Generated {len(pairs):,} pairs")
            else:
                if isinstance(module, CrossEncoderModule):
                    pairs = module.rerank_pairs(pairs, df)
                else:
                    reranked = []
                    for idx1, idx2, old_score in pairs:
                        new_score = module.predict_pair(
                            df.loc[idx1, 'text'],
                            df.loc[idx2, 'text']
                        )
                        if new_score > module.threshold:
                            reranked.append((idx1, idx2, new_score))
                    pairs = reranked
                    print(f"After {module.name}: {len(pairs):,} pairs remain")

        return pairs

    def __repr__(self):
        module_names = [m.name for m in self.modules]
        return f"Pipeline({' -> '.join(module_names)})"
