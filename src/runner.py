import os
import uuid
from typing import List, Dict, Optional

import pandas as pd
from autorag.deploy import GradioRunner
from autorag.deploy.api import RetrievedPassage
from autorag.nodes.generator.base import BaseGenerator
from autorag.utils import fetch_contents

empty_retrieved_passage = RetrievedPassage(
    content="", doc_id="", filepath=None, file_page=None, start_idx=None, end_idx=None
)


class GradioStreamRunner(GradioRunner):
    def __init__(self, config: Dict, project_dir: Optional[str] = None):
        super().__init__(config, project_dir)

        data_dir = os.path.join(project_dir, "data")
        self.corpus_df = pd.read_parquet(
            os.path.join(data_dir, "corpus.parquet"), engine="pyarrow"
        )

    def stream_run(self, query: str):
        previous_result = pd.DataFrame(
            {
                "qid": str(uuid.uuid4()),
                "query": [query],
                "retrieval_gt": [[]],
                "generation_gt": [""],
            }
        )  # pseudo qa data for execution

        for module_instance, module_param in zip(
                self.module_instances, self.module_params
        ):
            if not isinstance(module_instance, BaseGenerator):
                new_result = module_instance.pure(
                    previous_result=previous_result, **module_param
                )
                duplicated_columns = previous_result.columns.intersection(
                    new_result.columns
                )
                drop_previous_result = previous_result.drop(
                    columns=duplicated_columns
                )
                previous_result = pd.concat(
                    [drop_previous_result, new_result], axis=1
                )
            else:
                # retrieved_passages = self.extract_retrieve_passage(
                # 	previous_result
                # )
                # yield "", retrieved_passages
                # Start streaming of the result
                assert len(previous_result) == 1
                prompt: str = previous_result["prompts"].tolist()[0]
                for delta in module_instance.stream(prompt=prompt,
                                                    **module_param):
                    yield delta, [empty_retrieved_passage]

    def extract_retrieve_passage(self, df: pd.DataFrame) -> List[RetrievedPassage]:
        retrieved_ids: List[str] = df["retrieved_ids"].tolist()[0]
        contents = fetch_contents(self.corpus_df, [retrieved_ids])[0]
        if "path" in self.corpus_df.columns:
            paths = fetch_contents(self.corpus_df, [retrieved_ids], column_name="path")[
                0
            ]
        else:
            paths = [None] * len(retrieved_ids)
        metadatas = fetch_contents(
            self.corpus_df, [retrieved_ids], column_name="metadata"
        )[0]
        if "start_end_idx" in self.corpus_df.columns:
            start_end_indices = fetch_contents(
                self.corpus_df, [retrieved_ids], column_name="start_end_idx"
            )[0]
        else:
            start_end_indices = [None] * len(retrieved_ids)
        return list(
            map(
                lambda content, doc_id, path, metadata, start_end_idx: RetrievedPassage(
                    content=content,
                    doc_id=doc_id,
                    filepath=path,
                    file_page=metadata.get("page", None),
                    start_idx=start_end_idx[0] if start_end_idx else None,
                    end_idx=start_end_idx[1] if start_end_idx else None,
                ),
                contents,
                retrieved_ids,
                paths,
                metadatas,
                start_end_indices,
            )
        )
