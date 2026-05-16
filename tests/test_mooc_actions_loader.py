from pathlib import Path

import torch

from src.data.loaders.mooc_actions import MoocActionsLoader


def _write_csv(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "mooc.csv"
    p.write_text(content)
    return p


def test_parse_mooc_extracts_user_item_ts(tmp_path: Path):
    csv = (
        "user_id,item_id,timestamp,state_label,feat0,feat1,feat2,feat3\n"
        "0,100,0.0,0,0.1,0.2,0.3,0.4\n"
        "1,101,1.5,0,0.2,0.2,0.3,0.4\n"
        "0,101,2.0,1,0.5,0.2,0.3,0.4\n"
    )
    p = _write_csv(tmp_path, csv)
    df = MoocActionsLoader().parse(p)
    assert list(df.columns) == ["src", "dst", "ts"]
    assert len(df) == 3
    # parse() shifts dst by max_user+1 to keep ID ranges disjoint
    # max_user in this test is 1, so dst shifts +2: 100→102, 101→103
    assert df.iloc[0].tolist() == [0, 102, 0]


def test_build_mooc_dynamic_graph(tmp_path: Path):
    lines = ["user_id,item_id,timestamp,state_label,f0,f1,f2,f3"]
    # 50 events alternating between 5 users (ids 0-4) and 4 items (ids 100-103)
    for i in range(50):
        lines.append(f"{i % 5},{100 + (i % 4)},{i * 1.0},0,0,0,0,0")
    p = _write_csv(tmp_path, "\n".join(lines) + "\n")
    g = MoocActionsLoader().build(p, tmp_path / "cache", num_time_steps=5, beta=0.8)
    assert g.dataset_name == "mooc_actions"
    # Bipartite union after remap: 5 users + 4 items = 9 unique node IDs
    assert g.num_nodes == 9
    assert g.num_time_steps == 5
