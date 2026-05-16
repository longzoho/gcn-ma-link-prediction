from pathlib import Path

from src.data.loaders.lastfm import LastFMLoader


def test_parse_lastfm(tmp_path: Path):
    csv = (
        "user_id,item_id,timestamp,state_label,f0,f1\n"
        "0,200,0.0,0,0.1,0.2\n"
        "0,201,1.0,0,0.1,0.2\n"
        "1,200,2.0,0,0.1,0.2\n"
    )
    p = tmp_path / "lastfm.csv"
    p.write_text(csv)
    df = LastFMLoader().parse(p)
    assert list(df.columns) == ["src", "dst", "ts"]
    assert len(df) == 3


def test_build_lastfm_dynamic_graph(tmp_path: Path):
    lines = ["user_id,item_id,timestamp,state_label,f0,f1"]
    for i in range(40):
        lines.append(f"{i % 4},{200 + (i % 6)},{i * 1.0},0,0,0")
    p = tmp_path / "lastfm.csv"
    p.write_text("\n".join(lines) + "\n")
    g = LastFMLoader().build(p, tmp_path / "cache", num_time_steps=4, beta=0.8)
    assert g.dataset_name == "lastfm"
    assert g.num_nodes == 10  # 4 users + 6 items
