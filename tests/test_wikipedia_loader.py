from pathlib import Path

from src.data.loaders.wikipedia import WikipediaLoader


def test_parse_wikipedia(tmp_path: Path):
    # Real Wikipedia has 172 feature columns; truncate to 3 in the test for brevity.
    csv = (
        "user_id,item_id,timestamp,state_label,f0,f1,f2\n"
        "0,500,0.0,0,0.1,0.2,0.3\n"
        "1,501,1.0,0,0.1,0.2,0.3\n"
    )
    p = tmp_path / "wikipedia.csv"
    p.write_text(csv)
    df = WikipediaLoader().parse(p)
    assert list(df.columns) == ["src", "dst", "ts"]
    assert len(df) == 2


def test_build_wikipedia_dynamic_graph(tmp_path: Path):
    lines = ["user_id,item_id,timestamp,state_label,f0,f1,f2"]
    for i in range(30):
        lines.append(f"{i % 5},{500 + (i % 4)},{i * 1.0},0,0,0,0")
    p = tmp_path / "wikipedia.csv"
    p.write_text("\n".join(lines) + "\n")
    g = WikipediaLoader().build(p, tmp_path / "cache", num_time_steps=3, beta=0.8)
    assert g.dataset_name == "wikipedia"
    assert g.num_nodes == 9  # 5 editors + 4 pages
