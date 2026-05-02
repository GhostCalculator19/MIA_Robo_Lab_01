import subprocess
import graphviz
import os

def get_stage_color(stage_name):
    if stage_name.startswith("preprocess"):
        return "lightblue"
    elif stage_name.startswith("train"):
        return "lightgreen"
    elif stage_name.startswith("test"):
        return "orange"
    else:
        return "white"


def generate_pretty_dvc_graph(output_path="Reports/figures/dvc_pretty_pipeline"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Получаем DOT от DVC
    result = subprocess.run(
        ["dvc", "dag", "--dot"],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(result.stderr)
        return

    dot = graphviz.Digraph(format="png")

    lines = result.stdout.splitlines()

    edges = []
    nodes = set()

    # Парсим DOT (очень простой парсинг)
    for line in lines:
        if "->" in line:
            parts = line.strip().replace(";", "").split("->")
            src = parts[0].strip()
            dst = parts[1].strip()
            edges.append((src, dst))
            nodes.add(src)
            nodes.add(dst)

    # Создаем группы
    with dot.subgraph(name="cluster_preprocess") as c:
        c.attr(label="Preprocess", style="filled", color="lightblue")
        for node in nodes:
            if node.startswith("preprocess"):
                c.node(node, style="filled", fillcolor="lightblue")

    with dot.subgraph(name="cluster_train") as c:
        c.attr(label="Train Models", style="filled", color="lightgreen")
        for node in nodes:
            if node.startswith("train"):
                c.node(node, style="filled", fillcolor="lightgreen")

    with dot.subgraph(name="cluster_test") as c:
        c.attr(label="Test Models", style="filled", color="orange")
        for node in nodes:
            if node.startswith("test"):
                c.node(node, style="filled", fillcolor="orange")

    # Добавляем рёбра
    for src, dst in edges:
        dot.edge(src, dst)

    # Сохраняем
    dot.render(filename=output_path, cleanup=True)

    print(f"Красивый DAG сохранён: {output_path}.png")


if __name__ == "__main__":
    generate_pretty_dvc_graph()