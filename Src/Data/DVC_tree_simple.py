import subprocess
import graphviz
import os

def generate_dvc_graph(output_path="Reports/figures/dvc_pipeline"):
    # Создать папку если нет
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    result = subprocess.run(
        ["dvc", "dag", "--dot"],
        capture_output=True,
        text=True
    )

    dot_graph = result.stdout
    graph = graphviz.Source(dot_graph)

    graph.render(filename=output_path, format="png", cleanup=True)

    print(f"Graph saved to {output_path}.png")


if __name__ == "__main__":
    generate_dvc_graph()