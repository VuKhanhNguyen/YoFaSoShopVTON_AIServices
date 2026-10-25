"""Trích hàm thuần từ notebook đã duyệt; không execute cell hoặc tải model.

Chạy lại có chủ ý khi notebook thay đổi, sau đó review diff và test regression.
Không được gọi script này trên mỗi request hoặc lúc worker khởi động.
"""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "api/vision"


def source(name, index):
    """Đọc source một cell mà không chạy thao tác upload/Colab trong nó."""
    notebook = json.loads((ROOT / "ml-training/notebooks" / name).read_text(encoding="utf-8"))
    return ast.parse("".join(notebook["cells"][index]["source"]))


def emit(filename, header, nodes):
    """Ghi snapshot hàm thuần có docstring, tránh global tác vụ của notebook."""
    for node in nodes:
        if isinstance(node, ast.FunctionDef) and not ast.get_docstring(node):
            node.body.insert(0, ast.Expr(value=ast.Constant(value=f"Bước {node.name}: helper thuần trích từ notebook; xem service gọi để biết luồng dữ liệu.")))
    TARGET.mkdir(exist_ok=True)
    (TARGET / filename).write_text(header + "\n\n" + ast.unparse(ast.Module(body=nodes, type_ignores=[])) + "\n", encoding="utf-8")


def main():
    """Chỉ giữ định nghĩa/constant, không giữ các câu lệnh suy luận ở cấp module."""
    product = "product_reference_analysis_colab.ipynb"
    nodes = []
    for index in (6, 8):
        for n in source(product, index).body:
            if isinstance(n, ast.Assign) and all(isinstance(t, ast.Name) and t.id.isupper() for t in n.targets):
                nodes.append(n)
            if isinstance(n, ast.FunctionDef) and not n.name.startswith("upload_") and n.name not in {"read_upload", "render_split_swatch", "split_swatch_html"}:
                nodes.append(n)
    header = '''"""Snapshot logic sản phẩm từ notebook; không load model khi import.
Model/runtime được service worker gắn trong một process, không chạy trong API.
Nguồn: product_reference_analysis_colab.ipynb, cell 6 và 8.
"""
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
DEVICE = "cpu"
fashion_model = None
pose_model = None
'''
    emit("product_primitives.py", header, nodes)
    allowed = {"prepare_oot_person", "restore_oot_result", "garment_edit_mask", "outside_garment_change_score", "change_to_preservation_score", "composite_preserving_source", "upper_body_protected_mask", "protected_region_artifact_score", "garment_fit_score", "calculate_limb_integrity_score", "evaluate_dress_proportions_and_color", "garment_color_similarity", "candidate_match_quality"}
    nodes = [n for n in source("week4_experiment_colab_ver2.ipynb", 12).body if isinstance(n, ast.FunctionDef) and n.name in allowed or isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "DRESS_STYLE_FAMILY" for t in n.targets)]
    emit("candidate_primitives.py", '''"""Snapshot hình học/màu/gate từ notebook VTON cell 12; không gọi Space."""
import cv2
import numpy as np
from PIL import Image
''', nodes)


if __name__ == "__main__":
    main()
