"""Rule theo routingForTryon.md; không fallback sang model sai vùng cơ thể."""
from api.core.errors import ServiceError

UPPER = {"short_sleeved_shirt", "long_sleeved_shirt", "short_sleeved_outwear", "long_sleeved_outwear", "vest", "sling"}
LOWER = {"shorts", "trousers", "skirt"}
DRESSES = {"short_sleeved_dress", "long_sleeved_dress", "vest_dress", "sling_dress"}
ALIASES = {"short_sleeve_top": "short_sleeved_shirt", "long_sleeve_top": "long_sleeved_shirt", "short_sleeve_outwear": "short_sleeved_outwear", "long_sleeve_outwear": "long_sleeved_outwear", "short_sleeve_dress": "short_sleeved_dress", "long_sleeve_dress": "long_sleeved_dress"}
ALIASES.update({"short_sleeve_outerwear": "short_sleeved_outwear", "long_sleeve_outerwear": "long_sleeved_outwear"})


def normalize_label(label: str) -> str:
    """Chuẩn hóa nhãn checkpoint DeepFashion2, không dựa tên file người dùng."""
    value = label.lower().strip().replace("-", "_").replace(" ", "_")
    return ALIASES.get(value, value)


def category_for(label: str) -> str:
    """Gom subtype thành ba vùng VTON; nhãn lạ phải được duyệt."""
    label = normalize_label(label)
    for labels, category in ((UPPER, "upper_body"), (LOWER, "lower_body"), (DRESSES, "dress")):
        if label in labels:
            return category
    raise ServiceError("GARMENT_CATEGORY_UNKNOWN", "Không nhận diện được loại trang phục hỗ trợ.")


def route(category: str, frame: str, subtype: str | None = None, length: str = "UNKNOWN") -> dict:
    """Kiểm tra frame/đồ dài trước inference; preview không chứng minh ảnh thực tế đạt."""
    if category not in {"upper_body", "lower_body", "dress"}:
        raise ServiceError("GARMENT_CATEGORY_UNKNOWN", "Category không được hỗ trợ.")
    if subtype and category_for(subtype) != category:
        raise ServiceError("CATEGORY_MISMATCH", "Subtype không thuộc vùng trang phục đã chọn.")
    warnings = []
    accepted = frame != "invalid"
    if not accepted:
        warnings.append("Ảnh cần thấy rõ ít nhất vai và hông.")
    needs_full = category != "upper_body" and (normalize_label(subtype or "") == "trousers" or length in {"MIDI", "LONG"} or (length == "UNKNOWN" and normalize_label(subtype or "") not in {"shorts"}))
    if needs_full and frame != "full_body":
        accepted = False
        warnings.append("Vui lòng dùng ảnh toàn thân thấy rõ mắt cá chân cho đồ dài hoặc chưa rõ chiều dài.")
    elif accepted and category != "upper_body" and frame != "full_body":
        warnings.append("Chỉ đánh giá vùng đang thấy trong ảnh; nên dùng ảnh toàn thân.")
    return {"accepted": accepted, "vtonCategory": category, "personFrame": frame,
            "backend": "OOTDiffusion VITON-HD" if category == "upper_body" else "OOTDiffusion DressCode",
            "modelType": "hd" if category == "upper_body" else "dc",
            "endpoint": "/process_hd" if category == "upper_body" else "/process_dc",
            "warnings": warnings}
