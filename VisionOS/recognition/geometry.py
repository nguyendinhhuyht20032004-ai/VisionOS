"""Hình học 2D thuần Python cho vùng đếm và vạch cắt.

Đây là phần *dễ sai và cần chắc chắn nhất* của phần đếm/cảnh báo theo vùng:
kiểm tra một điểm có nằm trong đa giác (vùng giám sát) hay không, và xác định
một track có **cắt vạch** giữa hai frame liên tiếp hay không (đếm vào/ra). Vì thế
nó được tách riêng, không phụ thuộc numpy/supervision, để unit-test kỹ.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

Point = Tuple[float, float]

__all__ = [
    "point_in_polygon",
    "polygon_area",
    "side_of_line",
    "segments_intersect",
    "line_crossing",
]


def point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    """Ray-casting: điểm có nằm trong đa giác (kể cả lồi/lõm) không.

    Đa giác cho bằng danh sách đỉnh theo thứ tự; tự động khép kín. Điểm nằm đúng
    trên cạnh được coi là *bên trong* (đủ dùng cho vùng giám sát).
    """
    n = len(polygon)
    if n < 3:
        return False
    x, y = point
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        # Điểm trùng đỉnh -> coi như bên trong.
        if (xi == x and yi == y):
            return True
        intersect = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
        )
        if intersect:
            inside = not inside
        j = i
    return inside


def polygon_area(polygon: Sequence[Point]) -> float:
    """Diện tích đa giác (công thức shoelace), luôn trả giá trị không âm."""
    n = len(polygon)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def side_of_line(p: Point, a: Point, b: Point) -> float:
    """Dấu của tích có hướng: điểm ``p`` nằm phía nào của vạch a→b.

    >0 : bên trái hướng a→b, <0 : bên phải, ==0 : nằm trên đường thẳng. Dùng để
    phát hiện đổi phía (tức là cắt vạch) giữa hai frame.
    """
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def _on_segment(p: Point, q: Point, r: Point) -> bool:
    return (
        min(p[0], r[0]) <= q[0] <= max(p[0], r[0])
        and min(p[1], r[1]) <= q[1] <= max(p[1], r[1])
    )


def segments_intersect(p1: Point, p2: Point, p3: Point, p4: Point) -> bool:
    """Hai đoạn thẳng p1p2 và p3p4 có giao nhau không (kể cả chạm mút)."""
    d1 = side_of_line(p3, p1, p2)
    d2 = side_of_line(p4, p1, p2)
    d3 = side_of_line(p1, p3, p4)
    d4 = side_of_line(p2, p3, p4)
    if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)):
        return True
    if d1 == 0 and _on_segment(p1, p3, p2):
        return True
    if d2 == 0 and _on_segment(p1, p4, p2):
        return True
    if d3 == 0 and _on_segment(p3, p1, p4):
        return True
    if d4 == 0 and _on_segment(p3, p2, p4):
        return True
    return False


def line_crossing(
    prev: Point, curr: Point, line_start: Point, line_end: Point
) -> Optional[str]:
    """Xác định track cắt vạch theo hướng nào giữa ``prev`` -> ``curr``.

    Trả về:
      * ``"in"``  nếu đi từ phía phải sang phía trái vạch (side âm -> dương),
      * ``"out"`` nếu ngược lại,
      * ``None``  nếu không cắt vạch trong bước này.

    Quy ước IN/OUT ở đây thuần hình học (giống ghi chú trong ``README_TESTS``):
    tuỳ bố trí camera có thể đảo nhãn, còn logic đếm không đổi.
    """
    if not segments_intersect(prev, curr, line_start, line_end):
        return None
    s_prev = side_of_line(prev, line_start, line_end)
    s_curr = side_of_line(curr, line_start, line_end)
    if s_prev == s_curr:
        return None  # chạm rồi quay lại cùng phía -> không tính
    return "in" if s_curr > s_prev else "out"
