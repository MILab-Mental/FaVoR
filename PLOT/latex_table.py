#!/usr/bin/env python3
"""把分类、多标签分类和回归实验指标导出成一张 LaTeX 表。

用法:
    python PLOT/latex_table.py                       # 用默认通配符，写 PLOT/output/stat/tables/
    python PLOT/latex_table.py --out paper/tables    # 换个输出目录
    python PLOT/latex_table.py --favor 'OUTPUT/.../FaVoR-112px-48f-8fps-e11'

输出 ``table_all_tasks.tex`` 和一份 ``preamble.tex``（三行 ``\\usepackage``），
表格文件头部也带同样内容的注释。**务必先把宏包加进论文导言区**：少了
``multirow``，LaTeX 只报一条 ``Undefined control sequence`` 就把 ``\\multirow`` 丢掉，
``{2}{*}{MER242526 (Emotion)}`` 会被原样排成 ``2*MER242526 (Emotion)``——看起来就像
``\\multirow`` 没生效，实际是宏包没加载。

总表只使用一个 ``tabular``，按任务类型插入三个分组及各自的指标表头；每个任务占两行
（上 FaVoR、下 V-JEPA 2.1 官方权重）。多标签和回归指标通过 ``\\multicolumn`` 均匀铺满
分类表确定的 10 个指标列，因此不会产生一排尾部空列。
被判定的方法在某个指标上更优时加粗；该方法这一行还没跑（目录不存在）就整行留空。
训练还没跑满 epoch 的行，方法名后面会加 ``\\dag``，并在 caption 里注明当前进度。

取哪个 epoch 的值：和训练时保存 ``best.pt`` 的规则完全一致（分类 / 多标签取
``val_f1_macro`` 最大，回归取 ``val_rmse`` 最小），然后**把该 epoch 整行的指标
一起读出**——即这一行就是 best.pt 的真实评测结果，不是每个指标各自取历史最优。

指标读取、任务类型推断、best epoch 选择全部复用 ``stat.py``（见 ``load_stat_helpers``），
不重复实现，避免两个脚本的口径漂移。
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import sys
from pathlib import Path

PLOT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PLOT_ROOT.parent
DEFAULT_OUTPUT = PLOT_ROOT / "output" / "stat" / "tables"

# 复用的列；每个 task 类型一份。注释里标注了该列是否越小越好
METRICS = {
    "classification": [
        ("val_accuracy", "Acc", r"$\uparrow$"),
        ("val_balanced_accuracy", "BAcc", r"$\uparrow$"),
        ("val_precision_macro", "Prec", r"$\uparrow$"),
        ("val_recall_macro", "Rec", r"$\uparrow$"),
        ("val_f1_macro", "F1", r"$\uparrow$"),
        ("val_jaccard_macro", "Jacc", r"$\uparrow$"),
        ("val_cohen_kappa", "Kappa", r"$\uparrow$"),
        ("val_matthews_corrcoef", "MCC", r"$\uparrow$"),
        ("val_auc_ovr_macro", "AUC", r"$\uparrow$"),
        ("val_average_precision_macro", "AP", r"$\uparrow$"),
    ],
    "multi_label_classification": [
        # 不放 val_accuracy：多标签下它是 subset/exact-match accuracy（23 个标签全对
        # 才算对），实测恒为 0.0000，放表里只会误导。Hamming / F1 才是有信息的。
        ("val_hamming_loss", "Hamming", r"$\downarrow$"),
        ("val_precision_macro", "Prec", r"$\uparrow$"),
        ("val_recall_macro", "Rec", r"$\uparrow$"),
        ("val_f1_macro", "F1", r"$\uparrow$"),
        ("val_jaccard_macro", "Jacc", r"$\uparrow$"),
        ("val_auc_ovr_macro", "AUC", r"$\uparrow$"),
        ("val_average_precision_macro", "AP", r"$\uparrow$"),
    ],
    "regression": [
        ("val_mae", "MAE", r"$\downarrow$"),
        ("val_rmse", "RMSE", r"$\downarrow$"),
        ("val_mse", "MSE", r"$\downarrow$"),
        ("val_r2", r"R$^2$", r"$\uparrow$"),
        ("val_adjusted_r2", r"Adj.~R$^2$", r"$\uparrow$"),
    ],
}

# task 名尾缀 -> 展示用的写法（`*_` 在 LaTeX 里要转义，这里直接换成连字符）
TASK_SUFFIX_LABELS = {
    "emotion": "Emotion",
    "intensity": "Intensity",
    "valence": "Valence",
    "activation": "Activation",
    "dominance": "Dominance",
    "pos_intensity": "Pos-Intensity",
    "26openset": "26-OpenSet",
    "PHQ": "PHQ",
}

METHODS = (("FaVoR", "favor"), ("V-JEPA~2.1", "vjepaori"))

SECTIONS = (
    # (task 类型, 分组标题)
    ("classification", "Classification"),
    ("multi_label_classification", "Multi-label classification"),
    ("regression", "Regression"),
)
TABLE_FILE = "table_all_tasks.tex"
TABLE_LABEL = "tab:all-results"

# 单一 tabular 固定为 Task + Method + 10 个指标网格列。分类恰好 10 个指标；多标签和
# 回归用 multicolumn 横跨网格列，既允许各任务类型保留不同表头，又能铺满同一张表。
METRIC_SPANS = {
    "classification": (1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
    "multi_label_classification": (2, 2, 2, 1, 1, 1, 1),
    "regression": (2, 2, 2, 2, 2),
}
GRID_METRIC_COLUMNS = 10

# 生成物依赖的宏包，以及可直接 \input 的导言区片段文件名。
# 少写 multirow 时 LaTeX 只报一条 Undefined control sequence 就把 \multirow 整个丢掉，
# 后面的 {2}{*}{MER242526 (Emotion)} 会被当成普通文本原样排出来 —— PDF 里显示成
# “2*MER242526 (Emotion)”，看起来就像 \multirow 没生效。所以依赖要写在生成物里。
PREAMBLE_FILE = "preamble.tex"
PREAMBLE_LINES = (
    "\\usepackage{booktabs}",
    "\\usepackage{multirow}",
    "\\usepackage{graphicx}   % 仅列数 > 9 的表用 \\resizebox 时需要",
)


def uses_resizebox(task_type: str, fit: bool) -> bool:
    """该表是否会用 ``\\resizebox`` 贴齐 ``\\textwidth``（决定要不要 graphicx）。"""
    return fit and len(METRICS[task_type]) > 9


def dependency_banner(task_types: list[str], fit: bool) -> str:
    """生成物头部的注释块，把宏包依赖写在最显眼的地方。"""
    needed = [line for line in PREAMBLE_LINES if "graphicx" not in line]
    if any(uses_resizebox(task_type, fit) for task_type in task_types):
        needed = list(PREAMBLE_LINES)
    lines = [
        "% " + "=" * 74,
        "% 本文件是表格片段，不是完整文档；请确认论文导言区已加载：",
        *[f"%     {line}" for line in needed],
        "% 漏掉 multirow 时，LaTeX 只会报一条 Undefined control sequence，然后把",
        "% \\multirow 丢掉、把它的参数当普通文本排出来，于是 MER242526 那组会印成",
        "% “2*MER242526 (Emotion)”，看起来就像 \\multirow 没生效。",
        f"% 可直接 \\input 的导言区片段见同目录 {PREAMBLE_FILE}。",
        "% " + "=" * 74,
    ]
    return "\n".join(lines) + "\n"


def count(n: int, noun: str) -> str:
    """`1 task` / `7 tasks` —— 论文 caption 里单复数错了很扎眼。"""
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def load_stat_helpers():
    """按文件路径加载 stat.py。

    不能用 ``import stat`` —— 那会顶掉标准库同名的 ``stat`` 模块（pathlib 等会用到），
    这里用显式路径加载并起一个不会冲突的模块名。
    """
    path = Path(__file__).resolve().with_name("stat.py")
    spec = importlib.util.spec_from_file_location("favor_stat_helpers", path)
    module = importlib.util.module_from_spec(spec)
    # 必须先登记进 sys.modules：stat.py 用了 @dataclass，dataclasses 在解析字符串
    # 注解时要靠 sys.modules[cls.__module__] 找回命名空间。
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


stat = load_stat_helpers()


def escape(text: str) -> str:
    """转义 LaTeX 特殊字符（任务名里理论上还有下划线残留时兜底）。"""
    for old, new in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                     ("#", r"\#"), ("_", r"\_"), ("$", r"\$")):
        text = text.replace(old, new)
    return text


def task_label(folder_name: str) -> str:
    """`MER242526-pos_intensity` -> `MER242526 (Pos-Intensity)`。

    按**最后一个**连字符切分，这样 `CREMA-D-emotion` 会得到 `CREMA-D` 而不是 `CREMA`。
    """
    dataset, sep, suffix = folder_name.rpartition("-")
    if not sep:
        return escape(folder_name)
    pretty = TASK_SUFFIX_LABELS.get(suffix, suffix.replace("_", "-"))
    return f"{escape(dataset)} ({escape(pretty)})"


def index_by_task(pattern: str) -> dict[str, Path]:
    """展开通配符，按「任务目录名 -> 实验目录」建索引。"""
    found: dict[str, Path] = {}
    for hit in glob.glob(str(Path(pattern).expanduser()), recursive=True):
        path = Path(hit)
        if path.is_dir():
            found.setdefault(path.parent.name, path)
    return found


def measure(folder: Path | None) -> dict:
    """读一个实验目录，返回该目录 best epoch 那一行的全部指标。"""
    empty = {"task": None, "values": {}, "best_epoch": None,
             "epochs_seen": 0, "target_epochs": None, "in_progress": False}
    if folder is None:
        return empty

    # 复用 stat.py 的任务推断 + best epoch 选择（口径与 best.pt 完全一致）
    record = stat.collect(str(folder), folder, None)
    history = folder / "logs" / "history.csv"
    if record.best_epoch is None or not history.is_file():
        return {**empty, "task": record.task, "epochs_seen": record.epochs_seen,
                "target_epochs": record.target_epochs}

    rows, _duplicates, _fields = stat.read_history(history)
    target = record.target_epochs
    return {
        "task": record.task,
        "values": rows.get(record.best_epoch, {}),
        "best_epoch": record.best_epoch,
        "epochs_seen": record.epochs_seen,
        "target_epochs": target,
        "in_progress": bool(target and record.epochs_seen < target),
    }


def fmt(value) -> str:
    return f"{value:.4f}" if isinstance(value, (int, float)) else ""


def winner(favor_value, vjepaori_value, arrow: str):
    """返回 `'favor'` / `'vjepaori'` / None，None 表示不比（缺值或持平）。"""
    if not isinstance(favor_value, (int, float)) or not isinstance(vjepaori_value, (int, float)):
        return None
    if favor_value == vjepaori_value:
        return None
    better_is_larger = arrow == r"$\uparrow$"
    favor_wins = (favor_value > vjepaori_value) if better_is_larger else (favor_value < vjepaori_value)
    return "favor" if favor_wins else "vjepaori"


def build_section(task_type: str, title: str,
                  rows: list[tuple[str, dict, dict]]) -> list[str]:
    """把一种任务类型渲染成单一 tabular 中的一组行。"""
    metrics = METRICS[task_type]
    spans = METRIC_SPANS[task_type]
    if len(metrics) != len(spans) or sum(spans) != GRID_METRIC_COLUMNS:
        raise ValueError(f"invalid metric spans for {task_type}")

    header_cells = [
        f"\\multicolumn{{{span}}}{{c}}{{\\textbf{{{head}}} {arrow}}}"
        for (_column, head, arrow), span in zip(metrics, spans)
    ]
    body = [
        f"\\multicolumn{{{GRID_METRIC_COLUMNS + 2}}}{{l}}{{\\textbf{{{title}}} "
        f"({count(len(rows), 'task')})}} \\\\",
        "\\addlinespace[2pt]",
        "\\textbf{Task} & \\textbf{Method} & " + " & ".join(header_cells) + " \\\\",
        "\\midrule",
    ]

    for index, (name, favor, vjepaori) in enumerate(rows):
        for position, (method, key) in enumerate(METHODS):
            side = favor if key == "favor" else vjepaori
            cells = []
            for (column, _head, arrow), span in zip(metrics, spans):
                text = fmt(side["values"].get(column))
                if text and winner(
                    favor["values"].get(column), vjepaori["values"].get(column), arrow
                ) == key:
                    text = f"\\textbf{{{text}}}"
                cells.append(f"\\multicolumn{{{span}}}{{c}}{{{text}}}")
            mark = r"\textsuperscript{\dag}" if side["in_progress"] else ""
            first = f"\\multirow{{2}}{{*}}{{{task_label(name)}}}" if position == 0 else ""
            body.append(f"{first} & {method}{mark} & " + " & ".join(cells) + " \\\\")
        if index != len(rows) - 1:
            body.append(f"\\cmidrule(lr){{1-{GRID_METRIC_COLUMNS + 2}}}")
    return body


def build_combined_table(sections: list[tuple[str, str, list[tuple[str, dict, dict]]]],
                         fit: bool = True) -> str:
    """把三种任务类型排进同一个 ``table*`` 和同一个 ``tabular``。"""
    all_rows = [row for _task_type, _title, rows in sections for row in rows]
    missing = [name for name, _favor, vjepaori in all_rows if vjepaori["task"] is None]
    progress = []
    for name, favor, vjepaori in all_rows:
        for method, key in METHODS:
            side = favor if key == "favor" else vjepaori
            if side["in_progress"]:
                progress.append(
                    f"{task_label(name)}--{method} "
                    f"({side['epochs_seen']}/{side['target_epochs']})"
                )

    notes = [
        f"Results on {count(len(all_rows), 'downstream task')}, grouped by task type.",
        "Best epoch is selected by validation F1-macro (highest) for classification and "
        "multi-label classification, and by validation RMSE (lowest) for regression; "
        "all metrics in a row are read from that same epoch.",
        r"FaVoR uses our self-supervised initialization; V-JEPA~2.1 uses the official "
        r"released weights.",
    ]
    if missing:
        notes.append(
            f"Blank V-JEPA~2.1 rows indicate unavailable runs ({count(len(missing), 'task')})."
        )
    if progress:
        notes.append(
            r"$^{\dag}$~Training still in progress, best-so-far: " + ", ".join(progress) + "."
        )

    table_body = [
        f"\\begin{{tabular}}{{ll{'c' * GRID_METRIC_COLUMNS}}}",
        "\\toprule",
    ]
    for index, (task_type, title, rows) in enumerate(sections):
        if index:
            table_body += ["\\midrule", "\\addlinespace[2pt]"]
        table_body += build_section(task_type, title, rows)
    table_body += ["\\bottomrule", "\\end{tabular}"]

    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\small",
        f"\\caption{{{' '.join(notes)}}}",
        f"\\label{{{TABLE_LABEL}}}",
    ]
    if fit:
        lines += ["\\resizebox{\\textwidth}{!}{%", *table_body, "}"]
    else:
        lines += table_body
    lines.append("\\end{table*}")
    return "\n".join(lines) + "\n"


def print_preview(name: str, values: dict, task_type: str) -> None:
    """终端里打一份纯文本预览，方便和日志/历史表对一眼。"""
    columns = "".join(f"{head:>9}" for _c, head, _a in METRICS[task_type])
    print(f"  {name:<22}{columns}")
    for method, key in METHODS:
        side = values[key]
        cells = "".join(f"{fmt(side['values'].get(c)):>9}" for c, _h, _a in METRICS[task_type])
        mark = " *" if side["in_progress"] else ""
        print(f"    {method:<20}{cells}{mark}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="把分类 / 多标签 / 回归结果导出为一个 tabular 的 LaTeX 总表",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python PLOT/latex_table.py\n"
            "  python PLOT/latex_table.py --out paper/tables\n"
            "  python PLOT/latex_table.py --favor 'OUTPUT/finetune_v/vitl16/*/FaVoR-112px-48f-8fps-e11'\n"
        ),
    )
    parser.add_argument("--favor", default=str(PROJECT_ROOT / "OUTPUT/finetune_v/vitl16/*/FaVoR-112px-48f-8fps-e5"),
                        help="FaVoR 自监督预训练那一列的通配符")
    parser.add_argument("--vjepaori", default=str(PROJECT_ROOT / "OUTPUT/finetune_v/vitl16/*/FaVoR-112px-48f-8fps-vjepaori"),
                        help="V-JEPA 2.1 官方权重那一列的通配符（没跑的任务自动留空）")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT,
                        help=f"输出的 .tex 放哪个目录（默认 {DEFAULT_OUTPUT}）")
    parser.add_argument("--no-fit", action="store_true",
                        help="列数多的表不用 \\resizebox 贴齐 \\textwidth（默认会贴齐，避免溢出）")
    parser.add_argument("--preview", action="store_true",
                        help="同时把三个任务类型的纯文本预览打到终端")
    args = parser.parse_args()

    favor_dirs = index_by_task(args.favor)
    vjepaori_dirs = index_by_task(args.vjepaori)
    tasks = sorted(set(favor_dirs) | set(vjepaori_dirs))
    if not tasks:
        print(f"没有目录匹配: {args.favor} 或 {args.vjepaori}", file=sys.stderr)
        return 2

    measured = {name: (measure(favor_dirs.get(name)), measure(vjepaori_dirs.get(name))) for name in tasks}

    args.out.mkdir(parents=True, exist_ok=True)
    sections: list[tuple[str, str, list[tuple[str, dict, dict]]]] = []
    for task_type, title in SECTIONS:
        rows = [
            (name, favor, vjepaori)
            for name, (favor, vjepaori) in measured.items()
            if (favor["task"] or vjepaori["task"]) == task_type
        ]
        if not rows:
            continue
        # 按任务名排序（回归表按指标方向不同，名字序最稳定）
        rows.sort(key=lambda item: item[0])
        sections.append((task_type, title, rows))
        if args.preview:
            print(f"\n===== {task_type} ({len(rows)} 个任务) =====")
            for name, favor, vjepaori in rows:
                print_preview(name, {"favor": favor, "vjepaori": vjepaori}, task_type)
            print("  (* = 训练未跑完，当前阶段性最优)")

    if not sections:
        print("没有可写出的表", file=sys.stderr)
        return 2

    fit = not args.no_fit
    task_types = [task_type for task_type, _title, _rows in sections]
    (args.out / TABLE_FILE).write_text(
        dependency_banner(task_types, fit)
        + build_combined_table(sections, fit=fit),
        encoding="utf-8",
    )

    # 导言区片段单独落一份，方便直接 \input 到论文里，不用去正文里抄宏包名
    preamble = (
        "% 单一 tabular 指标总表所需的宏包，贴到论文导言区即可（已加载过的行可删）\n"
        + "\n".join(PREAMBLE_LINES)
        + "\n"
    )
    (args.out / PREAMBLE_FILE).write_text(preamble, encoding="utf-8")

    print()
    section_summary = ", ".join(
        f"{title} {len(rows)}" for _task_type, title, rows in sections
    )
    print(f"写入 {args.out / TABLE_FILE}  ({section_summary})")
    print(f"写入 {args.out / PREAMBLE_FILE}")
    print(
        "\n\033[1m依赖宏包（漏了 multirow 会把 \\multirow{2}{*}{任务名} 原样印成 "
        "\"2*任务名\"）:\033[0m"
    )
    for line in PREAMBLE_LINES:
        print(f"    {line}")
    print(f"  → 已写好 {args.out / PREAMBLE_FILE}，在导言区 \\input 它即可。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
