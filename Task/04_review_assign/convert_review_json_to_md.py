#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

try:
    import markdown as md_lib
except Exception:  # noqa: BLE001
    md_lib = None


def esc_md(text: Any) -> str:
    s = str(text if text is not None else "")
    s = s.replace("\n", "<br>")
    s = s.replace("|", "\\|")
    return s


def build_markdown(report: Dict[str, Any], source_path: Path) -> str:
    lines: List[str] = []
    lines.append("# CSV 四项规则审查报告")
    lines.append("")
    lines.append(f"- 源文件: `{source_path.name}`")
    lines.append(f"- 审查时间: `{report.get('review_time', '')}`")
    lines.append(f"- 目录: `{report.get('directory', '')}`")
    lines.append(f"- 文件数: `{report.get('file_count', 0)}`")
    lines.append("")

    files = report.get("files", [])
    total_bad = sum(int(f.get("non_compliant_rows", 0)) for f in files if "error" not in f)
    lines.append(f"- 不合规总条数: **{total_bad}**")
    lines.append("")

    lines.append("## 汇总")
    lines.append("")
    lines.append("| 文件 | 审查区间 | 不合规条数 | 空值 | 含/或括号 | 占位文本 | 词数>=5 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for f in files:
        if "error" in f:
            lines.append(
                f"| {esc_md(f.get('file_name', ''))} | - | - | - | - | - | - |"
            )
            continue
        cr = f.get("checked_range", ["", ""])
        r = f.get("rule_row_counts", {})
        lines.append(
            "| "
            + f"{esc_md(f.get('file_name', ''))} | "
            + f"{cr[0]}-{cr[1]} | "
            + f"{f.get('non_compliant_rows', 0)} | "
            + f"{r.get('empty', 0)} | "
            + f"{r.get('forbidden_symbol', 0)} | "
            + f"{r.get('placeholder', 0)} | "
            + f"{r.get('word_ge_5', 0)} |"
        )
    lines.append("")

    lines.append("## 明细")
    lines.append("")
    for f in files:
        file_name = f.get("file_name", "")
        lines.append(f"### {esc_md(file_name)}")
        lines.append("")
        if "error" in f:
            lines.append(f"- 错误: `{esc_md(f.get('error', ''))}`")
            lines.append("")
            continue

        lines.append(
            f"- 审查区间: `{f.get('checked_range', ['', ''])[0]}-{f.get('checked_range', ['', ''])[1]}`，"
            f"不合规: **{f.get('non_compliant_rows', 0)}**"
        )
        lines.append("")

        issues = f.get("issues", [])
        if not issues:
            lines.append("- 无不合规条目。")
            lines.append("")
            continue

        lines.append("| 行号 | id | 字段 | 触发规则 | 值 |")
        lines.append("|---:|---|---|---|---|")
        for item in issues:
            row_no = item.get("file_row", "")
            rid = esc_md(item.get("id", ""))
            for fi in item.get("field_issues", []):
                field = esc_md(fi.get("field", ""))
                reasons = esc_md(", ".join(fi.get("reasons", [])))
                value = esc_md(fi.get("value", ""))
                lines.append(f"| {row_no} | {rid} | {field} | {reasons} | {value} |")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def build_html_from_markdown(md_text: str, title: str) -> str:
    css = """
body{font-family:Segoe UI,Arial,sans-serif;line-height:1.45;color:#1f2937;margin:28px;}
h1,h2,h3{color:#111827;margin-top:1.1em;}
code{background:#f3f4f6;padding:2px 4px;border-radius:4px;}
pre{background:#f9fafb;border:1px solid #e5e7eb;padding:10px;border-radius:6px;overflow:auto;}
table{border-collapse:collapse;width:100%;margin:14px 0;font-size:12px;table-layout:fixed;word-wrap:break-word;}
th,td{border:1px solid #d1d5db;padding:6px 8px;vertical-align:top;}
th{background:#f3f4f6;}
tr:nth-child(even) td{background:#fcfcfd;}
"""
    if md_lib is not None:
        body = md_lib.markdown(md_text, extensions=["tables", "fenced_code"])
    else:
        # Fallback when markdown package is unavailable.
        escaped = (
            md_text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        body = f"<pre>{escaped}</pre>"

    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{title}</title><style>{css}</style></head><body>{body}</body></html>"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert review_4rules JSON report to both Markdown and HTML."
    )
    parser.add_argument("json_path", help="Path to review_4rules_*.json")
    parser.add_argument("--output", help="Output .md path. Default: same name beside json.")
    parser.add_argument(
        "--html-output",
        help="Output .html path. Default: same stem beside markdown output.",
    )
    args = parser.parse_args()

    json_path = Path(args.json_path).resolve()
    report = json.loads(json_path.read_text(encoding="utf-8"))

    if args.output:
        out_path = Path(args.output).resolve()
    else:
        out_path = json_path.with_suffix(".md")

    md = build_markdown(report, json_path)
    out_path.write_text(md, encoding="utf-8")
    if args.html_output:
        html_out_path = Path(args.html_output).resolve()
    else:
        html_out_path = out_path.with_suffix(".html")

    html = build_html_from_markdown(md, out_path.name)
    html_out_path.write_text(html, encoding="utf-8")

    print(f"Markdown written: {out_path}")
    print(f"HTML written: {html_out_path}")


if __name__ == "__main__":
    main()
