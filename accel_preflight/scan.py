# -*- coding: utf-8 -*-
"""扫描器：把命令行 / 环境变量 / requirements / pyproject 扫成 findings。

每条 finding 结构：
    {id, severity, status, platform, match, message, evidence, note}
status ∈ ok / variant / partial / unknown / blocked（blocked 才算 blocker）。
"""

import re
import shlex

from accel_preflight import db

# 平台缺省状态：规则没写该平台 → unknown，不猜测
_DEFAULT_STATUS = "unknown"

_ENV_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
_PKG_SPLIT_RE = re.compile(r"[\[\]><=!~;,\s]+")


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def _status_for(rule, platform):
    return rule["platforms"].get(platform, _DEFAULT_STATUS)


def _severity_for(rule, platform):
    return db.STATUS_SEVERITY[_status_for(rule, platform)]


def _evidence_for(rule, platform):
    # 缺平台状态时，把「去哪查」指给用户：dcu 指海光官方组织，其他指规则自身证据
    if platform in rule["platforms"]:
        return rule["evidence"]
    if platform == "dcu":
        return "S7"
    return rule["evidence"]


def _make_finding(rule, platform, match, status=None):
    status = status or _status_for(rule, platform)
    return {
        "id": rule["id"],
        "severity": db.STATUS_SEVERITY[status],
        "status": status,
        "platform": platform,
        "match": match,
        "message": rule["note"],
        "evidence": _evidence_for(rule, platform),
    }


def _normalize_pkg_name(raw):
    """FlashInfer[all]==0.2.2 → flashinfer；去掉 extras / 版本约束 / 环境标记。"""
    return _PKG_SPLIT_RE.split(raw.strip())[0].lower()


def _version_of(raw):
    """取 torch 一行的版本串（若无则空串）。torch==2.9.1+cu128 → 2.9.1+cu128"""
    m = re.search(r"(?:==|>=|<=|~=)\s*([0-9][^\s,;\]]*)", raw)
    return m.group(1) if m else ""


# ---------------------------------------------------------------------------
# 命令行扫描
# ---------------------------------------------------------------------------

def _tokenize_command(cmd):
    """shlex 切词，区分环境变量赋值与普通 token（引号内的内容不会被拆开）。"""
    try:
        tokens = shlex.split(cmd)
    except ValueError:
        # 引号不配对等异常输入：退化为普通 split，仍然可扫
        tokens = cmd.split()
    env, argv = {}, []
    for tok in tokens:
        m = _ENV_RE.match(tok)
        if m and not tok.startswith("--"):
            env[m.group(1)] = m.group(2)
        else:
            argv.append(tok)
    return env, argv


def _iter_flags(argv):
    """产出 (flag_name, value)；--flag value 与 --flag=value 两种形态都覆盖。"""
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok.startswith("--") and "=" in tok:
            name, _, value = tok.partition("=")
            yield name, value
            i += 1
        elif tok.startswith("--"):
            value = None
            if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                value = argv[i + 1]
                i += 2
            else:
                i += 1
            yield tok, value
        else:
            i += 1


def scan_command(cmd, platform):
    env, argv = _tokenize_command(cmd)
    findings = []

    # 启动参数规则
    for flag_name, value in _iter_flags(argv):
        for rule in db.flag_rules():
            if flag_name != rule["flag"]:
                continue
            if value is None:
                continue
            v = value.lower()
            if any(pat in v for pat in rule["values"]):
                findings.append(_make_finding(rule, platform, f"{flag_name} {value}"))

    # 环境变量规则
    for key, value in env.items():
        for rule in db.env_rules():
            if key.upper() == rule["key"] and value.upper() == rule["value"]:
                findings.append(
                    _make_finding(rule, platform, f"{key}={value}")
                )

    return findings


# ---------------------------------------------------------------------------
# 依赖扫描
# ---------------------------------------------------------------------------

def _scan_deps(raw_lines, platform):
    findings = []
    seen_torch = False
    for raw in raw_lines:
        line = raw.split("#", 1)[0].strip()  # 去注释
        if not line or line.startswith("-"):  # 空行 / -r -e 等指令行跳过
            continue
        name = _normalize_pkg_name(line)

        if name == "torch":
            seen_torch = True
            findings.append(_torch_wheel_finding(line, platform))
            continue

        matched = False
        for rule in db.package_rules():
            if name == rule["package"] or name.startswith(rule["package"] + "-"):
                matched = True
                findings.append(_make_finding(rule, platform, line))
                break
        if not matched:
            # 未收录的包：明确写 unknown，不猜。这条是逻辑规则、不在 db.ALL_RULES 里
            # （它不针对任何具体包，完整性测试的「至少一个平台表态」对它无意义）。
            unknown_rule = {
                "id": "PKG-UNKNOWN-001",
                "kind": "package",
                "severity_hint": "info",
                "platforms": {},
                "evidence": "S0",
                "note": "数据库未收录该包的平台状态——答案必须是 unknown，需要人工查证",
            }
            findings.append(_make_finding(unknown_rule, platform, line))
    return findings


def _torch_wheel_finding(line, platform):
    """torch wheel 本地版本后缀规则：+cuXXX 属 CUDA，+rocmXXX 属 ROCm。"""
    version = _version_of(line)
    local = ""
    if "+" in version:
        local = version.split("+", 1)[1].lower()
    rule = next(r for r in db.ALL_RULES if r["id"] == "PKG-TORCHWHEEL-001")

    if local.startswith("cu"):
        status = {"nvidia": "ok", "rocm": "blocked", "dcu": "blocked"}[platform]
    elif local.startswith("rocm"):
        status = {"nvidia": "blocked", "rocm": "ok", "dcu": "unknown"}[platform]
    else:
        # 没写本地版本 → 不知道装的是哪个平台的 wheel
        status = "unknown"

    match = line if version else line + "（未固定版本，无法判断 wheel 平台）"
    return _make_finding(rule, platform, match, status=status)


def scan_requirements(text, platform):
    return _scan_deps(text.splitlines(), platform)


def scan_pyproject(text, platform):
    """从 pyproject.toml 抽 [project].dependencies 数组（正则足够，避免引入 toml 依赖）。"""
    deps = []
    m = re.search(r"dependencies\s*=\s*\[(.*?)\]", text, re.DOTALL)
    if m:
        deps = re.findall(r"['\"]([^'\"]+)['\"]", m.group(1))
    return _scan_deps(deps, platform)


def scan_text(text, platform):
    """通用入口：同时按命令行与依赖两种语义扫（适合扫 README / 部署文档片段）。"""
    return scan_command(text, platform) + scan_requirements(
        [ln for ln in text.splitlines()], platform
    )
