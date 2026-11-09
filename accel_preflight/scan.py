# -*- coding: utf-8 -*-
"""扫描器：把命令行 / 环境变量 / requirements / pyproject 扫成 findings。

每条 finding 结构：
    {id, severity, status, basis, platform, match, message, evidence, note, fix, scope?}
status ∈ ok / variant / partial / unknown / blocked（blocked 才算 blocker）。
basis ∈ observed / upstream-doc / unknown。observed 只来自一手证据且状态不是 unknown。
"""

import re
import shlex

from accel_preflight import db

# 平台缺省状态：规则没写该平台 → unknown，不猜测
_DEFAULT_STATUS = "unknown"

_ENV_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
_PKG_SPLIT_RE = re.compile(r"[\[\]><=!~;,\s]+")
_PEP503_RE = re.compile(r"[-_.]+")
_VERSION_RE = re.compile(r"(?:==|>=|<=|~=|===)\s*([0-9][^\s,;\]]*)")
# 2026-11-09 在 pytorch.org 安装页上实际看到的是 whl/cu118 这种索引，不是任意 "cu" 子串。
_CUDA_INDEX_RE = re.compile(r"whl/cu\d+", re.IGNORECASE)
_CUDA_TAG_RE = re.compile(r"(?:\+|%2B)cu\d+", re.IGNORECASE)
_INDEX_OPTS = ("--index-url", "--extra-index-url", "--find-links", "-f", "-i")

# torch 本地版本 × 目标平台。cpu 必须单独分出来：`cpu`.startswith("cu") 会把 CPU 轮子误判成 CUDA。
# dcu + rocm 标签保持 unknown：ROCm 标签不是 DCU 实测。
TORCH_TAG_STATUS = {
    ("cu", "nvidia"): "ok",
    ("cu", "rocm"): "blocked",
    ("cu", "dcu"): "blocked",
    ("rocm", "nvidia"): "blocked",
    ("rocm", "rocm"): "ok",
    ("rocm", "dcu"): "unknown",
    ("cpu", "nvidia"): "unknown",
    ("cpu", "rocm"): "unknown",
    ("cpu", "dcu"): "unknown",
    ("", "nvidia"): "unknown",
    ("", "rocm"): "unknown",
    ("", "dcu"): "unknown",
}

# 非 torch 包只在「标签和目标平台明显不是一路」时给结论。nvidia 上的 +cu 不升成 ok。
OTHER_TAG_STATUS = {
    ("cu", "rocm"): "blocked",
    ("cu", "dcu"): "blocked",
    ("rocm", "nvidia"): "blocked",
}


def _status_for(rule, platform):
    return rule["platforms"].get(platform, _DEFAULT_STATUS)


def _evidence_for(rule, platform):
    return db.evidence_id_for(rule, platform)


def _make_finding(rule, platform, match, status=None):
    status = status or _status_for(rule, platform)
    evidence = _evidence_for(rule, platform)
    basis = db.basis_for(evidence, status)
    finding = {
        "id": rule["id"],
        "severity": db.STATUS_SEVERITY[status],
        "status": status,
        "basis": basis,
        "platform": platform,
        "match": match,
        "message": rule["note"],
        "evidence": evidence,
        "fix": rule.get("fix", ""),
    }
    if basis == "observed":
        scope = db.EVIDENCE[evidence].get("scope") or ""
        if scope:
            finding["scope"] = scope
    return finding


def _pep503(name):
    """PEP 503：大小写不分，`-` `.` `_` 等价。不归一化就会让 flash_attn 漏出闸门。"""
    return _PEP503_RE.sub("-", name.strip().lower())


def _normalize_pkg_name(raw):
    """FlashInfer[all]==0.2.2 → flashinfer；去掉 extras / 版本约束 / 环境标记。"""
    head = _PKG_SPLIT_RE.split(raw.strip())[0]
    return _pep503(head)


def _version_of(raw):
    """取一行里的版本串（若无则空串）。torch==2.9.1+cu128 → 2.9.1+cu128"""
    match = _VERSION_RE.search(raw)
    return match.group(1) if match else ""


def _local_of(version):
    if "+" not in version:
        return ""
    return version.split("+", 1)[1].lower()


def _tag_family(local):
    """cu128 → cu；cpu → cpu（不能并进 cu）；rocm7.2 → rocm。"""
    local = (local or "").lower()
    if not local:
        return ""
    if local.startswith("cpu"):
        return "cpu"
    if local.startswith("cu"):
        return "cu"
    if local.startswith("rocm"):
        return "rocm"
    return "other"


def _rule_names(rule):
    names = []
    if rule.get("package"):
        names.append(rule["package"])
    names.extend(rule.get("also") or [])
    return [_pep503(n) for n in names if n]


def _match_package_rule(name):
    """精确命中优先于前缀命中。

    `vllm-plugin-das` 既精确命中自己的规则，也满足 `vllm` 的前缀。
    先按规则顺序取第一条会把它判成上游 vllm。
    """
    rules = [r for r in db.package_rules() if _rule_names(r)]
    for rule in rules:
        if name in _rule_names(rule):
            return rule, "exact"
    for rule in rules:
        if any(name.startswith(n + "-") for n in _rule_names(rule)):
            return rule, "prefix"
    return None, None


def _rule_by_id(rule_id):
    return next(r for r in db.ALL_RULES if r["id"] == rule_id)


def _tokenize_command(cmd):
    """shlex 切词，区分环境变量赋值与普通 token（引号内的内容不会被拆开）。"""
    try:
        tokens = shlex.split(cmd)
    except ValueError:
        tokens = cmd.split()
    env, argv = {}, []
    for tok in tokens:
        match = _ENV_RE.match(tok)
        if match and not tok.startswith("--"):
            env[match.group(1)] = match.group(2)
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

    for flag_name, value in _iter_flags(argv):
        for rule in db.flag_rules():
            if flag_name != rule["flag"]:
                continue
            if value is None:
                continue
            folded = value.lower()
            if any(pat in folded for pat in rule["values"]):
                findings.append(_make_finding(rule, platform, f"{flag_name} {value}"))

    for key, value in env.items():
        for rule in db.env_rules():
            if key.upper() == rule["key"] and value.upper() == rule["value"]:
                findings.append(_make_finding(rule, platform, f"{key}={value}"))

    return findings


def _is_index_line(line):
    head = line.split("=", 1)[0].strip()
    # `-iURL` 这种粘连写法少见；只认独立选项，避免把包名里的 -i 算进去。
    token = head.split()[0] if head else ""
    return token in _INDEX_OPTS


def _is_include_line(line):
    token = line.split()[0] if line.split() else ""
    return token in ("-r", "--requirement")


def _index_finding(line, platform):
    if not _CUDA_INDEX_RE.search(line):
        return None
    rule = _rule_by_id("PKG-INDEXURL-001")
    status = {"nvidia": "ok", "rocm": "blocked", "dcu": "blocked"}[platform]
    return _make_finding(rule, platform, line.strip(), status=status)


def _torch_tag_finding(line, platform):
    version = _version_of(line)
    family = _tag_family(_local_of(version))
    if family == "other":
        family = ""
    status = TORCH_TAG_STATUS[(family, platform)]
    rule = _rule_by_id("PKG-TORCHWHEEL-001")
    match = line if version else line + "（未固定版本，无法判断 wheel 平台）"
    finding = _make_finding(rule, platform, match, status=status)
    if family == "cpu":
        finding["message"] = (
            "本地版本以 cpu 开头，不能当成 CUDA 标签（cu 前缀会误伤 cpu）。"
            "本台账没有单独的 CPU 渠道结论，所以是 unknown，不是 ok。"
        )
    elif family == "rocm" and platform == "dcu":
        finding["message"] = (
            "+rocm 标签只说明轮子自称 ROCm 渠道（S9）。DCU 没有实测，保持 unknown。"
        )
    elif family == "cu" and platform == "dcu":
        finding["message"] = (
            "+cu 是 CUDA 渠道标签（S9）。DCU 目标上拒绝它是渠道不匹配，不是 DCU 实测失败。"
        )
    return finding


def _other_tag_finding(line, platform):
    """非 torch 行上的 CUDA/ROCm 标签或 whl/cu URL。没有标签就不发这条。"""
    version = _version_of(line)
    family = _tag_family(_local_of(version))
    url_cuda = bool(_CUDA_TAG_RE.search(line) or _CUDA_INDEX_RE.search(line))
    if family == "cpu":
        return None
    if family not in ("cu", "rocm") and url_cuda:
        family = "cu"
    status = OTHER_TAG_STATUS.get((family, platform))
    if status is None:
        return None
    rule = _rule_by_id("PKG-WHEELTAG-001")
    finding = _make_finding(rule, platform, line.strip(), status=status)
    if platform == "dcu" and status == "blocked":
        finding["message"] = (
            "依赖字符串带 CUDA 渠道标签（+cu 或 whl/cu，S9）。"
            "在 DCU 上拒绝它是渠道不匹配，本仓库没有 DCU 安装记录。"
        )
    return finding


def _unknown_package(line, platform):
    rule = {
        "id": "PKG-UNKNOWN-001",
        "kind": "package",
        "severity_hint": "info",
        "platforms": {},
        "evidence": "S0",
        "platform_evidence": {"nvidia": "S0", "rocm": "S0", "dcu": "S0"},
        "note": "数据库未收录该包的平台状态——答案必须是 unknown，需要人工查证",
        "fix": "查该包上游 README 的安装矩阵与厂商适配仓库（DCU 见 S7，"
               "用 --explain S7 看清单），确认后再把结论补进规则库",
    }
    return _make_finding(rule, platform, line)


def _include_finding(line, platform):
    rule = {
        "id": "PKG-INCLUDE-001",
        "kind": "package",
        "severity_hint": "info",
        "platforms": {},
        "evidence": "S0",
        "platform_evidence": {"nvidia": "S0", "rocm": "S0", "dcu": "S0"},
        "note": "这一行是 -r/--requirement。本工具不打开被包含的文件，所以里面的 CUDA-only 依赖看不见。",
        "fix": "把被包含文件的内容贴进同一次扫描；不要把 exit 0 理解成那些传递依赖已经查过。",
    }
    return _make_finding(rule, platform, line.strip())


def _apply_prefix(finding):
    """前缀命中不能继承 ok。blocked 保持 blocked，避免 CUDA-only 家族漏掉。"""
    if finding["status"] != "ok":
        return finding
    finding = dict(finding)
    finding["status"] = "partial"
    finding["severity"] = db.STATUS_SEVERITY["partial"]
    finding["basis"] = db.basis_for(finding["evidence"], "partial")
    finding["message"] = (
        finding["message"] + " 前缀命中，不是规则里的精确包名，不能记成 ok。"
    )
    finding.pop("scope", None)
    return finding


def _scan_deps(raw_lines, platform):
    findings = []
    for raw in raw_lines:
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if _is_include_line(line):
            findings.append(_include_finding(line, platform))
            continue
        if line.startswith("-"):
            if _is_index_line(line):
                found = _index_finding(line, platform)
                if found is not None:
                    findings.append(found)
            # 其他 pip 选项（--trusted-host 等）不猜。
            continue

        name = _normalize_pkg_name(line)
        if name == "torch":
            finding = _torch_tag_finding(line, platform)
            cuda_url = bool(_CUDA_TAG_RE.search(line) or _CUDA_INDEX_RE.search(line))
            # `torch @ https://.../whl/cu128/...` 没有 == 版本，标签矩阵会落成 unknown。
            # URL 里已经写了 CUDA 渠道时，ROCm/DCU 目标不能放行。
            if cuda_url and platform in ("rocm", "dcu") and finding["status"] != "blocked":
                finding = _make_finding(
                    _rule_by_id("PKG-TORCHWHEEL-001"), platform, line.strip(), status="blocked"
                )
                finding["message"] = (
                    "这一行的 URL 含 whl/cu 或 +cu（S9 的 CUDA 渠道）。"
                    "拒绝它是渠道字符串，不是在目标机上安装过。"
                )
            findings.append(finding)
            continue

        tag_finding = _other_tag_finding(line, platform)
        if tag_finding is not None and tag_finding["status"] == "blocked":
            findings.append(tag_finding)
            continue

        rule, how = _match_package_rule(name)
        if rule is None:
            if tag_finding is not None:
                findings.append(tag_finding)
            else:
                findings.append(_unknown_package(line, platform))
            continue
        found = _make_finding(rule, platform, line)
        if how == "prefix":
            found = _apply_prefix(found)
        findings.append(found)
    return findings


def scan_requirements(text, platform):
    if not isinstance(text, str):
        raise TypeError("scan_requirements expects the file text, not a list of lines")
    return _scan_deps(text.splitlines(), platform)


def _sections(text):
    matches = list(re.finditer(r"(?m)^\[([^\]]+)\]\s*$", text))
    sections = []
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append((match.group(1).strip(), text[match.end():end]))
    return sections


def scan_pyproject(text, platform):
    """抽 [project].dependencies、optional-dependencies 和 dependency-groups。

    不用 toml 库。只认这些节里的引号字符串，不跟 `-r`，也不读 tool.uv 之外的任意表。
    """
    deps = []
    for name, body in _sections(text):
        if name == "project":
            match = re.search(r"(?m)^dependencies\s*=\s*\[(.*?)\]", body, re.DOTALL)
            if match:
                deps.extend(re.findall(r"['\"]([^'\"]+)['\"]", match.group(1)))
        elif name in ("project.optional-dependencies", "dependency-groups"):
            deps.extend(re.findall(r"['\"]([^'\"]+)['\"]", body))
    return _scan_deps(deps, platform)


def scan_text(text, platform):
    """同时按命令行与依赖两种语义扫一段文字。依赖扫描吃原文，不吃 token 列表。"""
    return scan_command(text, platform) + scan_requirements(text, platform)
