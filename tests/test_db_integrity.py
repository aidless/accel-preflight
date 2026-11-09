# -*- coding: utf-8 -*-
"""规则库完整性测试：每条规则必须有 id / 严重度 / 平台状态 / 证据指针，且证据台账闭合。"""
import unittest

from accel_preflight import db


class TestDbIntegrity(unittest.TestCase):
    def test_rule_ids_unique(self):
        ids = [r["id"] for r in db.ALL_RULES]
        self.assertEqual(len(ids), len(set(ids)), f"规则 id 重复: {ids}")

    def test_every_rule_has_required_fields(self):
        for r in db.ALL_RULES:
            for key in ("id", "kind", "severity_hint", "platforms", "evidence", "note"):
                self.assertIn(key, r, f"规则 {r.get('id')} 缺字段 {key}")
            self.assertIn(r["kind"], ("flag", "env", "package", "torch-wheel"))
            # 每条规则至少对一个平台表态
            self.assertTrue(len(r["platforms"]) >= 1, f"规则 {r['id']} 没有任何平台状态")

    def test_platform_status_values_legal(self):
        legal = {"ok", "blocked", "partial", "variant", "unknown"}
        for r in db.ALL_RULES:
            for plat, status in r["platforms"].items():
                self.assertIn(plat, db.PLATFORMS, f"规则 {r['id']} 平台 {plat} 非法")
                self.assertIn(status, legal, f"规则 {r['id']} 状态 {status} 非法")

    def test_evidence_ledger_closes(self):
        # 每条规则引用的证据必须在台账里，且台账每条都有来源
        ledger_ids = set(db.EVIDENCE)
        for r in db.ALL_RULES:
            self.assertIn(r["evidence"], ledger_ids, f"规则 {r['id']} 引用了不存在的证据 {r['evidence']}")
        for sid, ev in db.EVIDENCE.items():
            for key in ("kind", "source", "note"):
                self.assertIn(key, ev, f"证据 {sid} 缺字段 {key}")
            self.assertIn(ev["kind"], ("upstream-doc", "first-party", "meta"))

    def test_first_party_evidence_declares_date(self):
        # 一手实测证据必须带 as_of 日期，否则无法判断时效
        for sid, ev in db.EVIDENCE.items():
            if ev["kind"] == "first-party":
                self.assertIn("as_of", ev, f"一手证据 {sid} 缺 as_of")

    def test_first_party_never_claims_dcu(self):
        # 一手证据不能给 DCU 背书。非 unknown 的状态若引用一手证据，平台只能是 rocm。
        # nvidia=unknown 仍可指向 S8，表示「只测过 ROCm，所以 NVIDIA 不给结论」。
        for r in db.ALL_RULES:
            platforms = set(r["platforms"]) | set(r.get("platform_evidence") or {})
            for plat in platforms:
                ev_id = db.evidence_id_for(r, plat)
                self.assertIn(ev_id, db.EVIDENCE, f"{r['id']} 的 {plat} 指向不存在的 {ev_id}")
                if db.EVIDENCE[ev_id]["kind"] != "first-party":
                    continue
                self.assertNotEqual(plat, "dcu", f"规则 {r['id']} 用一手证据 {ev_id} 给 DCU 背书")
                status = r["platforms"].get(plat, "unknown")
                if status != "unknown":
                    self.assertEqual(
                        plat, "rocm",
                        f"规则 {r['id']} 用一手证据 {ev_id} 给 {plat}={status} 背书",
                    )

    def test_platform_evidence_keys_are_real(self):
        for r in db.ALL_RULES:
            for plat, ev_id in (r.get("platform_evidence") or {}).items():
                self.assertIn(plat, db.PLATFORMS, r["id"])
                self.assertIn(ev_id, db.EVIDENCE, f"{r['id']} {plat} → {ev_id}")


if __name__ == "__main__":
    unittest.main()
