"""SQLite事务账本：预算、失败、冻结和揭晓共享同一状态机。"""

import json
import math
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .models import ARMS, canonical, digest


class Ledger:
    def __init__(
        self,
        path: Path,
        plan_hash: str,
        data_hash: str,
        code_hash: str,
        budget: int = 3,
        arms: tuple[str, ...] = ARMS,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path, self.budget, self.arms = path, budget, arms
        if budget < 1:
            raise ValueError("预算必须为正数")
        with self.connection() as db:
            db.executescript("""
              CREATE TABLE IF NOT EXISTS identity (
                id INTEGER PRIMARY KEY, plan TEXT, data TEXT, code TEXT, budget INTEGER);
              CREATE TABLE IF NOT EXISTS events (
                seq INTEGER PRIMARY KEY, kind TEXT, payload TEXT, previous TEXT, hash TEXT);
              CREATE TABLE IF NOT EXISTS trials (
                batch TEXT, arm TEXT, attempt INTEGER, candidate TEXT,
                status TEXT, score REAL, payload TEXT, PRIMARY KEY(batch,arm,attempt));
              CREATE TABLE IF NOT EXISTS batches (
                batch TEXT PRIMARY KEY, state TEXT, winners TEXT, reveal TEXT);
            """)
            existing = db.execute(
                "SELECT plan,data,code,budget FROM identity WHERE id=1"
            ).fetchone()
            expected = (plan_hash, data_hash, code_hash, budget)
            if existing and existing != expected:
                raise ValueError("账本绑定的数据、代码或配置不一致")
            if existing is None:
                db.execute("INSERT INTO identity VALUES (1,?,?,?,?)", expected)
                self._event(
                    db,
                    "registered",
                    {"plan": plan_hash, "data": data_hash, "code": code_hash, "budget": budget},
                )

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=10)
        try:
            with db:
                yield db
        finally:
            db.close()

    def _event(self, db: sqlite3.Connection, kind: str, payload: dict[str, Any]) -> None:
        last = db.execute("SELECT seq,hash FROM events ORDER BY seq DESC LIMIT 1").fetchone()
        seq, previous = (last[0] + 1, last[1]) if last else (1, "genesis")
        body = {"seq": seq, "kind": kind, "payload": payload, "previous": previous}
        db.execute(
            "INSERT INTO events VALUES (?,?,?,?,?)",
            [seq, kind, canonical(payload), previous, digest(body)],
        )

    def start_batch(self, batch: str) -> None:
        with self.connection() as db:
            db.execute("INSERT INTO batches VALUES (?, 'searching', NULL, NULL)", [batch])
            self._event(db, "batch_started", {"batch": batch})

    def record_attempt(
        self,
        batch: str,
        arm: str,
        budget: int,
        candidate: str,
        status: str,
        score: float | None,
        payload: dict[str, Any],
    ) -> int:
        if arm not in self.arms or status not in {"ok", "failed"} or budget != self.budget:
            raise ValueError("试验状态、方法或预算非法")
        if (status == "ok" and (score is None or not math.isfinite(score))) or (
            status == "failed" and score is not None
        ):
            raise ValueError("成功必须有有限分数；失败不得伪造分数")
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            state = db.execute("SELECT state FROM batches WHERE batch=?", [batch]).fetchone()
            if state != ("searching",):
                raise ValueError("只允许在搜索阶段增加试验")
            count = db.execute(
                "SELECT count(*) FROM trials WHERE batch=? AND arm=?", [batch, arm]
            ).fetchone()[0]
            if count >= budget:
                raise ValueError("尝试预算已用尽，失败尝试也计数")
            entry = {
                "batch": batch,
                "arm": arm,
                "attempt": count + 1,
                "candidate": candidate,
                "status": status,
                "score": score,
                "payload": payload,
            }
            db.execute(
                "INSERT INTO trials VALUES (?,?,?,?,?,?,?)",
                [batch, arm, count + 1, candidate, status, score, canonical(payload)],
            )
            self._event(db, "attempt", entry)
        return int(count) + 1

    def trials(self, batch: str, arm: str) -> list[dict[str, Any]]:
        with self.connection() as db:
            rows = db.execute(
                "SELECT attempt,candidate,status,score,payload FROM trials "
                "WHERE batch=? AND arm=? ORDER BY attempt",
                [batch, arm],
            ).fetchall()
        return [
            {
                "attempt": row[0],
                "candidate": row[1],
                "status": row[2],
                "score": row[3],
                "payload": json.loads(row[4]),
            }
            for row in rows
        ]

    def freeze(self, batch: str, winners: dict[str, str], budget: int) -> str:
        if set(winners) != set(self.arms) or budget != self.budget:
            raise ValueError("所有搜索方法必须一起冻结")
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT state FROM batches WHERE batch=?", [batch]).fetchone() != (
                "searching",
            ):
                raise ValueError("批次不是搜索状态")
            for arm, winner in winners.items():
                rows = db.execute(
                    "SELECT candidate,status,score FROM trials WHERE batch=? AND arm=? "
                    "ORDER BY attempt",
                    [batch, arm],
                ).fetchall()
                if len(rows) != budget:
                    raise ValueError("方法的实际尝试数不等于预注册预算")
                valid = [row for row in rows if row[1] == "ok"]
                if not valid or winner != max(valid, key=lambda row: (row[2], row[0]))[0]:
                    raise ValueError("冻结结果必须是该方法已评估的验证集优胜者")
            commitment = digest({"batch": batch, "winners": winners, "budget": budget})
            db.execute(
                "UPDATE batches SET state='frozen',winners=? WHERE batch=?",
                [canonical(winners), batch],
            )
            self._event(
                db,
                "selection_frozen",
                {"batch": batch, "winners": winners, "budget": budget, "commitment": commitment},
            )
        return commitment

    def frozen_winners(self, batch: str) -> dict[str, str]:
        with self.connection() as db:
            row = db.execute("SELECT state,winners FROM batches WHERE batch=?", [batch]).fetchone()
        if row is None or row[0] != "frozen":
            raise ValueError("测试集在所有方法冻结前不可揭晓，已揭晓批次不得重复读取")
        return dict(json.loads(row[1]))

    def reveal(self, batch: str, results: dict[str, Any]) -> None:
        if set(results) != set(self.arms):
            raise ValueError("必须批量揭晓全部方法，不能选择性公布")
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT state FROM batches WHERE batch=?", [batch]).fetchone() != (
                "frozen",
            ):
                raise ValueError("只允许揭晓一次已冻结批次")
            db.execute(
                "UPDATE batches SET state='revealed',reveal=? WHERE batch=?",
                [canonical(results), batch],
            )
            self._event(db, "holdout_revealed", {"batch": batch, "results": results})

    def verify(self, expected_head: str | None = None) -> str:
        previous = "genesis"
        projected_trials = []
        projected_batches: dict[str, dict[str, Any]] = {}
        expected_seq = 1
        with self.connection() as db:
            for seq, kind, payload, prior, entry_hash in db.execute(
                "SELECT * FROM events ORDER BY seq"
            ):
                decoded = json.loads(payload)
                body = {"seq": seq, "kind": kind, "payload": decoded, "previous": prior}
                if seq != expected_seq or prior != previous or entry_hash != digest(body):
                    raise ValueError("审计事件链被修改")
                previous = entry_hash
                expected_seq += 1
                if kind == "registered":
                    identity = db.execute(
                        "SELECT plan,data,code,budget FROM identity WHERE id=1"
                    ).fetchone()
                    if identity != (
                        decoded["plan"],
                        decoded["data"],
                        decoded["code"],
                        decoded["budget"],
                    ):
                        raise ValueError("账本身份表与事件链不符")
                elif kind == "attempt":
                    projected_trials.append(
                        (
                            decoded["batch"],
                            decoded["arm"],
                            decoded["attempt"],
                            decoded["candidate"],
                            decoded["status"],
                            decoded["score"],
                            canonical(decoded["payload"]),
                        )
                    )
                elif kind == "batch_started":
                    projected_batches[decoded["batch"]] = {
                        "state": "searching",
                        "winners": None,
                        "reveal": None,
                    }
                elif kind == "selection_frozen":
                    projected_batches[decoded["batch"]].update(
                        state="frozen", winners=canonical(decoded["winners"])
                    )
                elif kind == "holdout_revealed":
                    projected_batches[decoded["batch"]].update(
                        state="revealed", reveal=canonical(decoded["results"])
                    )
            actual = db.execute("SELECT * FROM trials ORDER BY batch,arm,attempt").fetchall()
            if sorted(projected_trials) != actual:
                raise ValueError("试验表与事件链不符")
            batches = db.execute("SELECT * FROM batches ORDER BY batch").fetchall()
            projection = [
                (key, value["state"], value["winners"], value["reveal"])
                for key, value in sorted(projected_batches.items())
            ]
            if batches != projection:
                raise ValueError("封存状态表与事件链不符")
        if expected_head is not None and previous != expected_head:
            raise ValueError("审计链与已导出检查点不同")
        return previous
