"""Verify WooCommerce MCP order refunds against the isolated fixture database."""

import argparse
import json
from pathlib import Path

from gyms.toolathlon_gym.episode import Episode
from tests.toolathlon_gym.check_environment import probe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    episode = Episode("wc-refund-analysis-notion", args.output / "episode", image=args.image,
                      subagent_port=8025)

    def call(tool, arguments):
        result = json.loads(probe(episode, "woocommerce-mcp", "dist/index.js", tool, arguments))
        return json.loads(result["content"][0]["text"])

    try:
        episode.start()
        for statuses in (["completed"], ["completed", "refunded"]):
            rows = call("woo_orders_list", {"perPage": 1, "status": statuses})
            assert rows and rows[0]["status"] in statuses, "Order status array filter loses matching orders"
        sql = """SELECT json_agg(r ORDER BY id) FROM
                 (SELECT order_id, id, COALESCE(reason, '') AS reason,
                  (-abs(amount))::text AS total FROM wc.refunds) r"""
        expected = json.loads(episode.command("exec", episode.pg, "psql", "-U", "eigent",
                              "-d", "toolathlon_gym", "-Atc", sql).stdout)
        assert expected, "Regression fixture must contain refunds"
        ids = json.loads(episode.command("exec", episode.pg, "psql", "-U", "eigent",
                         "-d", "toolathlon_gym", "-Atc",
                         "SELECT json_agg(id ORDER BY id) FROM wc.orders").stdout)
        orders = []
        for order_id in sorted({row["order_id"] for row in expected}):
            rows = call("woo_orders_list", {"perPage": 1, "page": ids.index(order_id) + 1,
                                             "orderby": "id", "order": "asc"})
            assert len(rows) == 1 and rows[0]["id"] == order_id, "Order pagination is inconsistent"
            orders.extend(rows)
        actual = [{"order_id": order["id"], **refund}
                  for order in orders for refund in order.get("refunds", [])]
        result = {"expected": expected, "actual": actual, "orders": len(orders)}
        (args.output / "result.json").write_text(json.dumps(result, indent=2))
        assert sorted(actual, key=lambda r: r["id"]) == expected, "Order list hides or corrupts refunds"
        for order_id in sorted({row["order_id"] for row in expected}):
            order = call("woo_orders_get", {"orderId": order_id})
            wanted = [{k: v for k, v in row.items() if k != "order_id"}
                      for row in expected if row["order_id"] == order_id]
            assert order["refunds"] == wanted, "Single-order refund summaries disagree"
        print(f"PASS: {len(expected)} refunds exposed consistently through list/get MCP calls")
    finally:
        episode.close()


if __name__ == "__main__":
    main()
