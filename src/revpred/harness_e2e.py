"""End-to-end synthetic harness for validating Phase 1-6 pipelines with planted effects."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from revpred.config import get_project_config


def make_synthetic_orders(
    target_dir: Path | str,
    seed: int = 42,
    months: list[str] | None = None,
    n_products: int = 30,
) -> dict[str, Path]:
    """Generate five Olist-schema CSVs with a planted category-region effect.

    Planted effect:
    - Products in category 'toys' bought by customers in 'Southeast' receive a 3x price premium.
    - Statuses include valid 'delivered' orders and excluded 'canceled' orders to verify filtering.
    """
    _rng = np.random.default_rng(seed)
    out_path = Path(target_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if months is None:
        # 14 consecutive months: 2017-01 to 2018-02
        months = [
            f"2017-{m:02d}" for m in range(1, 13)
        ] + ["2018-01", "2018-02"]

    # 1. Categories and translations
    categories_spec = [
        ("brinquedos", "toys"),
        ("perfumaria", "perfumery"),
        ("telefonia", "telephony"),
    ]
    trans_path = out_path / "product_category_name_translation.csv"
    with open(trans_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["product_category_name", "product_category_name_english"])
        for port, eng in categories_spec:
            writer.writerow([port, eng])

    # 2. Products
    products = []
    prod_path = out_path / "olist_products_dataset.csv"
    with open(prod_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "product_id",
            "product_category_name",
            "product_name_lenght",
            "product_description_lenght",
            "product_photos_qty",
            "product_weight_g",
            "product_length_cm",
            "product_height_cm",
            "product_width_cm",
        ])
        for i in range(n_products):
            pid = f"prod_{i:04d}"
            cat_port = categories_spec[i % len(categories_spec)][0]
            products.append((pid, cat_port))
            writer.writerow([pid, cat_port, 40, 200, 2, 500, 20, 20, 20])

    # 3. States and region mapping (all states valid)
    state_pool = ["SP", "RJ", "MG", "PR", "RS", "BA", "PE", "GO", "AM"]

    # 4. Generate orders, customers, order_items
    orders_rows = []
    customers_rows = []
    items_rows = []

    order_idx = 0
    item_idx = 0

    cfg = get_project_config()
    state_to_region: dict[str, str] = {}
    for region_name, states in cfg["region_map"].items():
        for st in states:
            state_to_region[st] = region_name

    for m in months:
        # Generate recurring orders for each product to ensure causal history and entry
        for p_idx, (pid, cat_port) in enumerate(products):
            cat_eng = dict(categories_spec)[cat_port]
            # 2 orders per product per month in different states
            for order_k in range(2):
                order_id = f"ord_{order_idx:06d}"
                cust_id = f"cust_{order_idx:06d}"
                order_idx += 1

                st = state_pool[(p_idx + order_k) % len(state_pool)]
                region = state_to_region.get(st, "Southeast")

                # Timestamp in middle of month
                ts = f"{m}-15 12:00:00"
                orders_rows.append([
                    order_id,
                    cust_id,
                    "delivered",
                    ts,
                    ts,
                    ts,
                    ts,
                    ts,
                ])
                customers_rows.append([
                    cust_id,
                    f"uniq_{cust_id}",
                    "10000",
                    "Sao Paulo",
                    st,
                ])

                # Planted effect: category 'toys' in 'Southeast' gets 3x base price
                base_price = 50.0 + (p_idx % 5) * 10.0
                if cat_eng == "toys" and region == "Southeast":
                    price = round(base_price * 3.0, 2)
                else:
                    price = round(base_price, 2)

                item_idx += 1
                items_rows.append([
                    order_id,
                    1,
                    pid,
                    f"seller_{p_idx % 5}",
                    ts,
                    price,
                    15.0,
                ])

    # Add 2 canceled orders to verify status filtering exclusion
    canceled_order_id = f"ord_{order_idx:06d}"
    canceled_cust_id = f"cust_{order_idx:06d}"
    orders_rows.append([
        canceled_order_id,
        canceled_cust_id,
        "canceled",
        f"{months[0]}-10 10:00:00",
        f"{months[0]}-10 10:00:00",
        "",
        "",
        "",
    ])
    customers_rows.append([
        canceled_cust_id,
        f"uniq_{canceled_cust_id}",
        "10000",
        "Sao Paulo",
        "SP",
    ])
    items_rows.append([
        canceled_order_id,
        1,
        products[0][0],
        "seller_0",
        f"{months[0]}-10 10:00:00",
        999.0,
        0.0,
    ])

    # Write orders
    orders_path = out_path / "olist_orders_dataset.csv"
    with open(orders_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "order_id",
            "customer_id",
            "order_status",
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ])
        writer.writerows(orders_rows)

    # Write customers
    customers_path = out_path / "olist_customers_dataset.csv"
    with open(customers_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "customer_id",
            "customer_unique_id",
            "customer_zip_code_prefix",
            "customer_city",
            "customer_state",
        ])
        writer.writerows(customers_rows)

    # Write items
    items_path = out_path / "olist_order_items_dataset.csv"
    with open(items_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "order_id",
            "order_item_id",
            "product_id",
            "seller_id",
            "shipping_limit_date",
            "price",
            "freight_value",
        ])
        writer.writerows(items_rows)

    return {
        "orders": orders_path,
        "items": items_path,
        "customers": customers_path,
        "products": prod_path,
        "translation": trans_path,
    }


def reconcile_synthetic_csvs(raw_dir: Path | str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Reconcile synthetic CSVs without pandas dependency during raw read, producing valid order_lines."""
    cfg = get_project_config()
    raw_path = Path(raw_dir)

    # Region map
    state_to_region: dict[str, str] = {}
    for region_name, states in cfg["region_map"].items():
        for st in states:
            state_to_region[st] = region_name

    # Customers
    cust_map: dict[str, str] = {}
    with open(raw_path / "olist_customers_dataset.csv", "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            cust_map[r["customer_id"]] = state_to_region.get(r["customer_state"], "unknown")

    # Translations
    trans_map: dict[str, str] = {}
    with open(raw_path / "product_category_name_translation.csv", "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            trans_map[r["product_category_name"]] = r["product_category_name_english"]

    # Products
    prod_map: dict[str, str] = {}
    with open(raw_path / "olist_products_dataset.csv", "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            port = r["product_category_name"]
            prod_map[r["product_id"]] = trans_map.get(port, port)

    # Orders (filter out canceled/unavailable)
    exclude_statuses = set(cfg["exclude_statuses"])
    orders_map: dict[str, tuple[str, str]] = {}
    with open(raw_path / "olist_orders_dataset.csv", "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["order_status"] not in exclude_statuses:
                m = r["order_purchase_timestamp"][:7]
                orders_map[r["order_id"]] = (m, r["customer_id"])

    # Items reconciliation
    order_lines_rows = []
    raw_non_excl_items_sum = 0.0
    csv_order_lines_sum = 0.0
    unmatched_rev = 0.0

    with open(raw_path / "olist_order_items_dataset.csv", "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            oid = r["order_id"]
            if oid in orders_map:
                price = float(r["price"])
                raw_non_excl_items_sum += price
                m, cid = orders_map[oid]
                region = cust_map.get(cid)
                pid = r["product_id"]
                cat = prod_map.get(pid)
                if region is None or cat is None:
                    unmatched_rev += price
                else:
                    csv_order_lines_sum += price
                    order_lines_rows.append({
                        "order_id": oid,
                        "order_item_id": int(r["order_item_id"]),
                        "product_id": pid,
                        "category": cat,
                        "region": region,
                        "month": m,
                        "price": price,
                    })

    recon_abs_diff = abs((csv_order_lines_sum + unmatched_rev) - raw_non_excl_items_sum)
    df_order_lines = (
        pd.DataFrame(order_lines_rows)
        .sort_values(["month", "order_id", "order_item_id"])
        .reset_index(drop=True)
    )

    stats = {
        "raw_non_excl_items_sum": raw_non_excl_items_sum,
        "csv_order_lines_sum": csv_order_lines_sum,
        "unmatched_rev": unmatched_rev,
        "recon_abs_diff": recon_abs_diff,
        "n_order_lines": len(df_order_lines),
    }
    return df_order_lines, stats
