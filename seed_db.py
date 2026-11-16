import os
import sqlite3
import random
from datetime import datetime, timedelta

DB_PATH = "dummy_ecommerce.db"


def create_dummy_database(db_path: str = DB_PATH, num_customers: int = 10_000, num_orders: int = 250_000):
    if os.path.exists(db_path):
        os.remove(db_path)

    print(f"[SETUP] Membuat database dummy '{db_path}' (2 tabel berelasi)...")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Tabel 1: Master Customers (10.000 baris)
    cur.execute("""
        CREATE TABLE customers (
            customer_id INTEGER PRIMARY KEY,
            customer_name TEXT NOT NULL,
            city TEXT NOT NULL,
            tier TEXT NOT NULL
        );
    """)

    # Tabel 2: Transaksi Orders (250.000 baris, TANPA indeks pada Foreign Key customer_id)
    cur.execute("""
        CREATE TABLE orders (
            order_id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id INTEGER NOT NULL,
            product_code TEXT NOT NULL,
            total_amount REAL NOT NULL,
            status TEXT NOT NULL,
            notes TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
        );
    """)

    # Isi data dummy customers (hanya ~25 pelanggan yang berstatus VIP agar selektif)
    cities = ["Bandung", "Jakarta", "Surabaya", "Yogyakarta", "Medan"]
    customer_records = []
    for cid in range(1, num_customers + 1):
        tier = "VIP" if cid % 400 == 0 else random.choice(["REGULAR", "SILVER", "GOLD"])
        customer_records.append((cid, f"Customer-{cid}", random.choice(cities), tier))

    cur.executemany(
        "INSERT INTO customers (customer_id, customer_name, city, tier) VALUES (?, ?, ?, ?);",
        customer_records
    )

    # Isi data dummy orders (250.000 baris)
    base_date = datetime(2026, 1, 1)
    statuses = ["COMPLETED", "PENDING", "SHIPPED", "CANCELLED"]
    order_records = []

    for i in range(1, num_orders + 1):
        cust_id = 1042 if i % 500 == 0 else random.randint(1, num_customers)
        prod_code = f"PRD-{random.randint(100, 999)}"
        amount = round(random.uniform(50000, 2500000), 2)
        status = random.choice(statuses)
        notes = f"Transaksi pelanggan #{cust_id} untuk order ke-{i} dengan catatan pengiriman standar."
        created_at = (base_date + timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M:%S")
        order_records.append((cust_id, prod_code, amount, status, notes, created_at))

    cur.executemany("""
        INSERT INTO orders (customer_id, product_code, total_amount, status, notes, created_at)
        VALUES (?, ?, ?, ?, ?, ?);
    """, order_records)

    conn.commit()
    conn.close()
    print(f"[SELESAI] Tabel 'customers' ({num_customers:,} baris) & 'orders' ({num_orders:,} baris) siap diuji.")


if __name__ == "__main__":
    create_dummy_database()