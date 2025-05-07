# generate_car_sales_db.py
import sqlite3
from datetime import datetime, timedelta
import random

# Create and connect to SQLite DB file
conn = sqlite3.connect("car_sales.db")
cursor = conn.cursor()

# Create table
cursor.execute("""
CREATE TABLE IF NOT EXISTS car_sales (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    sale_date TEXT NOT NULL
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS bookings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model TEXT,
    date TEXT,
    customer TEXT
)
""")

# Sample car models
car_models = [
    "Toyota Corolla 2020",
    "Honda Civic 2023",
    "Hyundai Elantra 2021",
    "Kia Sportage 2022",
    "Ford F-150 2019",
    "Tesla Model 3 2022"
]

# Generate 100 random sales from the past 30 days
today = datetime.today()
for _ in range(100):
    model = random.choice(car_models)
    quantity = random.randint(1, 5)
    sale_date = today - timedelta(days=random.randint(0, 30))
    cursor.execute(
        "INSERT INTO car_sales (model, quantity, sale_date) VALUES (?, ?, ?)",
        (model, quantity, sale_date.strftime("%Y-%m-%d"))
    )

# Commit and close
conn.commit()
conn.close()

print("✅ car_sales.db created with 100 sample rows.")
