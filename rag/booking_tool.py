import sqlite3
from datetime import datetime
import os
from langchain_core.tools import Tool

DB_PATH = os.path.join("db", "car_sales.db")

def book_car(model: str, date: str, customer: str = "Anonymous") -> str:
    try:
        datetime.strptime(date, "%Y-%m-%d")
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO bookings (model, date, customer) VALUES (?, ?, ?)",
            (model, date, customer)
        )
        conn.commit()
        conn.close()
        return f"✅ Booking confirmed for {model} on {date} for {customer}."
    except Exception as e:
        return f"❌ Booking failed: {e}"


def get_booking_tool():
    return Tool(
        name="book_car",
        func=book_car,
        description="Use this tool to book a car by providing model, date (YYYY-MM-DD), and optionally customer name."
    )
