import sqlite3


def main() -> None:
    conn = sqlite3.connect("sentiments.db")
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM sentiments")
        rows = cur.fetchall()
        if not rows:
            print("Bảng sentiments đang rỗng.")
        else:
            for r in rows:
                print(r)
    finally:
        conn.close()


if __name__ == "__main__":
    main()


