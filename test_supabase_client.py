import os, sys, pandas as pd
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()  # <- esto hace que os.getenv lea del .env

URL = os.getenv("SUPABASE_URL", "https://rhmbfrqkyeusahtgpfgc.supabase.co")
KEY = os.getenv("SUPABASE_ANON_KEY", "")
TABLE = os.getenv("SUPABASE_TABLE", "ventas_frutto")

def q(c): return f'"{c}"'

def main():
    if not KEY:
        print("Falta SUPABASE_ANON_KEY"); sys.exit(1)

    client = create_client(URL, KEY)
    cols = ["product","sales_rep","invoice_#","total_revenue","total_profit_$","reqs._date"]
    select_list = ",".join(q(c) for c in cols)

    resp = client.table(TABLE).select(select_list).limit(5).execute()
    print("HTTP OK" if resp.data is not None else "SIN DATA")
    df = pd.DataFrame(resp.data or [])
    print("Rows:", len(df), "Cols:", list(df.columns))
    print(df.head())

    from datetime import date, timedelta
    start = (date.today() - timedelta(days=365)).isoformat() + "T00:00:00Z"
    end   = (date.today() + timedelta(days=1)).isoformat() + "T00:00:00Z"

    resp2 = (client.table(TABLE)
                .select(select_list)
                .gte(q("reqs._date"), start)
                .lt(q("reqs._date"), end)
                .limit(5)
                .execute())
    df2 = pd.DataFrame(resp2.data or [])
    print("\nRango último año → rows:", len(df2))
    if not df2.empty:
        print(df2[["reqs._date","sales_rep","total_revenue"]].head())

if __name__ == "__main__":
    main()
