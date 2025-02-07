import argparse
import csv

def main():
    parser = argparse.ArgumentParser(description="""
        1) Read a CSV of 10k examples (no repeated queries).
        2) Sort by descending Query Formality Score.
        3) Keep top 5000.
        4) Among those, count how many have sbert>0.7 and Response Formality Score<0.5.
        5) Print summary.
    """)
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the CSV file.")
    args = parser.parse_args()

    # Read CSV
    data = []
    with open(args.csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)

    if not data:
        print("No rows found in the CSV.")
        return

    # Parse float safely
    def safe_float(x, default=0.0):
        try:
            return float(x)
        except:
            return default

    # Sort by descending Query Formality Score
    data.sort(key=lambda r: safe_float(r.get("Query Formality Score"), 0.0), reverse=True)

    # Keep top 5000
    top_5k = data[:5000]

    # Count how many in top_5k have sbert>0.7 and Response Formality Score<0.5
    count_pass = 0
    for row in top_5k:
        sbert_val = safe_float(row.get("sbert_similarity"), 0.0)
        resp_formality = safe_float(row.get("Response Formality Score"), 1.0)
        if sbert_val > 0.7 and resp_formality < 0.5:
            count_pass += 1

    # Print results
    print(f"Total rows in CSV: {len(data)}")
    print(f"Top 5000 by Query Formality Score selected.")
    print(f"Among those 5000, {count_pass} have sbert>0.7 and Response Formality Score<0.5.")

if __name__ == "__main__":
    main()