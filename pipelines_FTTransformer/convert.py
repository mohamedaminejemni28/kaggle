import pandas as pd

files = [
    "data/processed/young_old_2024_train.xlsx",
    "data/processed/young_old_2024_test.xlsx"
]

label_map = {
    "Control": 0,
    "control": 0,
    "CONTROL": 0,

    "Flatfoot": 1,
    "flatfoot": 1,
    "Flat Foot": 1,
    "flat foot": 1,

    "Pes Planus": 1,
    "pes planus": 1,
    "PES PLANUS": 1,

    "0": 0,
    "1": 1,
    0: 0,
    1: 1
}

for file in files:
    print(f"\nProcessing: {file}")

    excel = pd.ExcelFile(file)
    output_sheets = {}

    for sheet_name in excel.sheet_names:
        df = pd.read_excel(file, sheet_name=sheet_name)

        if "Group" not in df.columns:
            raise ValueError(f"Group column not found in {file}, sheet {sheet_name}")

        # Clean labels
        df["Group"] = df["Group"].astype(str).str.strip()

        # Map labels to 0/1
        df["Group"] = df["Group"].replace(label_map)

        # Force numeric conversion
        df["Group"] = pd.to_numeric(df["Group"], errors="coerce")

        # Check if any value failed
        if df["Group"].isna().any():
            print("Problematic Group values:")
            print(df["Group"].value_counts(dropna=False))
            raise ValueError("Some Group labels could not be converted to 0/1.")

        # Convert to integer
        df["Group"] = df["Group"].astype(int)

        print(f"Sheet: {sheet_name}")
        print(df["Group"].value_counts(dropna=False))
        print("Unique values:", df["Group"].unique())
        print("Dtype:", df["Group"].dtype)

        output_sheets[sheet_name] = df

    # Save back to same Excel file
    with pd.ExcelWriter(file, engine="openpyxl") as writer:
        for sheet_name, df in output_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

print("\nDone. Young/Older Group column converted to numeric 0/1.")