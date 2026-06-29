import pandas as pd

files = [
    r"data\processed\autism_2024_test.xlsx",
    r"data\processed\autism_2024_train.xlsx"
]
label_map = {
    "Control": 0,
    "Autism": 1
}

for file in files:
    print(f"Processing: {file}")

    excel = pd.ExcelFile(file)
    output_sheets = {}

    for sheet_name in excel.sheet_names:
        df = pd.read_excel(file, sheet_name=sheet_name)

        if "Group" not in df.columns:
            raise ValueError(f"'Group' column not found in {file}, sheet {sheet_name}")

        # Clean text spaces
        df["Group"] = df["Group"].astype(str).str.strip()

        # Replace labels
        df["Group"] = df["Group"].replace(label_map)

        # Check result
        print(f"Sheet: {sheet_name}")
        print(df["Group"].value_counts(dropna=False))

        output_sheets[sheet_name] = df

    # Save back to same file
    with pd.ExcelWriter(file, engine="openpyxl") as writer:
        for sheet_name, df in output_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

print("Done. Autism Group column converted to 0/1.")