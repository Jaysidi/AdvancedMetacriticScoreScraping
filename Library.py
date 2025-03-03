import os
import glob
import pandas as pd
import sqlite3
import zipfile
from datetime import datetime


def get_unique_filename(directory: str, filename: str, extension: str) -> str:
    """
    Generates a unique filename by appending an index if the file already exists.

    Args:
        directory (str): Directory where the file will be saved.
        filename (str): Desired filename without extension.
        extension (str): File extension (e.g., 'csv', 'xlsx', 'json').

    Returns:
        str: A unique filename with the correct format.
    """
    base_path = os.path.join(directory, f"{filename}.{extension}")
    if not os.path.exists(base_path):
        return base_path  # No conflict, return the original name

    index = 1
    while True:
        new_path = os.path.join(directory, f"{filename}_{index}.{extension}")
        if not os.path.exists(new_path):
            return new_path
        index += 1


def merge_csv_files(input_folder: str, output_file: str, output_format: str = "csv", pattern: str = "*.csv",
                    save_to_db: bool = False, db_name: str = "merged_data.db", table_name: str = "merged_table",
                    compress: bool = False):
    """
    Merges all CSV files in a given folder into a single file, optionally saves to SQLite,
    and compresses the original CSV files into a ZIP archive.

    Args:
        input_folder (str): Folder containing the CSV files.
        output_file (str): Name of the output file (without extension).
        output_format (str): Output format ('csv', 'xlsx', 'json').
        pattern (str): Pattern to use with glob.
        save_to_db (bool): Whether to save the merged data to a SQLite database (default: False).
        db_name (str): Name of the SQLite database (default: 'merged_data.db').
        table_name (str): Name of the table where data will be stored (default: 'merged_table').
        compress (bool): Whether to compress the original CSV files into a ZIP archive (default: False).

    Returns:
        str: Path of the merged file.
    """
    # Find all CSV files in the folder using glob
    csv_files = glob.glob(os.path.join(input_folder, pattern))

    if not csv_files:
        raise FileNotFoundError("No CSV files found in the folder.")

    dataframes = []  # List to store DataFrames

    # Read each CSV file and append to the list
    for file in csv_files:
        df = pd.read_csv(file)  # Read the CSV
        dataframes.append(df)

    # Merge all DataFrames into one
    merged_df = pd.concat(dataframes, ignore_index=True)

    # Remove duplicate rows
    merged_df.drop_duplicates(inplace=True)

    # Generate a unique output file path
    output_path = get_unique_filename(input_folder, output_file, output_format)

    # Save the merged data based on the selected format
    if output_format == "csv":
        merged_df.to_csv(output_path, index=False)
    elif output_format == "xlsx":
        merged_df.to_excel(output_path, index=False, engine="openpyxl")
    elif output_format == "json":
        merged_df.to_json(output_path, orient="records", indent=4)
    else:
        raise ValueError("Unsupported output format. Choose 'csv', 'xlsx', or 'json'.")

    print(f"💾 Merged file saved as: {output_path}")

    # Save to SQLite database if enabled
    if save_to_db:
        with sqlite3.connect(db_name) as conn:
            merged_df.to_sql(table_name, conn, if_exists="replace", index=False)
            print(f"💾 Data saved to SQLite database: {db_name}, Table: {table_name}")

    # Compress original CSV files into a ZIP archive if enabled
    if compress:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")  # Get current date-time
        zip_filename = f"{output_file}_{timestamp}.zip"
        zip_path = os.path.join(input_folder, zip_filename)

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file in csv_files:
                zipf.write(file, os.path.basename(file))  # Add each CSV file to the ZIP
                os.remove(file)  # Delete the original CSV file

        print(f"💾 CSV files compressed and saved as: {zip_path}")
        print("🗑️ Original CSV files deleted after compression.")

    return output_path


def k_to_thousands(n):
    """
    Transforms 'kilo' string numbers to thousands. eg. 1.3k -> 1300
    Args:
      n: str, number to transform.
    Returns:
      str, transformed number.
    """
    if n[-1] == 'k':
        return str(int(float(n[:-1]) * 1000))
    else:
        return n