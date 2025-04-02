import csv
import json
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)


def parse_date(date_str, input_format):
    try:
        return datetime.strptime(date_str, input_format).strftime("%d/%m/%Y")
    except ValueError as e:
        logging.error(
            f"Date parsing error: {e}. Input: {date_str}, Format: {input_format}"
        )
        return None


def normalize_amount(amount_str):
    try:
        return float(amount_str.replace(",", "").replace(" ", ""))
    except ValueError:
        logging.error(f"Amount normalization error. Input: {amount_str}")
        return None


def get_description(row, bank_config):
    # Handle description field(s)
    if isinstance(bank_config["column_mapping"]["description"], list):
        # Get all non-empty values from the specified fields
        desc_parts = [
            str(row.get(field, "")).strip()
            for field in bank_config["column_mapping"]["description"]
        ]
        desc_parts = [part for part in desc_parts if part]  # Remove empty strings

        # Check if we have fallback configuration and no description parts
        if not desc_parts and "description_fallback" in bank_config:
            fallback = bank_config["description_fallback"]

            # Check if conditional field exists and meets the condition
            if "conditional_field" in fallback and "condition" in fallback:
                cond_field = fallback["conditional_field"]
                cond_value = row.get(cond_field, "").strip()

                # Currently supporting 'empty' condition
                if fallback["condition"] == "empty" and not cond_value:
                    # Use primary field as description
                    primary_field = fallback["primary_field"]
                    if primary_field in row and row[primary_field].strip():
                        return row[primary_field].strip()

            # If we have a composite format, use it as a fallback
            if "composite_format" in fallback:
                try:
                    return fallback["composite_format"].format(**row)
                except KeyError as e:
                    logging.warning(f"Could not format description using template: {e}")
                    # Fall through to default handling

        # Default handling
        return " | ".join(desc_parts) if desc_parts else "Unknown"
    else:
        # Single field description
        return row.get(bank_config["column_mapping"]["description"], "Unknown")


def read_csv(file_path, bank_config, default_currency):
    transactions = []
    with open(file_path, "r", encoding="utf-8-sig") as file:
        for _ in range(bank_config.get("skip_rows", 0)):
            next(file)
        reader = csv.DictReader(file, delimiter=bank_config.get("separator", ","))

        for row_num, row in enumerate(reader, start=1):
            try:
                date = parse_date(
                    row[bank_config["column_mapping"]["date"]],
                    bank_config["date_format"],
                )
                amount = normalize_amount(row[bank_config["column_mapping"]["amount"]])
                description = get_description(row, bank_config)

                currency = row.get(
                    bank_config["column_mapping"].get("currency"), default_currency
                )

                # Handle direction for Wise CSV
                if "direction" in bank_config["column_mapping"]:
                    direction = row[bank_config["column_mapping"]["direction"]]
                    if direction == "OUT":
                        amount = -abs(amount)
                    elif direction == "IN":
                        amount = abs(amount)

                if date and amount is not None and currency:
                    # Apply the values_negative flag only if direction is not handled
                    if "direction" not in bank_config[
                        "column_mapping"
                    ] and bank_config.get("convert_values_to_negative", False):
                        amount *= -1
                    transactions.append([date, amount, description, currency])
                else:
                    missing_fields = []
                    if not date:
                        missing_fields.append("date")
                    if amount is None:
                        missing_fields.append("amount")
                    if not currency:
                        missing_fields.append("currency")

                    logging.warning(
                        f"Skipping row {row_num} due to missing required fields: {', '.join(missing_fields)}. Row data: {row}"
                    )
            except KeyError as e:
                logging.error(f"Missing key {e} in row {row_num}: {row}")
            except Exception as e:
                logging.error(f"Error processing row {row_num}: {e}")

    return transactions


def write_csv(file_path, transactions):
    with open(file_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["date", "amount", "description", "currency"])
        writer.writerows(transactions)


def map_columns(columns):
    return {k.strip().lower(): v.strip().lower() for k, v in columns.items() if v}


def main():
    try:
        with open("txConverter/config.json", "r") as config_file:
            config = json.load(config_file)

        all_transactions = []
        for bank in config["banks"]:
            file_path = bank["file_path"]
            default_currency = bank.get("defaultCurrency", "USD")
            transactions = read_csv(file_path, bank, default_currency)
            all_transactions.extend(transactions)

        if all_transactions:
            write_csv("txConverter/data/target.csv", all_transactions)
            logging.info(
                f"Successfully processed {len(all_transactions)} transactions."
            )
        else:
            logging.warning(
                "No transactions were processed. Check your input files and configuration."
            )

    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing config.json: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
