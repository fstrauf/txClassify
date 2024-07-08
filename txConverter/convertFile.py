import csv
import json
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)

def parse_date(date_str, input_format):
    try:
        return datetime.strptime(date_str, input_format).strftime('%d/%m/%Y')
    except ValueError as e:
        logging.error(f"Date parsing error: {e}. Input: {date_str}, Format: {input_format}")
        return None

def normalize_amount(amount_str):
    try:
        return float(amount_str.replace(',', '').replace(' ', ''))
    except ValueError:
        logging.error(f"Amount normalization error. Input: {amount_str}")
        return None

def read_csv(file_path, bank_config, default_currency):
    transactions = []
    with open(file_path, 'r', encoding='utf-8-sig') as file:
        for _ in range(bank_config.get('skip_rows', 0)):
            next(file)
        reader = csv.DictReader(file, delimiter=bank_config.get('separator', ','))
        
        for row_num, row in enumerate(reader, start=1):
            try:
                date = parse_date(row[bank_config['column_mapping']['date']], bank_config['date_format'])
                amount = normalize_amount(row[bank_config['column_mapping']['amount']])
                description = row[bank_config['column_mapping']['description']]
                currency = row.get(bank_config['column_mapping'].get('currency'), default_currency)
                
                if all([date, amount, description, currency]):
                    # Apply the values_negative flag
                    if bank_config.get('convert_values_to_negative', False):
                        amount *= -1
                    transactions.append([date, amount, description, currency])
                else:
                    logging.warning(f"Skipping row {row_num} due to missing or invalid data: {row}")
            except KeyError as e:
                logging.error(f"Missing key {e} in row {row_num}: {row}")
            except Exception as e:
                logging.error(f"Error processing row {row_num}: {e}")

    return transactions

def write_csv(file_path, transactions):
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['date', 'amount', 'description', 'currency'])
        writer.writerows(transactions)
        
def map_columns(columns):
    return {k.strip().lower(): v.strip().lower() for k, v in columns.items() if v}

def main():
    try:
        with open('txConverter/config.json', 'r') as config_file:
            config = json.load(config_file)

        all_transactions = []
        for bank in config['banks']:
            file_path = bank['file_path']
            default_currency = bank.get('defaultCurrency', 'USD')
            transactions = read_csv(file_path, bank, default_currency)
            all_transactions.extend(transactions)

        if all_transactions:
            write_csv('txConverter/data/target.csv', all_transactions)
            logging.info(f"Successfully processed {len(all_transactions)} transactions.")
        else:
            logging.warning("No transactions were processed. Check your input files and configuration.")

    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing config.json: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    main()