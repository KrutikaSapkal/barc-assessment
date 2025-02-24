import os
import pandas as pd

# Initialize an empty list to store the extracted data
data = []

filename = os.path.join('../data', 'raw_data_begnin', 'BlueCoat_Large.webgateway')
absolute_path = os.path.abspath(filename)
print(f"Absolute path of the file: {absolute_path}")

# Read the log file
with open(absolute_path, 'r', encoding='utf-8') as file:
    print("Reading the log file...")
    for line_number, line in enumerate(file, start=1):
        # Remove leading/trailing whitespace
        line = line.strip()

        # Split the line into parts, preserving quoted fields
        parts = []
        in_quotes = False
        current_part = ""
        for char in line:
            if char == '"':
                in_quotes = not in_quotes  # Toggle the in_quotes flag
                current_part += char
            elif char == ' ' and not in_quotes:
                parts.append(current_part)
                current_part = ""
            else:
                current_part += char
        if current_part:
            parts.append(current_part)

        # Extract the relevant fields
        date_time = parts[0][1:]  # Remove the leading '['
        timezone = parts[1][:-1]  # Remove the trailing ']'
        user_name = parts[2].strip('"')
        source_ip = parts[3]
        destination_ip = parts[4]
        port = parts[5]  # Port number
        status_code = parts[6]  # Status code
        cache_result = parts[7]  # Cache result
        http_method = parts[8].strip('"')  # HTTP method
        url_requested = parts[9]  # URL requested
        http_version = parts[10].strip('"HTTP/')  # HTTP version
        domain_classification = parts[11].strip('"')
        risk_classification = parts[12].strip('"')
        mime_type = parts[13].strip('"')
        bytes_sent = parts[14]
        bytes_received = parts[15]
        user_agent_string = parts[16].strip('"')
        web_referrer_string = parts[17].strip('"')
        url_meta1 = parts[18].strip('"')
        url_meta2 = parts[19].strip('"')
        url_meta3 = parts[20].strip('"')
        url_meta4 = parts[21].strip('"')

        # Split the URL requested into HTTP method and URL
        # The URL requested field is in the format: "GET http://example.com"
        if ' ' in url_requested:
            http_method, url_requested = url_requested.split(' ', 1)
            http_method = http_method.strip('"')  # Remove quotes from HTTP method
            url_requested = url_requested.strip('"')  # Remove quotes from URL

        # Append the extracted data as a dictionary to the list
        data.append({
            'dateTime': date_time,
            'timezone': timezone,
            'userName': user_name,
            'sourceIP': source_ip,
            'destinationIP': destination_ip,
            'port': port,
            'statusCode': status_code,
            'cacheResult': cache_result,
            'httpMethod': http_method,
            'urlRequested': url_requested,
            'httpVersion': http_version,
            'domainClassification': domain_classification,
            'riskClassification': risk_classification,
            'mimeType': mime_type,
            'bytesSent': bytes_sent,
            'bytesReceived': bytes_received,
            'userAgentString': user_agent_string,
            'webReferrerString': web_referrer_string,
            'urlMeta1': url_meta1,
            'urlMeta2': url_meta2,
            'urlMeta3': url_meta3,
            'urlMeta4': url_meta4
        })

        if line_number % 100 == 0:  # Print a status update every 100 lines
            print(f"Processed {line_number} lines...")

print(f"Total lines processed: {line_number}")

# Convert the list of dictionaries to a DataFrame
df = pd.DataFrame(data)
print("Converted extracted data to DataFrame")

# Display the DataFrame
print(df)

folder_path = "../data/cleaned_begnin"  # Replace with your desired folder location

# Ensure the folder exists; if not, create it
os.makedirs(folder_path, exist_ok=True)
print(f"Folder path checked/created: {folder_path}")

# Save the DataFrame to a CSV file in the specified folder
file_path = os.path.join(folder_path, "log_data.csv")
df.to_csv(file_path, index=False)
print(f"Data saved to CSV file: {file_path}")
