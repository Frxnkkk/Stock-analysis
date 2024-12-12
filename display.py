import pandas as pd
from datetime import datetime
from IPython.display import HTML, display

def display_truck_data(csv_file):
    # Read CSV file
    df = pd.read_csv(csv_file)
    
    # Create HTML table with improved styling
    html_content = """
    <style>
        .truck-container {
            padding: 20px;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        }
        .truck-table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            background-color: white;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            border-radius: 8px;
            overflow: hidden;
            margin: 20px 0;
        }
        .truck-table th {
            background-color: #2E7D32;
            color: white;
            padding: 16px;
            text-align: left;
            font-size: 16px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .truck-table td {
            padding: 20px 16px;
            border-bottom: 1px solid #e0e0e0;
            font-size: 15px;
            vertical-align: middle;
        }
        .truck-table tr:last-child td {
            border-bottom: none;
        }
        .truck-table tr:hover {
            background-color: #f5f9f5;
        }
        .timestamp {
            color: #1a1a1a;
            font-family: monospace;
            font-size: 14px;
        }
        .license-plate {
            font-weight: 600;
            color: #1a1a1a;
            font-size: 16px;
        }
        .confidence {
            font-family: monospace;
            color: #2E7D32;
            font-weight: 600;
        }
        .truck-table img {
            max-width: 400px;
            height: auto;
            border-radius: 6px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: transform 0.2s ease;
        }
        .truck-table img:hover {
            transform: scale(1.02);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        .row-spacer {
            height: 12px;
        }
    </style>
    <div class="truck-container">
        <table class="truck-table">
            <thead>
                <tr>
                    <th>Timestamp</th>
                    <th>License Plate</th>
                    <th>Confidence</th>
                    <th>Image</th>
                </tr>
            </thead>
            <tbody>
    """
    
    # Process each row
    for _, row in df.iterrows():
        try:
            # Format timestamp
            timestamp = datetime.strptime(str(row['timestamp']), '%Y%m%d-%H%M%S')
            formatted_timestamp = timestamp.strftime('%Y-%m-%d %H:%M:%S')
            
            # Format confidence as percentage
            confidence_pct = f"{float(row['confidence']) * 100:.1f}%"
            
            # Add row to table
            html_content += f"""
                <tr>
                    <td><span class="timestamp">{formatted_timestamp}</span></td>
                    <td><span class="license-plate">{row['license_plate']}</span></td>
                    <td><span class="confidence">{confidence_pct}</span></td>
                    <td><img src="{row['image_path']}" alt="Truck {row['license_plate']}"></td>
                </tr>
            """
        except (ValueError, TypeError) as e:
            print(f"Skipping row due to error: {e}")
            continue
    
    html_content += """
            </tbody>
        </table>
    </div>
    """
    
    # Display the HTML table in the notebook
    display(HTML(html_content))

# Example usage - run this in a Jupyter notebook cell:
csv_file = "./detection_log.csv"
display_truck_data(csv_file)