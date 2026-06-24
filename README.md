# -Vision-Cart-AI-Powered-Smart-Shopping-Assistant

Vision Cart is a lightweight Streamlit web application that acts as an automated shopping assistant. It uses a TensorFlow backend to scan items via a camera feed or image upload, cross-references them against a local inventory database, and tracks user spending against a custom budget.

## Project Structure

```text
├── app.py            # Main Streamlit application logic
├── products.json     # Local catalog database containing prices and categories
└── requirements.txt  # Python dependency manifest
```

## Setup & Local Installation

To run this application locally, you will need a 64-bit installation of Python (tested on Python 3.11/3.12). 

1. Clone or download this project folder to your machine.
2. Open a terminal inside the project root directory and create a virtual environment:
   ```bash
   python -m venv venv
   ```
3. Activate the virtual environment:
   * **Windows (PowerShell):** `.\venv\Scripts\Activate.ps1`
   * **Mac/Linux:** `source venv/bin/activate`
4. Install the required application dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Launch the local development server:
   ```bash
   python -m streamlit run app.py
   ```

## Configuration

The application references a local asset inventory stored inside `products.json`. You can modify, append, or swap out product records directly within this file using standard JSON formatting:

```json
[   
    {"name": "watch", "price": 1500, "category": "accessories"},
    {"name": "phone", "price": 25000, "category": "electronics"},
    {"name": "bottle", "price": 40, "category": "home"}
]
```

## How It Works

* **Image Classification:** Images passed through the camera input or file uploader are normalized to 224x224 pixels and processed using a MobileNetV2 model pre-trained on ImageNet data.
* **Control Panel Override:** If an item cannot be cleanly identified due to poor lighting or framing, users can manually select the item from the sidebar dropdown menu.
* **Budget Metrics:** The UI dynamically calculates cost differences and visualizes budget consumption using basic native charts. If an item exceeds the designated financial cap, the system parses the active category block to locate and display the lowest-priced fallback option.
* **State Management:** Session states are utilized to allow adding items, removing specific indices, and tracking cumulative billing totals without resetting variables on page refreshes.

