import streamlit as st
from PIL import Image
import json
import pandas as pd
import tensorflow as tf
import numpy as np
import os

# Set page config at the absolute top
st.set_page_config(page_title="Vision Cart Assistant", layout="wide", page_icon="🛒")

# Load the AI model (it recognizes 1,000 objects)
@st.cache_resource
def load_model():
    return tf.keras.applications.MobileNetV2(weights='imagenet')

try:
    model = load_model()
except Exception as e:
    st.error(f"Failed to load AI Model: {e}")

def ai_detect(image):
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    
    preds = model.predict(img_array, verbose=0)
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=5)[0] # Increased to top 5 for better matching
    
    # Extract all top predictions to check against keywords
    labels = [d[1].replace("_", " ").lower() for d in decoded]
    highest_conf = float(decoded[0][2]) * 100

    # Enhanced Broad Keyword Mapping
    for label in labels:
        if any(x in label for x in ["watch", "clock", "timer"]):
            return "watch", highest_conf
        if any(x in label for x in ["phone", "mobile", "cellular", "ipod"]):
            return "phone", highest_conf
        if any(x in label for x in ["laptop", "notebook", "computer", "screen"]):
            return "laptop", highest_conf
        if any(x in label for x in ["bottle", "flask", "water"]):
            return "bottle", highest_conf
        if any(x in label for x in ["bag", "backpack", "luggage", "purse"]):
            return "bag", highest_conf
        if any(x in label for x in ["refrigerator", "fridge", "freezer"]):
            return "fridge", highest_conf
        if any(x in label for x in ["basket", "hamper", "bin"]):
            return "basket", highest_conf
        if any(x in label for x in ["headphone", "earphone", "earbud", "audio"]):
            return "earbuds", highest_conf

    # Fallback to top prediction if no keywords match
    return decoded[0][1].replace("_", " ").lower(), highest_conf

# Initialize Session States Safely
if 'my_cart' not in st.session_state:
    st.session_state.my_cart = []
if 'toast_msg' not in st.session_state:
    st.session_state.toast_msg = None

st.title("🛒 Vision Cart - Smart Shopping Assistant")
st.write("Upload an image or scan a product to check pricing, budgets, and manage your cart.")
st.divider()

# --- TOAST MESSAGE WORKAROUND FOR RERUN ---
if st.session_state.toast_msg:
    st.toast(st.session_state.toast_msg, icon="✅")
    st.session_state.toast_msg = None

# --- SIDEBAR: SHOPPING CART MANAGEMENT ---
if len(st.session_state.my_cart) > 0:
    st.sidebar.subheader("📝 Your Shopping List")
    total_bill = 0
    remove_index = None

    for i, product in enumerate(st.session_state.my_cart):
        col_item, col_btn = st.sidebar.columns([3, 1])
        col_item.write(f"**{i+1}. {product['item']}**\n₹{product['price']}")
        if col_btn.button("❌", key=f"remove_item_{i}", help="Remove item"):
            remove_index = i
        st.sidebar.write("---")
        total_bill += product['price']

    if remove_index is not None:
        st.session_state.my_cart.pop(remove_index)
        st.rerun()

    st.sidebar.divider()
    st.sidebar.metric("Total Bill", f"₹{total_bill}")

    if st.sidebar.button("Clear Cart", key="clear_cart_btn", use_container_width=True):
        st.session_state.my_cart = []
        st.rerun()
else:
    st.sidebar.info("🛒 Your cart is empty")

st.sidebar.header("🕹️ Control Panel")
user_budget = st.sidebar.number_input("Set your Budget (₹)", min_value=1, value=5000, step=500)
manual_choice = st.sidebar.selectbox(
    "Identify Scanned Item:",
    ["Auto Detect", "phone", "watch", "laptop", "fridge", "bottle", "basket", "earbuds", "bag"]
)

# --- PRODUCTS DATABASE UTILITIES ---
@st.cache_data
def load_products():   
    if not os.path.exists("products.json"):
        # Creative backup generation so the app never crashes
        mock_data = [
            {"name": "phone", "category": "electronics", "price": 15000},
            {"name": "watch", "category": "electronics", "price": 2500},
            {"name": "laptop", "category": "electronics", "price": 45000},
            {"name": "fridge", "category": "appliances", "price": 22000},
            {"name": "bottle", "category": "kitchen", "price": 300},
            {"name": "basket", "category": "kitchen", "price": 450},
            {"name": "earbuds", "category": "electronics", "price": 1999},
            {"name": "bag", "category": "fashion", "price": 1200}
        ]
        with open("products.json", "w") as f:
            json.dump(mock_data, f, indent=4)
    
    try:
        with open("products.json") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error reading products.json: {e}")   
        return []

def get_product_details(name):
    data = load_products()
    clean_name = name.lower().replace(" ", "").strip()
    for item in data:
        clean_item = item["name"].lower().replace(" ", "").strip()
        # Advanced bi-directional fuzzy substring evaluation
        if clean_name in clean_item or clean_item in clean_name:
            return item
    return None

def get_alternative(category, current_name, budget):
    data = load_products()   
    filtered = [
        item for item in data
        if item["category"].lower() == category.lower()
        and item["name"].lower() != current_name.lower()
        and item["price"] <= budget
    ]
    if filtered:
         return sorted(filtered, key=lambda x: x["price"])[0] # returns cheapest available alternative
    return None

# --- CORE USER INTERFACE APP LAYER ---
input_method = st.radio("Choose Input Method:", ["Upload Image", "Use Camera"], horizontal=True)

if input_method == "Upload Image":
     file = st.file_uploader("Upload Product Photo", type=["jpg", "png", "jpeg"])
else:
     file = st.camera_input("Scan Product")
    
if file is None:
    st.info("📷 Please upload or scan a product image to begin parsing data.")
    st.stop()

try:
    img = Image.open(file)
except:
    st.error("Corrupted or invalid image asset uploaded.")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    st.image(img, caption="📸 Scanned Source Item", use_container_width=True)
    st.caption(f"📐 **Resolution Dimensions:** {img.size[0]} x {img.size[1]} pixels")

if manual_choice == "Auto Detect":
    with st.spinner("AI Engine is executing image classification via MobileNetV2..."):
        item_name, confidence = ai_detect(img)
else:
    item_name = manual_choice
    confidence = 100.0

with col2:
    st.subheader(f"🔍 Detection Output: {item_name.title()}")
    st.write(f"🎯 **Model Confidence Rating:** {confidence:.2f}%")

    product_info = get_product_details(item_name)

    # Threshold Status UI
    if confidence < 35:
        st.error("❌ High classification uncertainty! Re-align the subject and try again.")
    elif confidence < 65:
        st.warning("⚠️ Micro-predictions yield weak accuracy values. Check manual override if incorrect.")
    else:
        st.success("✅ Secure image classification confidence verified!")

    if product_info:
        st.write(f"📦 **Database Category Match:** {product_info['category'].title()}")
        st.write(f"💵 **Base Market Valuation:** ₹{product_info['price']}")
        st.markdown(f"### Item Profile: `{item_name.title()}`")

        price = product_info["price"]

        # Budget Checks
        if price > user_budget:
            diff = price - user_budget
            percent = (diff / user_budget) * 100
            st.error(f"⚠️ Financial Boundary Breached! Exceeded allocated budget by **₹{diff}** ({percent:.1f}%)")
        else:
            st.success("✅ Line item fits within designated target budget profiles.")
            progress = min(price / user_budget, 1.0)
            st.progress(progress, text=f"Total Budget Depletion Scale: {int(progress*100)}%")

        # Recommendations Engine
        alt = get_alternative(product_info["category"], product_info["name"], user_budget)
        if alt:
            st.info(f"💡 **Budget-Friendly Alternative:** Try **{alt['name'].title()}** retail valued at only **₹{alt['price']}**")
        else:
            st.warning("📉 No cheaper valid intra-category entries parsed below target ceiling.")

        st.write("---")
        
        # Data Visualization Matrix
        chart_data = pd.DataFrame({
            "Financial Metric": ["Product Price", "Target Budget Ceiling"],
            "Cost Vector (₹)": [float(price), float(user_budget)]
        }).set_index("Financial Metric")
        st.bar_chart(chart_data, y="Cost Vector (₹)")

        # Persistent Cart State Handler
        if st.button(f"🛒 Append {item_name.title()} to Local Session Cart", key="add_to_cart_btn", use_container_width=True):
            st.session_state.my_cart.append({
                "item": item_name.title(),
                "price": product_info["price"]
            })
            st.session_state.toast_msg = f"Successfully appended {item_name.title()} to transaction array!"
            st.rerun()
    else:
        st.warning("⚠️ Entity classified successfully, but signature does not exist in backend inventory indices.")

# --- INVENTORY DATA VIEWER ---
st.divider()
with st.expander("🔐 Administrative Control: View Master Product Database Matrix"):
    try:
         data = load_products()
         if data:
              st.dataframe(data, use_container_width=True)
    except:
         st.write("Initialization fault mapping JSON databases configuration.")
    
