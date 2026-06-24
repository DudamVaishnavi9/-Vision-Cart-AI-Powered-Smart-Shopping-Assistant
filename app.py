import streamlit as st 
from PIL import Image
import json
import pandas as pd
import tensorflow as tf
import numpy as np

# Load the AI model (it recognizes 1,000 objects)
@st.cache_resource
def load_model():
    return tf.keras.applications.MobileNetV2(weights='imagenet')

model = load_model()

def ai_detect(image):
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    
    preds = model.predict(img_array, verbose=0)
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=3)[0]
    
    label = decoded[0][1].replace("_", " ").lower()
    conf = float(decoded[0][2]) * 100

    if any(x in label for x in ["watch", "clock"]):
        return "watch", conf
    elif any(x in label for x in ["phone", "mobile", "cellular"]):
        return "phone", conf
    elif "laptop" in label:
        return "laptop", conf
    elif "bottle" in label:
        return "bottle", conf
    elif "bag" in label:
        return "bag", conf
    elif "refrigerator" in label:
       return "fridge", conf
    elif "basket" in label:
        return "basket", conf
    elif "headphone" in label or "earphone" in label:
       return "earbuds", conf

    return label.replace("_", " "), conf

st.set_page_config(page_title="Vision Cart Assistant", layout="wide")

if 'my_cart' not in st.session_state:
    st.session_state.my_cart = []

st.title("🛒 Vision Cart - Smart Shopping Assistant")
st.write("Upload an image to detect product and get details")
st.divider()

# --- SHOW TOTAL BILL IN SIDEBAR ---
if len(st.session_state.my_cart) > 0:
    st.sidebar.subheader("📝 Your Shopping List")
    total_bill = 0
    remove_index = None

    for i, product in enumerate(st.session_state.my_cart):
        st.sidebar.write(f"{i+1}. {product['item']} - ₹{product['price']}")
        # FIX: Changed key to ensure uniqueness even if duplicate items are added
        if st.sidebar.button(f"❌ Remove {product['item']}", key=f"remove_item_{i}"):
            remove_index = i
        st.sidebar.write("---")
        total_bill += product['price']

    if remove_index is not None:
        st.session_state.my_cart.pop(remove_index)
        st.rerun()

    st.sidebar.divider()
    st.sidebar.metric("Total Bill", f"₹{total_bill}")

    if st.sidebar.button("Clear Cart", key="clear_cart_btn"):
        st.session_state.my_cart = []
        st.rerun()
else:
    st.sidebar.info("🛒 Your cart is empty")

st.sidebar.header("Use Control Panel")
user_budget = st.sidebar.number_input("Set your Budget (₹)", min_value=100, value=5000, step=500)
manual_choice = st.sidebar.selectbox(
    "Identify Scanned Item:",
    ["Auto Detect", "phone", "watch", "laptop", "fridge", "bottle", "basket", "earbuds", "bag"]
)

if user_budget == 0:
    st.sidebar.warning("Please set a budget greater than ₹0")

@st.cache_data
def load_products():   
    try:
        with open("products.json") as f:
            return json.load(f)
    except:
        st.error("products.json file missing!")   
        return []

def get_product_details(name):
    data = load_products()
    clean_name = name.lower().replace(" ", "")
    for item in data:
        clean_item = item["name"].lower().replace(" ", "")
        if clean_name in clean_item or clean_item in clean_name:
            return item
    return None

def get_alternative(category, current_name, budget):
    data = load_products()   
    filtered = [
        item for item in data
        if item["category"] == category
        and item["name"] != current_name
        and item["price"] <= budget
    ]
    if filtered:
         return sorted(filtered, key=lambda x: x["price"])[0] # cheapest
    return None

input_method = st.radio("Choose Input Method:", ["Upload Image", "Use Camera"])

if input_method == "Upload Image":
     file = st.file_uploader("Upload Product Photo", type=["jpg", "png", "jpeg"])
else:
     file = st.camera_input("Scan Product")
    
if file is None:
    st.info("📷 Please upload or scan a product image to begin.")
    st.stop()

try:
    img = Image.open(file)
except:
    st.error("Invalid image file")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    st.image(img, caption="📸 Scanned Item", use_container_width=True)
    st.markdown(f"**Resolution:** {img.size[0]} x {img.size[1]}")

if manual_choice == "Auto Detect":
    with st.spinner("AI is analysing the image..."):
        item_name, confidence = ai_detect(img)
else:
    item_name = manual_choice
    confidence = 100.0

with col2:
    st.subheader(f"🔍 Result: {item_name.replace('_', ' ').title()}")
    st.write(f"🎯 Confidence: {confidence:.2f}%")

    product_info = get_product_details(item_name)

    if confidence < 40:
        st.error("❌ Very low confidence! Try another image.")
    elif confidence < 70:
        st.warning("⚠️ Moderate confidence. Result may not be accurate.")
    else:
        st.success("✅ High confidence prediction!")

    if product_info:
        st.write(f"**Category:** {product_info['category'].title()}")
        st.write(f"**Market Price:** ₹{product_info['price']}")
        st.markdown(f"### 🏷️ {item_name.title()}")

        price = product_info["price"]

        if price > user_budget:
            diff = price - user_budget
            percent = (diff / user_budget) * 100 if user_budget > 0 else 0
            st.error(f"💰 Budget Exceeded by ₹{diff} ({percent:.1f}%)")
        else:
            st.success("✅ This fits your budget perfectly!")
            progress = min(product_info["price"] / user_budget, 1.0) if user_budget > 0 else 0
            st.progress(progress, text=f"Budget Usage: {int(progress*100)}%")

        alt = get_alternative(product_info["category"], product_info["name"], user_budget)
        if alt:
            st.info(f"💡 Better option: Try **{alt['name'].title()}** at ₹{alt['price']}")
        else:
            st.warning("No cheaper alternatives found.")

        st.write("---")
        chart_data = pd.DataFrame({
            "Label": ["Product Price", "Your Budget"],
            "Amount": [float(product_info["price"]), float(user_budget)]
        }).set_index("Label")
        st.bar_chart(chart_data)

        if st.button(f"🛒 Add {item_name.title()} to Cart", key="add_to_cart_btn"):
            st.session_state.my_cart.append({
                "item": item_name.title(),
                "price": product_info["price"]
            })
            st.toast(f"Added {item_name.title()} to cart!", icon="✅")
            st.rerun()
    else:
        st.warning("⚠️ Product detected but not in database. Try another item.")

st.divider()
with st.expander("Admin: View Product List"):
    try:
         data = load_products()
         if data:
              st.table(data)
    except:
         st.write("Please ensure products.json exists.")
