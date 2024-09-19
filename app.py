import streamlit as st
import pandas as pd
import openai
import pinecone
from pinecone import ServerlessSpec
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
import asyncio
import time
import logging
from threading import Thread
from slack_sdk.errors import SlackApiError
from datetime import datetime, date
import bcrypt
from dotenv import load_dotenv
import io

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN")

# Constants
ATTENDANCE_INDEX = "attendance-index"
LEAVE_INDEX = "leave-index"
STAFF_INDEX = "staff-index"
index_name = "annual-leave"

# Initialize Pinecone
pinecone_initialized = False
attendance_index = None
leave_index = None
staff_index = None

def init_pinecone(api_key):
    global pinecone_initialized, attendance_index, leave_index, staff_index
    try:
        pinecone.init(api_key=api_key, environment="us-east-1")
        if ATTENDANCE_INDEX not in pinecone.list_indexes():
            pinecone.create_index(ATTENDANCE_INDEX, dimension=768)
        if LEAVE_INDEX not in pinecone.list_indexes():
            pinecone.create_index(LEAVE_INDEX, dimension=768)
        if STAFF_INDEX not in pinecone.list_indexes():
            pinecone.create_index(STAFF_INDEX, dimension=768)

        attendance_index = pinecone.Index(ATTENDANCE_INDEX)
        leave_index = pinecone.Index(LEAVE_INDEX)
        staff_index = pinecone.Index(STAFF_INDEX)

        pinecone_initialized = True
        st.success("Connected to Pinecone indexes successfully")
    except Exception as e:
        st.error(f"Error connecting to Pinecone: {str(e)}")
        return False

# Initialize Slack app
app = AsyncApp(token=SLACK_BOT_TOKEN)

# Function to run the Slack bot
def run_slack_bot():
    async def start_bot():
        handler = AsyncSocketModeHandler(app, SLACK_APP_TOKEN)
        await handler.start_async()
    asyncio.run(start_bot())

# Cached function to generate embeddings
def get_embedding(text):
    try:
        response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
        return tuple(response['data'][0]['embedding'])
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return None

# Function to upload data to Pinecone
def upload_to_pinecone(records):
    try:
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        index = pinecone.Index(index_name)
        index.upsert(vectors=records)
        return True, "Data uploaded to Pinecone successfully!"
    except Exception as e:
        logger.error(f"Error uploading to Pinecone: {e}")
        return False, f"Error uploading data to Pinecone: {str(e)}"

# Function to query Pinecone
async def query_pinecone(query):
    try:
        index = pinecone.Index(index_name)
        query_embedding = get_embedding(query)
        if query_embedding is None:
            return None
        results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
        if results['matches']:
            context = " ".join([match['metadata']['text'] for match in results['matches']])
            return context
        else:
            return None
    except Exception as e:
        logger.error(f"Error querying Pinecone: {e}")
    return None

# Function to process attendance
def record_attendance(user_id):
    st.subheader("Record Daily Attendance")
    entry_date = st.date_input("📆 Date", date.today())
    entry_time = st.time_input("🕒 Entry Time", datetime.now().time())
    exit_time = st.time_input("🕒 Exit Time", datetime.now().time())

    entry_time_12hr = entry_time.strftime("%I:%M %p")
    exit_time_12hr = exit_time.strftime("%I:%M %p")

    if st.button("📝 Submit Attendance"):
        timestamp = datetime.now().isoformat()
        data = {
            "id": f"attendance_{timestamp}",
            "employee_id": user_id,
            "entry_date": entry_date.isoformat(),
            "entry_time": entry_time_12hr,
            "exit_time": exit_time_12hr
        }

        vector = get_embedding(f"Attendance: {user_id} {entry_date} {entry_time_12hr} {exit_time_12hr}")
        if store_in_pinecone(attendance_index, data, vector):
            st.success("✅ Attendance recorded successfully!")

# Function to submit leave request
def submit_leave_request(user_id):
    st.subheader("Submit Leave Request")
    leave_from = st.date_input("📅 Leave From")
    leave_to = st.date_input("📅 Leave To")
    leave_type = st.selectbox("🏷️ Leave Type", ["Annual Leave", "Sick Leave", "Personal Leave", "Other"])
    purpose = st.text_area("📝 Purpose of Leave")

    if st.button("📨 Submit Leave Request"):
        timestamp = datetime.now().isoformat()
        data = {
            "id": f"leave_{timestamp}",
            "employee_id": user_id,
            "leave_from": leave_from.isoformat(),
            "leave_to": leave_to.isoformat(),
            "leave_type": leave_type,
            "purpose": purpose,
            "status": "pending"
        }

        vector = get_embedding(f"Leave: {user_id} {leave_from} {leave_to} {leave_type} {purpose}")
        if store_in_pinecone(leave_index, data, vector):
            st.success("✅ Leave request submitted successfully!")

# Login function
def login():
    st.header("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == os.getenv("ADMIN_USERNAME") and password == os.getenv("ADMIN_PASSWORD"):
            st.session_state['user_role'] = 'admin'
            return True
        else:
            user_data = query_staff_by_username(username)
            if user_data and check_password(password, user_data['password']):
                st.session_state['user_role'] = 'staff'
                st.session_state['user_id'] = user_data['id']
                return True
        st.error("Invalid username or password")
    return False

# Hashing and checking passwords
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def check_password(password, hashed_password):
    return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))

# Admin panel
def admin_panel():
    st.header("Admin Panel")
    if st.button("Add New User"):
        with st.form("add_user_form"):
            new_username = st.text_input("Username")
            new_password = st.text_input("Password", type="password")
            new_email = st.text_input("Email")
            new_role = st.selectbox("Role", ["staff", "manager"])
            submit_button = st.form_submit_button("Add User")
            if submit_button:
                add_user(new_username, new_password, new_email, new_role)

    if st.button("View Existing Users"):
        users = fetch_users()
        st.subheader("Existing Users")
        st.write(users)

    if st.button("View All Leave Requests"):
        leave_data = fetch_leave_requests("All")
        st.subheader("All Leave Requests")
        st.write(leave_data)

# Staff panel
def staff_panel(user_id):
    st.header("Staff Panel")
    menu = ["📅 Daily Attendance", "🏖️ Leave Request"]
    choice = st.sidebar.radio("Select Option", menu)

    if choice == "📅 Daily Attendance":
        record_attendance(user_id)
    elif choice == "🏖️ Leave Request":
        submit_leave_request(user_id)

# Main function
def main():
    st.title("🗓️ LEAVES-BUDDY")
    
    # Initialize Pinecone
    if not pinecone_initialized:
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if pinecone_api_key:
            init_pinecone(pinecone_api_key)
        else:
            st.error("Pinecone API key not found in environment variables.")
            return

    # Start Slack bot
    if 'bot_running' not in st.session_state:
        st.session_state.bot_running = False

    if st.button("Start Slack Bot", disabled=st.session_state.bot_running):
        st.session_state.bot_running = True
        st.write("Starting Slack bot...")
        thread = Thread(target=run_slack_bot)
        thread.start()
        st.success("Slack bot is running!")

    # Login and role-based panel
    if 'user_role' not in st.session_state:
        if login():
            st.experimental_rerun()
    else:
        if st.sidebar.button("Logout"):
            st.session_state.clear()
            st.experimental_rerun()

        if st.session_state['user_role'] == 'admin':
            admin_panel()
        elif st.session_state['user_role'] == 'staff':
            staff_panel(st.session_state['user_id'])

if __name__ == "__main__":
    main()
