import streamlit as st
import sys
import pkg_resources
import pinecone
from datetime import datetime, date, time
from openai import OpenAI
import os
from dotenv import load_dotenv
import pandas as pd
import io
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
import bcrypt
import threading

# Load environment variables
load_dotenv()

# Set page config at the very beginning
st.set_page_config(page_title="LEAVES-BUDDY", page_icon="🗓️", layout="wide")

# Constants
ATTENDANCE_INDEX = "attendance-index"
LEAVE_INDEX = "leave-index"
STAFF_INDEX = "staff-index"

# Initialize Pinecone
pinecone_initialized = False
attendance_index = None
leave_index = None
staff_index = None

def init_pinecone(api_key):
    global pinecone_initialized, attendance_index, leave_index, staff_index
    try:
        # Initialize Pinecone
        pinecone.init(api_key=api_key, environment="gcp-starter")
        
        # Check and create the indexes if they do not exist
        if ATTENDANCE_INDEX not in pinecone.list_indexes():
            pinecone.create_index(ATTENDANCE_INDEX, dimension=768)
        if LEAVE_INDEX not in pinecone.list_indexes():
            pinecone.create_index(LEAVE_INDEX, dimension=768)
        if STAFF_INDEX not in pinecone.list_indexes():
            pinecone.create_index(STAFF_INDEX, dimension=768)
        
        # Connect to the indexes
        attendance_index = pinecone.Index(ATTENDANCE_INDEX)
        leave_index = pinecone.Index(LEAVE_INDEX)
        staff_index = pinecone.Index(STAFF_INDEX)
        
        pinecone_initialized = True
        st.success("Connected to Pinecone indexes successfully")
        return True
    except Exception as e:
        st.error(f"Error connecting to Pinecone: {str(e)}")
        return False

# Initialize OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Slack
slack_bot_token = os.getenv("SLACK_BOT_TOKEN")
slack_app_token = os.getenv("SLACK_APP_TOKEN")

if not slack_bot_token or not slack_app_token:
    raise ValueError("Slack tokens are missing. Please check your .env file.")

slack_client = WebClient(token=slack_bot_token)
app = App(token=slack_bot_token)

def init_slack():
    try:
        slack_client.auth_test()
        st.success("Connected to Slack successfully")
        handler = SocketModeHandler(app, slack_app_token)
        thread = threading.Thread(target=handler.start)
        thread.start()
    except SlackApiError as e:
        st.error(f"Error connecting to Slack: {e}")

def send_slack_notification(message, channel="#attendance-notifications"):
    try:
        response = slack_client.chat_postMessage(channel=channel, text=message)
        return True
    except SlackApiError as e:
        st.error(f"Error sending Slack notification: {e.response['error']}")
        return False

@app.event("app_mention")
def handle_mention(event, say):
    user = event['user']
    say(f"Hi there, <@{user}>! How can I help you with attendance or leave requests?")

def create_embedding(text):
    try:
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Error creating embedding: {str(e)}")
        return None

def store_in_pinecone(index, data, vector):
    if not pinecone_initialized:
        st.warning("Pinecone is not initialized. Data will not be stored.")
        return False
    try:
        string_data = {k: str(v) if v is not None else "" for k, v in data.items()}
        index.upsert(vectors=[(string_data['id'], vector, string_data)])
        return True
    except Exception as e:
        st.error(f"Error storing data in Pinecone: {str(e)}")
        return False

def query_gpt(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You are a helpful assistant analyzing attendance and leave data."},
                      {"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Unable to generate analysis: {str(e)}"

def calculate_working_hours(entry_time, exit_time):
    try:
        entry = datetime.strptime(entry_time, "%I:%M %p")
        exit = datetime.strptime(exit_time, "%I:%M %p")
        duration = exit - entry
        return max(0, duration.total_seconds() / 3600)
    except ValueError:
        return 0

def fetch_attendance(employee_id, from_date, to_date):
    try:
        query_vector = create_embedding(f"Attendance of {employee_id} from {from_date} to {to_date}")
        if query_vector:
            results = attendance_index.query(
                vector=query_vector,
                top_k=100,
                include_metadata=True
            )
            attendance_data = [
                r['metadata'] for r in results['matches']
                if from_date <= r['metadata'].get('entry_date', '') <= to_date
                and (employee_id == "All" or r['metadata'].get('employee_id') == employee_id)
            ]
            return attendance_data
        return []
    except Exception as e:
        st.error(f"An error occurred while querying Pinecone: {str(e)}")
        return []

def fetch_leave_requests(employee_id=None):
    try:
        query_vector = create_embedding(f"Leave requests {employee_id if employee_id else 'all users'}")
        if query_vector:
            results = leave_index.query(vector=query_vector, top_k=100, include_metadata=True)
            leave_data = [
                r['metadata'] for r in results['matches']
                if (employee_id == "All" or r['metadata'].get('employee_id') == employee_id)
            ]
            return leave_data
        return []
    except Exception as e:
        st.error(f"An error occurred while querying leave data: {str(e)}")
        return []

def download_to_excel(data, employee_id):
    df = pd.DataFrame(data)
    filename = f"{employee_id}_attendance_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.xlsx"
    
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False)
    output.seek(0)
    
    st.download_button(
        label="📥 Download Excel File",
        data=output,
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def check_password(password, hashed_password):
    return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))

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

def query_staff_by_username(username):
    query_vector = create_embedding(username)
    results = staff_index.query(vector=query_vector, top_k=1, include_metadata=True)
    if results['matches']:
        return results['matches'][0]['metadata']
    return None

def add_user(username, password, email, role):
    user_id = f"user_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    hashed_password = hash_password(password).decode('utf-8')  # Convert bytes to string
    user_data = {
        "id": user_id,
        "username": username,
        "password": hashed_password,
        "email": email,
        "role": role
    }
    vector = create_embedding(f"{username} {email} {role}")
    if store_in_pinecone(staff_index, user_data, vector):
        st.success("User added successfully!")
    else:
        st.error("Failed to add user.")

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

    st.subheader("View Attendance")
    employee_id = st.text_input("Employee ID (leave blank to view all)")
    from_date = st.date_input("From Date", date.today().replace(day=1))
    to_date = st.date_input("To Date", date.today())

    if st.button("View Attendance"):
        attendance_data = fetch_attendance(employee_id, from_date.isoformat(), to_date.isoformat())
        if attendance_data:
            for entry in attendance_data:
                entry_time = entry.get('entry_time', 'N/A')
                exit_time = entry.get('exit_time', 'N/A')
                working_hours = calculate_working_hours(entry_time, exit_time)
                st.write(f"Date: {entry.get('entry_date')}")
                st.write(f"Entry: {entry_time}, Exit: {exit_time}")
                st.write(f"Total hours worked: {working_hours:.2f}")
                st.write("---")
            download_to_excel(attendance_data, employee_id)

def staff_panel(user_id):
    st.header("Staff Panel")
    menu = ["📅 Daily Attendance", "🏖️ Leave Request"]
    choice = st.sidebar.radio("Select Option", menu)

    if choice == "📅 Daily Attendance":
        record_attendance(user_id)
    elif choice == "🏖️ Leave Request":
        submit_leave_request(user_id)

def fetch_users():
    query_vector = create_embedding("staff users")
    results = staff_index.query(vector=query_vector, top_k=100, include_metadata=True)
    return [r['metadata'] for r in results['matches']]

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
        
        vector = create_embedding(f"Attendance: {user_id} {entry_date} {entry_time_12hr} {exit_time_12hr}")
        
        if store_in_pinecone(attendance_index, data, vector):
            st.success("✅ Attendance recorded successfully!")
            working_hours = calculate_working_hours(entry_time_12hr, exit_time_12hr)
            st.info(f"🕒 Total hours worked: {working_hours:.2f}")
            
            slack_message = f"New attendance recorded: Employee {user_id} worked for {working_hours:.2f} hours on {entry_date}"
            send_slack_notification(slack_message)
        else:
            st.error("Failed to record attendance.")

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
        
        vector = create_embedding(f"Leave: {user_id} {leave_from} {leave_to} {leave_type} {purpose}")
        
        if store_in_pinecone(leave_index, data, vector):
            st.success("✅ Leave request submitted successfully!")
            
            slack_message = f"New leave request: Employee {user_id} requested {leave_type} from {leave_from} to {leave_to}"
            send_slack_notification(slack_message)
        else:
            st.error("Failed to submit leave request.")

def main():
    st.title("🗓️ LEAVES-BUDDY")

    if not pinecone_initialized:
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if pinecone_api_key:
            init_pinecone(pinecone_api_key)
        else:
            st.error("Pinecone API key not found in environment variables.")
            return

    init_slack()

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
