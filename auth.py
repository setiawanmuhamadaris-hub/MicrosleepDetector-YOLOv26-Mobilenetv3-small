import streamlit as st

def check_login(username, password):
    ADMIN_USERNAME = "admin"
    ADMIN_PASSWORD = "admin123"
    return username == ADMIN_USERNAME and password == ADMIN_PASSWORD

def show_login_form():
    st.sidebar.markdown("### 🔐 Login Admin")
    username = st.sidebar.text_input("Username", key="login_username")
    password = st.sidebar.text_input("Password", type="password", key="login_password")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Login", use_container_width=True):
            if check_login(username, password):
                st.session_state.logged_in = True
                st.session_state.show_login = False
                st.rerun()
            else:
                st.sidebar.error("Username atau password salah!")
    with col2:
        if st.button("Batal", use_container_width=True):
            st.session_state.show_login = False
            st.rerun()

def init_session_state():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'show_login' not in st.session_state:
        st.session_state.show_login = False
    if 'global_conf_threshold' not in st.session_state:
        st.session_state.global_conf_threshold = 0.5
    if 'global_alert_duration' not in st.session_state:
        st.session_state.global_alert_duration = 2.0
    if 'camera_mode' not in st.session_state:
        st.session_state.camera_mode = "Lokal (OpenCV)"
