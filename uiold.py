import os
import streamlit as st
import tempfile   # temporary file

# my_app.py
st.title("My Streamlit App")
st.write("Welcome to my interactive dashboard!")

data = []
if 'data' not in st.session_state:
    st.session_state['data'] = []
# Add widgets (e.g., sliders, buttons, etc.)
user_input = st.text_input("Enter your name:")
st.write(f"Hello, {user_input}!")


option = st.selectbox(
   "Please select the data source.",
   ("File Upload", "Website URL", "Sitemap"),
   index=None,
   placeholder="Select contact method...",
)

def saver(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    return tmp_file_path

if option == "File Upload":
    with st.form(key="UPLOAD"):
        #user_input = st.text_input("Enter text here:")
        uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True)
        submit_button = st.form_submit_button(label="Submit")

    if submit_button:
        for uploaded_file in uploaded_files:
            print("UPLOADED FILE :: ", uploaded_file)
            ppath = saver(uploaded_file)
            st.session_state['data'].append(("file", ppath))

elif option == "Website URL":
    
    # Create the text input and submit button within a form
    
    with st.form(key="text_input_form"):
        user_input = st.text_input("Enter text here:")
        submit_button = st.form_submit_button(label="Submit")

    # Check if the submit button was clicked
    if submit_button:
        # Validate user input (optional)
        if user_input:
        # Append the input to the list
            st.session_state['data'].append(("web",user_input))
            # Clear the text input field
            st.session_state["user_input"] = ""  # Use session state to store temporary data
        else:
            st.warning("Please enter some text before submitting.")

st.write(st.session_state['data'])
    




def save_uploadedfile(uploadedfile):
     with open(os.path.join("tempDir",uploadedfile.name),"wb") as f:
         f.write(uploadedfile.getbuffer())
     return st.success("Saved File:{} to tempDir".format(uploadedfile.name))

