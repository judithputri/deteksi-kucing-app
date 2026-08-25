import streamlit as st
import text
from model_engine import predict_image

# 1. Konfigurasi Halaman
st.set_page_config(page_title="Prediksi Kucing", page_icon="🐱")

# 2. Otentikasi Ala-Ala (Dummy Login)
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    st.title(text.AUTH_HEADER)
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        # Password/Username dummy sederhana
        if username == "admin" and password == "kucing123":
            st.session_state['logged_in'] = True
            st.rerun()
        else:
            st.error("Username atau password salah!")
    st.stop()

# 3. Halaman Utama (Setelah Login Berhasil)
st.title(text.TITLE)
st.write(text.WELCOME_MSG)

uploaded_file = st.file_uploader("Pilih Gambar Kucing...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Gambar yang Diunggah", use_container_width=True)
    
    if st.button("Deteksi Sekarang"):
        with st.spinner("Memproses gambar..."):
            label, accuracy = predict_image(uploaded_file)
            st.success(f"**Hasil Prediksi:** {label}")
            st.info(f"**Tingkat Keyakinan:** {accuracy:.2f}%")

st.markdown("---")
st.caption(text.FOOTER)