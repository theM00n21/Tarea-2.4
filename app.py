import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from io import BytesIO
import plotly.express as px
import matplotlib.pyplot as plt

# ------------------- CONFIGURACIÓN APP -------------------
st.set_page_config(page_title="K-Means con PCA", layout="wide")

st.title("📌 Clustering con K-Means y PCA")
st.caption("Hecho por **Patricio Conrado**")

# ------------------- SUBIR ARCHIVO -------------------
st.sidebar.header("Subir datos")
uploaded_file = st.sidebar.file_uploader("Selecciona un archivo CSV", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("### Vista previa de los datos")
    st.dataframe(data.head())

    numeric_cols = data.select_dtypes(include=['float64', 'int64', 'float32', 'int32']).columns.tolist()

    if len(numeric_cols) < 2:
        st.warning("El archivo debe contener al menos dos columnas numéricas.")
    else:
        st.sidebar.header("Configuración del Modelo")

        selected_cols = st.sidebar.multiselect(
            "Selecciona las columnas numéricas para el clustering:",
            numeric_cols,
            default=numeric_cols
        )

        k = st.sidebar.slider("Número de clusters (k):", 1, 10, 3)
        n_components = st.sidebar.radio("Visualización PCA:", [2, 3], index=0)

        # ------------------- PARÁMETROS SOLICITADOS EN LA TAREA -------------------
        init_option = st.sidebar.selectbox("init", ["k-means++", "random"], index=0)
        max_iter = st.sidebar.slider("max_iter", min_value=10, max_value=1000, value=300, step=10)
        n_init_option = st.sidebar.selectbox("n_init", ["auto", 1, 5, 10, 20, 50], index=3)
        random_state_val = st.sidebar.number_input("random_state", min_value=0, max_value=100000, value=0, step=1)
        # -------------------------------------------------------------------------

        X = data[selected_cols]

        kmeans = KMeans(
            n_clusters=k,
            init=init_option,
            max_iter=int(max_iter),
            n_init=n_init_option,
            random_state=int(random_state_val)
        )

        kmeans.fit(X)
        data['Cluster'] = kmeans.labels_

        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)

        pca_cols = [f'PCA{i+1}' for i in range(n_components)]
        pca_df = pd.DataFrame(X_pca, columns=pca_cols)
        pca_df['Cluster'] = data['Cluster']

        st.subheader("Distribución Original (PCA)")
        if n_components == 2:
            fig_before = px.scatter(pca_df, x='PCA1', y='PCA2')
        else:
            fig_before = px.scatter_3d(pca_df, x='PCA1', y='PCA2', z='PCA3')
        st.plotly_chart(fig_before, use_container_width=True)

        st.subheader(f"Datos Agrupados con K-Means (k = {k})")
        if n_components == 2:
            fig_after = px.scatter(pca_df, x='PCA1', y='PCA2', color=pca_df['Cluster'].astype(str))
        else:
            fig_after = px.scatter_3d(pca_df, x='PCA1', y='PCA2', z='PCA3', color=pca_df['Cluster'].astype(str))
        st.plotly_chart(fig_after, use_container_width=True)

        st.subheader("Centroides en Espacio PCA")
        centroides_pca = pd.DataFrame(pca.transform(kmeans.cluster_centers_), columns=pca_cols)
        st.dataframe(centroides_pca)

        st.subheader("Método del Codo")
        if st.button("Calcular Elbow"):
            inertias = []
            K = range(1, 11)
            for i in K:
                km = KMeans(n_clusters=i, n_init="auto", random_state=42)
                km.fit(X)
                inertias.append(km.inertia_)

            fig2, ax2 = plt.subplots(figsize=(8, 6))
            plt.plot(list(K), inertias, 'o-')
            plt.title('Método del Codo')
            plt.xlabel('Número de Clusters (k)')
            plt.ylabel('Inercia (SSE)')
            plt.grid(True)
            st.pyplot(fig2)

        st.subheader("Descargar datos con clusters")
        buffer = BytesIO()
        data.to_csv(buffer, index=False)
        buffer.seek(0)
        st.download_button(
            label="Descargar CSV",
            data=buffer,
            file_name="datos_clusterizados.csv",
            mime="text/csv"
        )

else:
    st.info("Carga un archivo CSV para comenzar.")
