
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import plotly.express as px

st.set_page_config(page_title="BioML Studio", layout="wide")
st.title("🧬 BioML Studio — Streamlit")
st.caption("Interactive pipelines for Molecular Biology datasets (EDA, PCA/t‑SNE, Clustering, Classification)")

with st.sidebar:
    st.header("⚙️ Settings")
    demo = st.toggle("Demo dataset (gene expression)", value=True)
    st.write("Upload a data matrix (samples × features) as CSV/TSV. Optionally upload labels (sample,class).")
    data_file = st.file_uploader("Data matrix", type=["csv", "tsv"])
    label_file = st.file_uploader("Labels (optional)", type=["csv", "tsv"])
    sep_choice = st.radio("Separator", ["CSV", "TSV"], index=0)
    sep = "," if sep_choice == "CSV" else "\t"

@st.cache_data
def load_demo():
    rng = np.random.RandomState(42)
    n, p, k = 150, 500, 3
    base = rng.normal(0, 1, size=(n, p))
    y = np.repeat(np.arange(k), n//k).astype(int)
    for i, cls in enumerate(np.unique(y)):
        idx = np.where(y == cls)[0]
        base[idx, i*50:(i+1)*50] += 2.5
    df = pd.DataFrame(base, index=[f"S{i:03d}" for i in range(n)], columns=[f"G{j:04d}" for j in range(p)])
    labels = pd.DataFrame({"sample": df.index, "class": [f"C{c}" for c in y]})
    return df, labels

@st.cache_data
def read_table(upload, sep):
    df = pd.read_csv(upload, sep=sep)
    return df

# Load data
if demo and data_file is None:
    X, labels = load_demo()
else:
    if data_file is None:
        st.warning("Upload a data matrix or enable the demo dataset from the sidebar.")
        st.stop()
    raw = read_table(data_file, sep)
    if raw.columns[0].lower() in {"sample", "id", "sample_id"} or raw[raw.columns[0]].dtype == object:
        raw = raw.set_index(raw.columns[0])
    X = raw.select_dtypes(include=[np.number])
    labels = None
    if label_file is not None:
        L = read_table(label_file, sep)
        L.columns = [c.lower() for c in L.columns]
        if "sample" in L.columns and "class" in L.columns:
            L = L[["sample", "class"]]
            L = L[L["sample"].isin(X.index)]
            labels = L

if labels is not None:
    labels = labels.set_index("sample").loc[X.index]
    y = labels["class"]
else:
    y = None

st.subheader("📁 Dataset")
st.write(f"Samples: **{X.shape[0]}**, Features: **{X.shape[1]}**")
st.dataframe(X.head(10))

X_scaled = X

tab_eda, tab_dim, tab_cluster, tab_clf, tab_team = st.tabs(
    ["🔎 EDA", "🧭 PCA / t-SNE", "🧩 Clustering", "🧪 Classification", "👥 About Team"]
)

with tab_eda:
    st.markdown("#### Feature variance & basic stats")
    var = X_scaled.var().sort_values(ascending=False)
    fig = px.histogram(var, nbins=50, labels={"value":"Variance"}, title="Feature Variance Distribution")
    st.plotly_chart(fig, use_container_width=True)
    st.write("Top 20 high-variance genes/features:")
    st.dataframe(var.head(20).to_frame("variance"))

with tab_dim:
    st.markdown("#### Dimensionality Reduction")
    n_comp = st.slider("PCA components", 2, min(10, X_scaled.shape[1]), 2)
    pca = PCA(n_components=n_comp, random_state=0)
    pcs = pca.fit_transform(X_scaled)
    exp = (pca.explained_variance_ratio_ * 100).round(2)
    st.write("Explained variance (%):", exp[:5], "…")
    df_pc = pd.DataFrame({"PC1": pcs[:,0], "PC2": pcs[:,1], "sample": X_scaled.index})
    if y is not None:
        df_pc["class"] = y.values
        color = "class"
    else:
        color = None
    fig = px.scatter(df_pc, x="PC1", y="PC2", color=color, hover_name="sample", title="PCA (PC1 vs PC2)")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("##### t-SNE (2D)")

    n_samples = X_scaled.shape[0]

    if n_samples <= 6:
        st.info(f"t-SNE: Χρειάζονται >6 δείγματα για να οριστεί έγκυρο perplexity "
                f"(έχεις {n_samples}). Δοκίμασε με μεγαλύτερο dataset.")
    else:
        upper = min(50, n_samples - 1)  # perplexity < n_samples
        default = min(30, upper - 1)  # σιγουρεύουμε default < upper
        perp = st.slider("Perplexity", 5, upper, default, step=1)

        try:
            tsne = TSNE(n_components=2, perplexity=perp, init="random",
                        learning_rate="auto", random_state=0)
            Z = tsne.fit_transform(X_scaled)
            df_tsne = pd.DataFrame(
                {"TSNE1": Z[:, 0], "TSNE2": Z[:, 1], "sample": X_scaled.index}
            )
            color = "class" if y is not None else None
            if y is not None:
                df_tsne["class"] = y.values
            fig = px.scatter(df_tsne, x="TSNE1", y="TSNE2", color=color,
                             hover_name="sample", title=f"t-SNE (2D) — perplexity={perp}")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"t-SNE απέτυχε: {e}")

with tab_cluster:
    st.markdown("#### Unsupervised Clustering")

    n_samples = X_scaled.shape[0]
    if n_samples < 2:
        st.info("Χρειάζονται ≥2 δείγματα για clustering.")
    else:
        algo = st.selectbox("Algorithm", ["KMeans", "Agglomerative"])

        # k δεν πρέπει να ξεπερνά τα δείγματα
        max_k = max(2, min(10, n_samples))
        k = st.slider(
            "Number of clusters (k)", 2, max_k, min(3, max_k),
            key=f"k_{n_samples}"   # reset του slider όταν αλλάζει το dataset
        )

        # Μικρή δικλίδα ασφαλείας
        if k > n_samples:
            st.warning(f"Το k ({k}) πρέπει να είναι ≤ στον αριθμό δειγμάτων ({n_samples}). Μείωσέ το.")
            st.stop()

        # Μοντέλο
        if algo == "KMeans":
            model = KMeans(n_clusters=k, n_init=10, random_state=0)
        else:
            model = AgglomerativeClustering(n_clusters=k)

        clusters = model.fit_predict(X_scaled)
        st.write("Cluster counts:", pd.Series(clusters).value_counts().sort_index())

        # Προβολή σε 2D με PCA
        pca = PCA(n_components=2, random_state=0).fit(X_scaled)
        coords = pca.transform(X_scaled)
        dfc = pd.DataFrame(
            {"PC1": coords[:, 0], "PC2": coords[:, 1],
             "cluster": clusters.astype(str), "sample": X_scaled.index}
        )
        fig = px.scatter(
            dfc, x="PC1", y="PC2", color="cluster",
            hover_name="sample", title="Clusters projected on PCA (2D)"
        )
        st.plotly_chart(fig, use_container_width=True)


with tab_clf:
    st.markdown("#### Supervised Classification")
    if y is None:
        st.info("Provide labels to enable supervised learning (labels file with 'sample','class'). Using demo labels if demo mode.")
        if demo:
            y = load_demo()[1].set_index("sample").loc[X.index]["class"]
        else:
            st.stop()
    algo = st.selectbox("Classifier", ["RandomForest", "LogisticRegression", "SVM"])
    test_size = st.slider("Test size", 0.1, 0.5, 0.2)
    random_state = st.number_input("Random state", value=0, step=1)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=random_state, stratify=y)
    if algo == "RandomForest":
        n_estimators = st.slider("n_estimators", 50, 500, 200, step=50)
        clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    elif algo == "LogisticRegression":
        C = st.slider("C (inverse regularization)", 0.01, 10.0, 1.0)
        clf = LogisticRegression(max_iter=2000, C=C)
    else:
        C = st.slider("C (regularization)", 0.01, 10.0, 1.0)
        clf = SVC(C=C, probability=True)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.metric("Accuracy", f"{acc:.3f}")
    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
    cm_df = pd.DataFrame(cm, index=np.unique(y), columns=np.unique(y))
    st.write("Confusion Matrix:")
    st.dataframe(cm_df)
    from sklearn.metrics import classification_report

    st.write("Classification Report:")
    report = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    st.dataframe(df_report.style.format("{:.2f}"))

with tab_team:
    st.markdown("### 👥 Ομάδα Έργου")
    st.write("**Τιμόθεος Πρασιάδης (Π2020072)** – Ανάπτυξη εφαρμογής")
