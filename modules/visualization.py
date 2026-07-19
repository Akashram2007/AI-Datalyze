import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import time
from io import BytesIO


def visualization(data):

    type = st.selectbox(
        "file Mode", ["Original", "Drop Null Values", "Fill Null Values"], width=300
    )

    if type == "Original":
        data = data

    if type == "Drop Null Values":
        data.dropna(axis=0, inplace=True)

    if type == "Fill Null Values":
        data = data.apply(
            lambda x: (
                x.fillna(x.mean())
                if x.dtype == "float"
                else x.fillna(x.value_counts().index[0])
            )
        )

    st.dataframe(data, width=550, height=250)

    plot = st.selectbox(
        "Select Plot Type",
        ["none", "Scatter", "Line", "Bar", "Histogram", "Heat map"],
        width=300,
    )

    columns = data.columns.to_list()
    numeric_columns = data.select_dtypes(include=["number"]).columns.to_list()
    MAX_ROWS = 5000

    if len(data) > MAX_ROWS:
        st.info(f"Large dataset detected ({len(data)} rows). Using a random sample of {MAX_ROWS} rows for faster visualization.")
        plot_data = data.sample(MAX_ROWS, random_state=42)
    else:
        plot_data = data

    if plot != "none":
        fig, a = plt.subplots()

        # ---------------- HEATMAP ----------------
        if plot == "Heat map":
            st.header("Heat Map")
            corr = plot_data.corr(numeric_only=True)
            sns.heatmap(corr, annot=True, cmap="coolwarm")

        # -------- COMMON INPUTS --------
        if plot != "Heat map":
            x = st.selectbox(
                "X axis", numeric_columns if plot != "Bar" else columns, width=300
            )
            plt.xlabel(x)

        if plot not in ["Histogram", "Heat map"]:
            y = st.selectbox(
                "Y axis",
                numeric_columns if plot != "Bar" else columns,
                key="y",
                width=300,
            )
            plt.ylabel(y)

        if plot != "Heat map":
            color = st.color_picker("Graph colour", value="#1DB0D6", width=300)

        generate = st.button("Generate Visualization", type="primary")

        if generate:
            # ---------------- HISTOGRAM ----------------
            if plot == "Histogram":
                st.header("Histogram Plot")
                bins = int(
                    st.number_input("Enter no of Bins", max_value=20, value=6, width=300)
                )
                sns.histplot(plot_data[x], bins=bins, color=color)
    
            # ---------------- SCATTER ----------------
            if plot == "Scatter":
                sns.scatterplot(data=plot_data, x=x, y=y, color=color)
    
            # ---------------- LINE ----------------
            if plot == "Line":
                sns.lineplot(data=plot_data, x=x, y=y, color=color)
    
            # ---------------- BAR ----------------
            if plot == "Bar":
                sns.barplot(data=plot_data, x=x, y=y, color=color)
    
            with st.spinner("Visualizing Your Data..."):
                time.sleep(0.5)
                plt.style.use("default")
                plt.xticks(rotation=45, ha="right")
                st.pyplot(fig, width="content")
                buf = BytesIO()
                fig.savefig(buf, format="png")
                st.download_button(
                    "Download Chart",
                    data=buf.getvalue(),
                    mime="image/png",
                    file_name=f"{plot} chart.png",
                )
