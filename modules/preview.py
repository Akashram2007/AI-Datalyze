import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO


def data_preview(data, file):

    st.success("File Uploaded Successfully!")
    st.write("### File Name :", file.name)

    st.write("##### Dataset Size :", data.size)
    st.write("##### Number of Rows :", data.shape[0])

    # ---------- Session State ----------
    if "filtered_data" not in st.session_state:
        st.session_state.filtered_data = data.copy()

    if "reset_counter" not in st.session_state:
        st.session_state.reset_counter = 0

    st.write("## 🔍 Smart Data Filter")

    # ---------- Filter Form ----------
    with st.expander("⚙️ Click to Apply Filters", expanded=True):

        with st.form("filter_form"):

            user_inputs = {}
            columns = list(data.columns)
            max_per_row = 3

            for i in range(0, len(columns), max_per_row):

                row_cols = st.columns(max_per_row)

                for j in range(max_per_row):

                    if i + j >= len(columns):
                        continue

                    col_name = columns[i + j]

                    with row_cols[j]:

                        # ---------- Numeric Columns ----------
                        if pd.api.types.is_numeric_dtype(data[col_name]):

                            numeric_col = (
                                data[col_name]
                                .replace([np.inf, -np.inf], np.nan)
                                .dropna()
                            )

                            if numeric_col.empty:
                                st.info("No valid values")
                                continue

                            min_val = float(numeric_col.min())
                            max_val = float(numeric_col.max())

                            if min_val == max_val:
                                st.write(f"**{col_name}**")
                                st.caption(f"Only value: {min_val}")
                                user_inputs[col_name] = (
                                    "numeric",
                                    (min_val, max_val),
                                )

                            else:

                                selected_range = st.slider(
                                    label=col_name,
                                    min_value=min_val,
                                    max_value=max_val,
                                    value=(min_val, max_val),
                                    key=f"slider_{col_name}_{st.session_state.reset_counter}",
                                )

                                user_inputs[col_name] = (
                                    "numeric",
                                    selected_range,
                                )

                        # ---------- Categorical Columns ----------
                        else:

                            unique_vals = (
                                data[col_name]
                                .dropna()
                                .astype(str)
                                .unique()
                                .tolist()
                            )

                            selected_vals = st.multiselect(
                                label=col_name,
                                options=unique_vals,
                                key=f"multi_{col_name}_{st.session_state.reset_counter}",
                            )

                            user_inputs[col_name] = (
                                "categorical",
                                selected_vals,
                            )

            # ---------- Buttons ----------
            col1, col2 = st.columns(2)

            with col1:
                apply_filter = st.form_submit_button(
                    "Apply Filter",
                    use_container_width=True,
                )

            with col2:
                reset_filter = st.form_submit_button(
                    "🔄 Reset Filter",
                    use_container_width=True,
                )

    # ---------- Apply Filter ----------
    if apply_filter:

        filtered_data = data.copy()

        for col, value in user_inputs.items():

            if value[0] == "numeric":

                low, high = value[1]

                filtered_data = filtered_data[
                    filtered_data[col].between(low, high)
                ]

            else:

                selected = value[1]

                if selected:
                    filtered_data = filtered_data[
                        filtered_data[col].astype(str).isin(selected)
                    ]

        st.session_state.filtered_data = filtered_data

    # ---------- Reset ----------
    if reset_filter:

        st.session_state.filtered_data = data.copy()
        st.session_state.reset_counter += 1
        st.rerun()

    # ---------- Preview ----------
    st.write("### 📊 Data Preview")

    display_data = st.session_state.filtered_data

    if display_data.empty:

        st.warning("No data matches your filter.")

    else:

        st.dataframe(display_data, use_container_width=True)

        st.divider()

        st.subheader("⬇️ Download Filtered Data")

        file_format = st.selectbox(
            "Select Format",
            ["CSV", "Excel"],
        )

        if file_format == "CSV":

            csv = display_data.to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download CSV",
                csv,
                f"filtered_{file.name}",
                "text/csv",
            )

        else:

            output = BytesIO()

            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                display_data.to_excel(writer, index=False)

            st.download_button(
                "Download Excel",
                output.getvalue(),
                "filtered_data.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
