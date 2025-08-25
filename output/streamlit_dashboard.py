import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Lettuce RL Dashboard", layout="wide")

st.title("🥬 Lettuce Environment Dashboard")

# ⬆️ رفع ملف CSV
uploaded_file = st.file_uploader("Upload your evaluation CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, sep="\t|,|;", engine="python")  # يقرا بأي فاصل
    st.success(f"✅ Loaded {df.shape[0]} timesteps, {df.shape[1]} features")

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Environment States",
        "Control Actions",
        "Rewards & Costs",
        "Violations",
        "Profit Analysis",
        "Raw Data"
    ])

    # ========= Environment States =========
    with tab1:
        st.subheader("🌡️ Environment States")
        cols = ["temp_air", "co2_air", "rh_air", "pipe_temp", "cFruit", "tSum"]
        for col in cols:
            if col in df.columns:
                fig, ax = plt.subplots()
                ax.plot(df.index, df[col], label=col)
                ax.set_xlabel("Timestep")
                ax.set_ylabel(col)
                ax.set_title(f"{col} over time")
                ax.legend()
                st.pyplot(fig)

    # ========= Control Actions =========
    with tab2:
        st.subheader("🎛️ Control Actions")
        controls = ["uBoil", "uCo2", "uThScr", "uVent", "uLamp", "uBlScr"]
        for col in controls:
            if col in df.columns:
                fig, ax = plt.subplots()
                ax.plot(df.index, df[col], label=col, color="orange")
                ax.set_xlabel("Timestep")
                ax.set_ylabel("Action level")
                ax.set_title(f"{col} over time")
                ax.legend()
                st.pyplot(fig)

    # ========= Rewards & Costs =========
    with tab3:
        st.subheader("💰 Rewards and Costs (per timestep)")
        metrics = ["Rewards", "EPI", "Revenue", "Heat costs", "CO2 costs", "Elec costs"]
        for col in metrics:
            if col in df.columns:
                fig, ax = plt.subplots()
                ax.plot(df.index, df[col], label=col, color="green")
                ax.set_xlabel("Timestep")
                ax.set_ylabel(col)
                ax.set_title(f"{col} over time")
                ax.legend()
                st.pyplot(fig)

        st.subheader("📈 Cumulative Rewards & Costs")
        cum_metrics = ["Rewards", "Revenue", "Heat costs", "CO2 costs", "Elec costs"]
        for col in cum_metrics:
            if col in df.columns:
                fig, ax = plt.subplots()
                ax.plot(df.index, df[col].cumsum(), label=f"Cumulative {col}", color="blue")
                ax.set_xlabel("Timestep")
                ax.set_ylabel(f"Cumulative {col}")
                ax.set_title(f"Cumulative {col} over time")
                ax.legend()
                st.pyplot(fig)

    # ========= Violations =========
    with tab4:
        st.subheader("⚠️ Constraint Violations")
        viols = ["temp_violation", "co2_violation", "rh_violation"]
        for col in viols:
            if col in df.columns:
                fig, ax = plt.subplots()
                ax.plot(df.index, df[col], label=col, color="red")
                ax.set_xlabel("Timestep")
                ax.set_ylabel(col)
                ax.set_title(f"{col} over time")
                ax.legend()
                st.pyplot(fig)

    # ========= Profit Analysis =========
        # ========= Profit Analysis =========
    with tab5:
        st.subheader("📊 Profit Analysis (Revenue vs Costs)")
        if all(c in df.columns for c in ["Revenue", "Heat costs", "CO2 costs", "Elec costs"]):
            df["Total Costs"] = df["Heat costs"] + df["CO2 costs"] + df["Elec costs"]
            df["Net Profit"] = df["Revenue"] - df["Total Costs"]

            # ---- Line Chart ----
            fig, ax = plt.subplots()
            ax.plot(df.index, df["Revenue"].cumsum(), label="Cumulative Revenue", color="green")
            ax.plot(df.index, df["Total Costs"].cumsum(), label="Cumulative Costs", color="red")
            ax.plot(df.index, df["Net Profit"].cumsum(), label="Cumulative Net Profit", color="blue")
            ax.set_xlabel("Timestep")
            ax.set_ylabel("€ (cumulative)")
            ax.set_title("Cumulative Revenue, Costs and Net Profit")
            ax.legend()
            st.pyplot(fig)

            # ---- Metrics ----
            st.subheader("📌 Net Profit Summary")
            col1, col2, col3 = st.columns(3)
            col1.metric("💰 Total Revenue", f"{df['Revenue'].sum():.2f}")
            col2.metric("💸 Total Costs", f"{df['Total Costs'].sum():.2f}")
            col3.metric("📊 Net Profit", f"{df['Net Profit'].sum():.2f}")

            # ---- Pie Chart (Costs breakdown) ----
            st.subheader("🥧 Cost Breakdown")
            costs = {
                "Heat": df["Heat costs"].sum(),
                "CO₂": df["CO2 costs"].sum(),
                "Electricity": df["Elec costs"].sum()
            }
            fig, ax = plt.subplots()
            ax.pie(costs.values(), labels=costs.keys(), autopct="%1.1f%%", startangle=90)
            ax.set_title("Cost Distribution")
            st.pyplot(fig)

            # ---- Bar Chart Revenue vs Costs ----
            st.subheader("📊 Revenue vs Costs (Total)")
            fig, ax = plt.subplots()
            ax.bar(["Revenue", "Costs", "Net Profit"], 
                   [df["Revenue"].sum(), df["Total Costs"].sum(), df["Net Profit"].sum()],
                   color=["green", "red", "blue"])
            ax.set_ylabel("€ (total)")
            ax.set_title("Comparison of Revenue, Costs and Net Profit")
            st.pyplot(fig)

            # ---- Histogram of Net Profit ----
            st.subheader("📉 Net Profit Distribution (per timestep)")
            fig, ax = plt.subplots()
            ax.hist(df["Net Profit"], bins=30, color="blue", alpha=0.7)
            ax.set_xlabel("Net Profit per timestep (€)")
            ax.set_ylabel("Frequency")
            ax.set_title("Distribution of Net Profit per timestep")
            st.pyplot(fig)

    # ========= Raw Data =========
    with tab6:
        st.subheader("📑 Raw Data Preview")
        st.dataframe(df.head(100))
    # ========= Profit Analysis =========
    
else:
    st.info("⬆️ Please upload your evaluation CSV file.")
