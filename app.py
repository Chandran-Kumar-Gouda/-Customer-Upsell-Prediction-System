import os
import json
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.stats import skew, kurtosis
import gradio as gr
import plotly.express as px

# File Loader
def load_file(file):
    ext = os.path.splitext(file.name)[-1].lower()

    if ext in [".csv"]:
        return pd.read_csv(file.name)
    elif ext in [".xlsx", ".xls"]:
        return pd.read_excel(file.name)
    elif ext in [".json"]:
        try:
            return pd.read_json(file.name)
        except ValueError:
            with open(file.name, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return pd.DataFrame.from_dict(data)
            elif isinstance(data, list):
                return pd.DataFrame(data)
            else:
                raise ValueError("Unsupported JSON structure.")
    elif ext in [".parquet"]:
        return pd.read_parquet(file.name)
    elif ext in [".txt"]:
        try:
            return pd.read_csv(file.name, sep=None, engine="python")
        except Exception:
            return pd.read_csv(file.name, delimiter="\t")
    else:
        raise ValueError(f"Unsupported file type: {ext}")

# ML Processing Function
def process_file(file):
    df = load_file(file)

    phone_numbers = df["Phone Number"] if "Phone Number" in df.columns else None
    df = df.drop(columns=["Phone Number"], errors="ignore")

    # Encode categorical features
    for col in df.select_dtypes(include=['object']).columns:
        if col != "Churn":
            df[col] = LabelEncoder().fit_transform(df[col])

    if "Churn" in df.columns:
        X = df.drop("Churn", axis=1)
        y = df["Churn"]
    else:
        raise ValueError("File must contain 'Churn' column.")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "eta": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 42
    }

    evals = [(dtrain, "train"), (dtest, "eval")]
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        evals=evals,
        early_stopping_rounds=50,
        verbose_eval=False
    )

    # Predict probabilities
    y_proba = model.predict(xgb.DMatrix(X))

    def classify_customer(prob):
        if prob >= 0.7:
            return "Retainable"
        elif prob >= 0.4:
            return "Monitoring"
        else:
            return "Upsellable"

    df["Churn_Prob"] = y_proba
    df["Category"] = df["Churn_Prob"].apply(classify_customer)

    output_df = X.copy()
    if phone_numbers is not None:
        output_df.insert(0, "Phone Number", phone_numbers)
    output_df["Category"] = df["Category"]

    return output_df


# Offer Assignment Functions
def filter_loyalty(df):
    if "Account Length" in df.columns:
        return df[df["Category"] == "Upsellable"].sort_values(
            by="Account Length", ascending=False
        ).head(20)
    return pd.DataFrame({"Error": ["Account Length column missing"]})

def filter_daytime(df):
    if "Day Mins" in df.columns:
        return df[df["Category"] == "Upsellable"].sort_values(
            by="Day Mins", ascending=False
        ).head(20)
    return pd.DataFrame({"Error": ["Day Mins column missing"]})

def filter_nighttime(df):
    if "Night Mins" in df.columns:
        return df[df["Category"] == "Upsellable"].sort_values(
            by="Night Mins", ascending=False
        ).head(20)
    return pd.DataFrame({"Error": ["Night Mins column missing"]})

def filter_international(df):
    if "Intl Mins" in df.columns:
        return df[df["Category"] == "Upsellable"].sort_values(
            by="Intl Mins", ascending=False
        ).head(20)
    return pd.DataFrame({"Error": ["Intl Mins column missing"]})


# Update Tables + Pie Chart
def update_tables(file):
    try:
        df = process_file(file)
        hidden_df.value = df
        upsell_df = df[df["Category"] == "Upsellable"]
        monitor_df = df[df["Category"] == "Monitoring"]
        retain_df = df[df["Category"] == "Retainable"]

        category_counts = df["Category"].value_counts().reset_index()
        category_counts.columns = ["Category", "Count"]
        fig = px.pie(
            category_counts,
            names="Category",
            values="Count",
            color="Category",
            title="Customer Distribution",
            hole=0.3
        )

        return upsell_df, monitor_df, retain_df, df, fig

    except Exception as e:
        return (
            pd.DataFrame({"Error": [str(e)]}),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            px.scatter(title="Error")
        )


# Gradio UI
custom_theme = gr.themes.Soft(
    primary_hue="rose",
    secondary_hue="slate",
    font=["ui-sans-serif", "sans-serif"]
).set(
    body_background_fill="#121212",
    body_text_color="#FFFFFF",
    body_text_color_subdued="#BBBBBB",
    button_primary_background_fill="#4CAF50",
    button_primary_background_fill_hover="#388E3C",
    button_primary_text_color="#FFFFFF",
    block_background_fill="#1A1A1A",
    block_border_color="#333333",
    block_shadow="0px 4px 12px rgba(0,0,0,0.4)"
)


with gr.Blocks(theme=custom_theme) as demo:

    gr.Markdown("## Customer Upsell & Retention Classifier")

    file_input = gr.File(label=" Upload Customer Data", file_types=None)
    hidden_df = gr.State()

    with gr.Tabs():
        with gr.TabItem("Overview"):
            gr.Markdown("### Customer Category Distribution")
            pie_chart = gr.Plot()

        with gr.TabItem("Upsellable Customers"):
            upsell_table = gr.Dataframe()
            gr.Markdown("Recommended Offers for Upsellable Customers:")
            with gr.Row():
                btn_offer1 = gr.Button("Loyalty Premium Plan")
                btn_offer2 = gr.Button("Day Time Plan")
                btn_offer3 = gr.Button("Night Time Plan")
                btn_offer4 = gr.Button("International Plan")
            offer_table = gr.Dataframe()

            btn_offer1.click(filter_loyalty, inputs=hidden_df, outputs=offer_table)
            btn_offer2.click(filter_daytime, inputs=hidden_df, outputs=offer_table)
            btn_offer3.click(filter_nighttime, inputs=hidden_df, outputs=offer_table)
            btn_offer4.click(filter_international, inputs=hidden_df, outputs=offer_table)

        with gr.TabItem("Monitoring Customers"):
            monitor_table = gr.Dataframe()
            gr.Markdown("Recommended Offers for Monitoring Customers:")
            with gr.Row():
                btn_offer5 = gr.Button("Day Time Plan")
                btn_offer6 = gr.Button("Night Time Plan")
                btn_offer7 = gr.Button("International Plan")
            offer_table_mon = gr.Dataframe()

            btn_offer5.click(filter_daytime, inputs=hidden_df, outputs=offer_table_mon)
            btn_offer6.click(filter_nighttime, inputs=hidden_df, outputs=offer_table_mon)
            btn_offer7.click(filter_international, inputs=hidden_df, outputs=offer_table_mon)

        with gr.TabItem("Retainable Customers"):
            retain_table = gr.Dataframe()
            gr.Markdown("Recommended Offers for Retainable Customers:")
            with gr.Row():
                btn_offer8 = gr.Button("'1' Month Free Plan")
            # left as-is (no filtering logic change)

    file_input.change(
        update_tables,
        inputs=file_input,
        outputs=[upsell_table, monitor_table, retain_table, hidden_df, pie_chart]
    )

demo.launch(share=True)
