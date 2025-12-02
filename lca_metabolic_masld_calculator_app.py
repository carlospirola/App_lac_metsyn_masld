import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# ============================================================================
# GENERAL SETTINGS (EASY TO EDIT)
# ============================================================================

APP_TITLE = "LCA Calculator for Cardio-metabolic Syndrome (CMS) and Associated Liver Disease"

# Mapeo de la escala ordinal de esteatosis a valores continuos de luxcapm
# >>> Si más adelante decides otros puntos de corte, SOLO modifica este diccionario <<<
LIVER_STEATOSIS_TO_LUXCAPM = {
    "No steatosis": 200.0,
    "Mild steatosis": 300.0,
    "Moderate steatosis": 400.0,
    "Severe steatosis": 500.0,
}

# ============================================================================
# ETHNICITY LABELS (NHANES ridreth3)
# ============================================================================

ETHNICITY_LABELS = {
    1: "Mexican American",
    2: "Other Hispanic",
    3: "Non-Hispanic White",
    4: "Non-Hispanic Black",
    6: "Non-Hispanic Asian",
    7: "Other race, including multi-racial",
}

# ============================================================================
# CLASS NAMES (PHENOTYPIC DESCRIPTIONS) - EDIT HERE WHEN YOU WANT TO REFINE
# ============================================================================

MALE_CLASS_LABELS = {
    1: "Healthy metabolic profile",
    2: "Moderate CMS with alcohol consumption",
    3: "Moderate CMS with severe hepatic injury and advanced liver fibrosis",
    4: "Severe dyslipidemia, diabetes and MASLD with moderate liver fibrosis",
    5: "Severe CMS with extreme central obesity, diabetes, and MASLD with liver fibrosis",
    6: "Mild metabolic alterations with minimal MASLD and no significant fibrosis",
    7: "Older men with low CMS and minimal MASLD",
    8: "Severe CMS with advanced MASLD and moderate liver fibrosis",
}

FEMALE_CLASS_LABELS = {
    1: "Healthy metabolic profile",
    2: "Severe obesity, hypertension and diabetes with MASLD and moderate liver fibrosis",
    3: "Moderate CMS with moderate MASLD and liver fibrosis",
    4: "CMS with severe atherogenic dyslipidemia with moderate MASLD",
    5: "CMS with insulin resistance, moderate MASLD and liver fibrosis",
    6: "Alcohol-related severe hepatic injury",
    7: "Central obesity with mild MASLD and minimal metabolic alterations",
    8: "Posibly hypertensive but low CMS in older women",
}

# ============================================================================
# MALE MODEL - CLASS MEMBERSHIP (MULTINOMIAL LOGIT)
# ============================================================================

MALE_CLASS_MLOGIT = {
    2: {"intercept": -4.299433, "age": 0.0411761,
        "eth": {2: -148.2009, 3: 1.208657, 4: 0.9617142, 6: -1.485093, 7: 1.610402}},
    3: {"intercept": -1.913282, "age": 0.0041656,
        "eth": {2: -0.2410322, 3: -0.9491634, 4: -1.645697, 6: -1.063983, 7: -1.758185}},
    4: {"intercept": -3.908778, "age": 0.0836619,
        "eth": {2: -1.675316, 3: -3.038362, 4: -3.438301, 6: -1.142504, 7: -3.642515}},
    5: {"intercept": -5.336366, "age": 0.1288568,
        "eth": {2: -0.720041, 3: -0.3957205, 4: -0.6615726, 6: -1.257337, 7: -0.4199548}},
    6: {"intercept": -0.8478142, "age": 0.0598029,
        "eth": {2: -0.8941382, 3: -1.190853, 4: -2.11468, 6: -0.9922657, 7: -1.782308}},
    7: {"intercept": -8.765198, "age": 0.1851344,
        "eth": {2: -0.4737323, 3: -0.3641895, 4: 0.4976444, 6: 0.6005041, 7: -0.1534424}},
    8: {"intercept": -2.660303, "age": 0.0638979,
        "eth": {2: -0.5745991, 3: -1.192711, 4: -1.495576, 6: -0.5761211, 7: -1.479588}},
}

# ============================================================================
# MALE MODEL - CLASS-SPECIFIC MEANS
# ============================================================================

MALE_CLASS_MEANS = {
    1: {"diab": -5.64992, "htn": -3.065706, "luxcapm": 216.4776, "lbxsgtsi": 20.77598,
        "lbxsassi": 20.93869, "lbxsatsi": 18.92439, "lbxstr": 98.42288,
        "bmxwaist": 86.60281, "lbdhdd": 53.72585, "dr1talco": 9.795836, "homa": 1.731633},
    2: {"diab": -4.399405, "htn": -0.7088396, "luxcapm": 239.6919, "lbxsgtsi": 31.28246,
        "lbxsassi": 24.25696, "lbxsatsi": 24.63771, "lbxstr": 102.8117,
        "bmxwaist": 92.24117, "lbdhdd": 66.33334, "dr1talco": 116.6922, "homa": 1.559169},
    3: {"diab": -2.190797, "htn": -0.1357592, "luxcapm": 304.4974, "lbxsgtsi": 150.9954,
        "lbxsassi": 96.92253, "lbxsatsi": 116.6269, "lbxstr": 171.7328,
        "bmxwaist": 103.8648, "lbdhdd": 48.08751, "dr1talco": 36.4418, "homa": 5.935443},
    4: {"diab": -0.0355359, "htn": -0.1607442, "luxcapm": 335.6827, "lbxsgtsi": 55.37636,
        "lbxsassi": 26.62771, "lbxsatsi": 36.03836, "lbxstr": 648.8614,
        "bmxwaist": 109.1343, "lbdhdd": 31.46755, "dr1talco": 30.78538, "homa": 96.7251},
    5: {"diab": -0.3019631, "htn": 1.121087, "luxcapm": 348.4445, "lbxsgtsi": 35.01314,
        "lbxsassi": 21.05124, "lbxsatsi": 26.06329, "lbxstr": 213.0197,
        "bmxwaist": 122.0544, "lbdhdd": 40.638, "dr1talco": 6.913903, "homa": 9.438209},
    6: {"diab": -4.977112, "htn": -0.9821469, "luxcapm": 276.5172, "lbxsgtsi": 33.10047,
        "lbxsassi": 21.7516, "lbxsatsi": 27.01063, "lbxstr": 168.698,
        "bmxwaist": 105.4485, "lbdhdd": 43.8693, "dr1talco": 12.39717, "homa": 2.969843},
    7: {"diab": -1.264564, "htn": 0.6339054, "luxcapm": 255.0255, "lbxsgtsi": 28.05723,
        "lbxsassi": 20.17097, "lbxsatsi": 18.97975, "lbxstr": 136.9401,
        "bmxwaist": 100.6455, "lbdhdd": 52.75228, "dr1talco": 7.698299, "homa": 2.848719},
    8: {"diab": -3.030969, "htn": 0.3759108, "luxcapm": 247.0107, "lbxsgtsi": 19.77853,
        "lbxsassi": 19.63088, "lbxsatsi": 16.63745, "lbxstr": 117.2257,
        "bmxwaist": 97.04659, "lbdhdd": 64.89275, "dr1talco": 5.892049, "homa": 2.701347},
}

MALE_VARIANCES = {
    "luxcapm": 1991.863, "lbxsgtsi": 1532.767, "lbxsassi": 63.34438,
    "lbxsatsi": 106.4585, "lbxstr": 12906.95, "bmxwaist": 130.6641,
    "lbdhdd": 115.2197, "dr1talco": 955.0364, "homa": 17.13621
}

# ============================================================================
# FEMALE MODEL - CLASS MEMBERSHIP (MULTINOMIAL LOGIT)
# ============================================================================

FEMALE_CLASS_MLOGIT = {
    2: {"intercept": -5.120774, "age": 0.1059971,
        "eth": {2: -0.6544237, 3: -0.9803473, 4: -0.0409927, 6: -1.241433, 7: 0.0147411}},
    3: {"intercept": -3.651439, "age": 0.0436321,
        "eth": {2: -0.9866063, 3: -0.9456708, 4: -0.814841, 6: -0.6960233, 7: 0.2466913}},
    4: {"intercept": -5.221722, "age": 0.0801936,
        "eth": {2: -1.425649, 3: -1.267639, 4: -1.581999, 6: -1.042967, 7: 0.0791409}},
    5: {"intercept": -7.366318, "age": 0.0935766,
        "eth": {2: -0.124116, 3: -0.1945387, 4: 0.5401637, 6: 0.0334841, 7: -1.826939}},
    6: {"intercept": -4.785539, "age": 0.0406374,
        "eth": {2: -0.9214368, 3: -1.804985, 4: -0.10499, 6: -2.348284, 7: -1.551739}},
    7: {"intercept": -0.2288798, "age": 0.0144441,
        "eth": {2: -0.8975548, 3: -1.035039, 4: -0.8085863, 6: -1.9143, 7: -0.3779484}},
    8: {"intercept": -8.529894, "age": 0.1568843,
        "eth": {2: -0.4891519, 3: 0.0752218, 4: 1.309547, 6: 0.2260208, 7: 1.080182}},
}

# ============================================================================
# FEMALE MODEL - CLASS-SPECIFIC MEANS
# ============================================================================

FEMALE_CLASS_MEANS = {
    1: {"diab": -3.861725, "htn": -3.4006, "luxcapm": 205.6366, "lbxsgtsi": 15.71436,
        "lbxsassi": 18.29138, "lbxsatsi": 15.00289, "lbxstr": 88.06121,
        "bmxwaist": 84.22788, "lbdhdd": 63.33566, "dr1talco": 8.282136, "homa": 1.837805},
    2: {"diab": -0.1109077, "htn": 0.8460349, "luxcapm": 328.644, "lbxsgtsi": 30.19265,
        "lbxsassi": 20.235, "lbxsatsi": 22.22998, "lbxstr": 182.5049,
        "bmxwaist": 115.7617, "lbdhdd": 47.99476, "dr1talco": 3.658745, "homa": 8.725501},
    3: {"diab": -1.376639, "htn": -0.2400739, "luxcapm": 307.9308, "lbxsgtsi": 43.00033,
        "lbxsassi": 52.20923, "lbxsatsi": 62.33096, "lbxstr": 138.8735,
        "bmxwaist": 109.5097, "lbdhdd": 55.83559, "dr1talco": 22.93062, "homa": 6.398774},
    4: {"diab": -0.9900589, "htn": -0.1060857, "luxcapm": 299.1021, "lbxsgtsi": 29.87667,
        "lbxsassi": 18.58798, "lbxsatsi": 19.46529, "lbxstr": 420.8522,
        "bmxwaist": 107.8573, "lbdhdd": 40.35242, "dr1talco": 2.527686, "homa": 6.652077},
    5: {"diab": -1.213038, "htn": 0.9263342, "luxcapm": 283.3167, "lbxsgtsi": 153.4141,
        "lbxsassi": 32.42399, "lbxsatsi": 34.50711, "lbxstr": 159.9876,
        "bmxwaist": 102.7821, "lbdhdd": 68.13479, "dr1talco": 20.20128, "homa": 11.05144},
    6: {"diab": -3.030969, "htn": 0.3759108, "luxcapm": 247.0107, "lbxsgtsi": 364.3688,
        "lbxsassi": 125.6646, "lbxsatsi": 99.34705, "lbxstr": 121.6286,
        "bmxwaist": 89.68642, "lbdhdd": 85.6148, "dr1talco": 61.74064, "homa": 3.301032},
    7: {"diab": -5.117643, "htn": -1.742758, "luxcapm": 278.0977, "lbxsgtsi": 17.52077,
        "lbxsassi": 16.22579, "lbxsatsi": 15.43474, "lbxstr": 132.0946,
        "bmxwaist": 110.2363, "lbdhdd": 51.23931, "dr1talco": 5.931952, "homa": 3.96975},
    8: {"diab": -2.085394, "htn": 0.9184931, "luxcapm": 251.7922, "lbxsgtsi": 19.77853,
        "lbxsassi": 19.63088, "lbxsatsi": 16.63745, "lbxstr": 117.2257,
        "bmxwaist": 97.04659, "lbdhdd": 64.89275, "dr1talco": 5.892049, "homa": 2.701347},
}

FEMALE_VARIANCES = {
    "luxcapm": 1976.251, "lbxsgtsi": 213.8812, "lbxsassi": 56.74282,
    "lbxsatsi": 67.24456, "lbxstr": 2710.455, "bmxwaist": 175.3809,
    "lbdhdd": 190.9053, "dr1talco": 338.8994, "homa": 43.9833
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def prior_class_probs(age, ridreth3, sex):
    if sex == "Male":
        mlogit_coefs = MALE_CLASS_MLOGIT
    else:
        mlogit_coefs = FEMALE_CLASS_MLOGIT

    logits = {1: 0.0}
    for k in range(2, 9):
        coef = mlogit_coefs[k]
        logit_k = coef["intercept"] + coef["age"] * age
        if ridreth3 in coef["eth"]:
            logit_k += coef["eth"][ridreth3]
        logits[k] = logit_k

    exp_logits = {k: np.exp(np.clip(v, -500, 500)) for k, v in logits.items()}
    total = sum(exp_logits.values())
    probs = {k: exp_logits[k] / total for k in range(1, 9)}
    return probs

def likelihood_given_class(data, class_k, sex):
    if sex == "Male":
        means = MALE_CLASS_MEANS[class_k]
        variances = MALE_VARIANCES
    else:
        means = FEMALE_CLASS_MEANS[class_k]
        variances = FEMALE_VARIANCES

    log_lik = 0.0

    # Binary outcomes
    if data["diab"] is not None:
        p_diab = sigmoid(means["diab"])
        log_lik += np.log(p_diab if data["diab"] == 1 else 1 - p_diab)

    if data["htn"] is not None:
        p_htn = sigmoid(means["htn"])
        log_lik += np.log(p_htn if data["htn"] == 1 else 1 - p_htn)

    # Gaussian outcomes
    continuous_vars = ["luxcapm", "lbxsgtsi", "lbxsassi", "lbxsatsi",
                       "lbxstr", "bmxwaist", "lbdhdd", "dr1talco", "homa"]

    for var in continuous_vars:
        if data[var] is not None:
            mean = means[var]
            std = np.sqrt(variances[var])
            log_lik += norm.logpdf(data[var], loc=mean, scale=std)

    return log_lik

def posterior_class_probs(data, age, ridreth3, sex):
    prior = prior_class_probs(age, ridreth3, sex)
    log_liks = {k: likelihood_given_class(data, k, sex) for k in range(1, 9)}
    log_post = {k: np.log(prior[k]) + log_liks[k] for k in range(1, 9)}
    max_log = max(log_post.values())
    unnorm = {k: np.exp(log_post[k] - max_log) for k in range(1, 9)}
    tot = sum(unnorm.values())
    post = {k: unnorm[k] / tot for k in range(1, 9)}
    return post

# ============================================================================
# STREAMLIT APP
# ============================================================================

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")

    st.title(APP_TITLE)
    st.markdown("""
    This tool estimates the probability of belonging to each of **8 latent classes**
    defined by a latent class analysis (LCA) of components of the metabolic syndrome
    and associated liver disease.
    """)

    # Sidebar: demographic
    st.sidebar.header("Demographic data")

    sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
    age = st.sidebar.number_input("Age (years)", min_value=18, max_value=120, value=45, step=1)

    ethnicity_code = st.sidebar.selectbox(
        "Ethnicity",
        options=list(ETHNICITY_LABELS.keys()),
        format_func=lambda x: ETHNICITY_LABELS[x]
    )

    # Sidebar: clinical
    st.sidebar.header("Clinical data")

    diab = st.sidebar.selectbox("Diabetes", ["Not specified", "No", "Yes"])
    diab_val = None if diab == "Not specified" else (1 if diab == "Yes" else 0)

    htn = st.sidebar.selectbox("Hypertension", ["Not specified", "No", "Yes"])
    htn_val = None if htn == "Not specified" else (1 if htn == "Yes" else 0)

    # Sidebar: Liver steatosis (ordinal → luxcapm)
    st.sidebar.header("Liver disease")

    steatosis_level = st.sidebar.selectbox(
        "Liver steatosis (non-invasive assessment)",
        list(LIVER_STEATOSIS_TO_LUXCAPM.keys())
    )
    luxcapm_value = LIVER_STEATOSIS_TO_LUXCAPM[steatosis_level]

    # Sidebar: labs
    st.sidebar.header("Laboratory data")

    c1, c2 = st.sidebar.columns(2)

    with c1:
        ggt = st.number_input("GGT (U/L)", min_value=0.0, value=40.0, step=1.0)
        ast = st.number_input("AST (U/L)", min_value=0.0, value=25.0, step=1.0)
        alt = st.number_input("ALT (U/L)", min_value=0.0, value=25.0, step=1.0)
        tg = st.number_input("Triglycerides (mg/dL)", min_value=0.0, value=150.0, step=1.0)
        hdl = st.number_input("HDL-C (mg/dL)", min_value=0.0, value=45.0, step=1.0)

    with c2:
        waist = st.number_input("Waist circumference (cm)", min_value=0.0, value=90.0, step=0.5)
        homa = st.number_input("HOMA-IR", min_value=0.0, value=2.0, step=0.1)
        alcohol = st.number_input("Alcohol intake (g/day)", min_value=0.0, value=0.0, step=1.0)

    # Data dict for likelihoods
    data = {
        "diab": diab_val,
        "htn": htn_val,
        "luxcapm": luxcapm_value,
        "lbxsgtsi": ggt,
        "lbxsassi": ast,
        "lbxsatsi": alt,
        "lbxstr": tg,
        "bmxwaist": waist,
        "lbdhdd": hdl,
        "dr1talco": alcohol,
        "homa": homa,
    }

    if st.sidebar.button("Calculate class probabilities"):
        post = posterior_class_probs(data, age, ethnicity_code, sex)

        if sex == "Male":
            labels = MALE_CLASS_LABELS
        else:
            labels = FEMALE_CLASS_LABELS

        st.header("Results")

        col_left, col_right = st.columns([1, 2])

        with col_left:
            st.subheader("Class probabilities")

            df = pd.DataFrame({
                "Class": [f"{k} – {labels[k]}" for k in range(1, 9)],
                "Probability": [post[k] for k in range(1, 9)]
            })
            df["Probability (%)"] = (df["Probability"] * 100).round(2)

            st.dataframe(df[["Class", "Probability (%)"]], use_container_width=True)

            max_class = max(post, key=post.get)
            st.success(
                f"Most likely class: {max_class} – {labels[max_class]} "
                f"({post[max_class]*100:.2f}%)"
            )

        with col_right:
            st.subheader("Probability distribution across classes")

            classes = list(range(1, 9))
            probs = [post[k] * 100 for k in classes]

            fig, ax = plt.subplots(figsize=(9, 5))
            bars = ax.bar(classes, probs, color="steelblue", alpha=0.8, edgecolor="black")

            bars[max_class - 1].set_color("crimson")

            ax.set_xlabel("Class", fontsize=12, fontweight="bold")
            ax.set_ylabel("Probability (%)", fontsize=12, fontweight="bold")
            ax.set_title(f"Class membership probabilities ({sex}, age {age})",
                         fontsize=13, fontweight="bold")
            ax.set_xticks(classes)
            ax.set_ylim(0, 100)
            ax.grid(axis="y", alpha=0.3)

            st.pyplot(fig)

        st.info(
            f"Input summary:\n"
            f"- Sex: {sex}\n"
            f"- Age: {age} years\n"
            f"- Ethnicity: {ETHNICITY_LABELS[ethnicity_code]}\n"
            f"- Liver steatosis: {steatosis_level} (mapped to luxcapm={luxcapm_value})\n"
            f"- Diabetes: {diab}\n"
            f"- Hypertension: {htn}\n"
        )


if __name__ == "__main__":
    main()
