# ==========================================
# ML-BASED COURSE RECOMMENDATION DASHBOARD
# ==========================================

import streamlit as st
import pandas as pd
import random
import base64

from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier

# ------------------------------------------
# STREAMLIT CONFIG
# ------------------------------------------
st.set_page_config(
    page_title="Student Course Recommendation System",
    layout="centered"
)

# ------------------------------------------
# BACKGROUND IMAGE (LOCAL FILE)
# ------------------------------------------
def add_bg_from_local(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# 🔹 YOUR IMAGE PATH
add_bg_from_local(
    r"D:\kiru\Screenshot 2026-02-20 220337.png"
)

# ------------------------------------------
# SIMPLE MULTI-LINE MOTIVATIONAL QUOTES
# ------------------------------------------
MOTIVATIONAL_QUOTES = [
    """Start where you are.
Use what you have.
Do what you can.""",

    """Learning takes time.
Be patient with yourself.
Progress will come.""",

    """Every skill you learn
takes you one step
closer to your goal.""",

    """Mistakes are part of learning.
Do not stop.
Keep moving forward.""",

    """Believe in the process.
Small efforts every day
create big success."""
]

# ------------------------------------------
# LOAD DATA & TRAIN MODEL
# ------------------------------------------
@st.cache_data
def load_and_train_model():
    df = pd.read_csv("Final dataset.csv")

    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    coding_level_map = {
        "beginner": "no_coding",
        "intermediate": "low_coding",
        "advanced": "high_coding"
    }
    df["coding_level"] = df["coding_level"].str.lower().map(coding_level_map)

    explanation_df = df[
        ["recommended_course", "why_should_prefer", "career_roles_after_course"]
    ].drop_duplicates()

    df["skills"] = df["skills"].fillna("").apply(
        lambda x: [s.strip().lower() for s in x.split(",") if s.strip()]
    )

    mlb = MultiLabelBinarizer()
    skills_encoded = pd.DataFrame(
        mlb.fit_transform(df["skills"]),
        columns=mlb.classes_
    )

    df = pd.concat([df.drop(columns=["skills"]), skills_encoded], axis=1)

    encoders = {}
    for col in ["degree", "major", "coding_level"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    X = df.drop(columns=[
        "recommended_course",
        "why_should_prefer",
        "career_roles_after_course"
    ]).select_dtypes(include=["int64", "float64"])

    y = df["recommended_course"]

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)

    return model, encoders, mlb, explanation_df, X.columns


model, encoders, mlb, explanation_df, feature_columns = load_and_train_model()

# ------------------------------------------
# DEGREE → MAJOR MAPPING
# ------------------------------------------
DEGREE_MAJOR_MAP = {
    "B.Sc": ["Statistics", "Electronics", "Computer Science", "Mathematics"],
    "B.Tech": ["Computer Engineering", "Computer Science", "Electronics", "IT"],
    "BBA": ["Business Administration", "Marketing"],
    "Diploma": ["Computer Engineering", "Electronics", "IT"],
    "BCA": ["Computer Applications", "Computer Science", "IT"],
    "B.Com": ["Commerce"]
}

# ------------------------------------------
# STRICT CODING LEVEL → COURSES
# ------------------------------------------
coding_based_courses = {
    "no_coding": [
        "Networking",
        "Cloud Computing",
        "Data Analysis (Power BI)"
    ],
    "low_coding": [
        "Cyber Security",
        "MySQL",
        "Web Designing"
    ],
    "high_coding": [
        "Data Science",
        "Machine Learning",
        "Deep Learning",
        "Full Stack",
        "Django",
        "Spring Boot"
    ]
}

# ------------------------------------------
# COURSE → SKILLS ACQUIRED
# ------------------------------------------
COURSE_SKILLS_MAP = {
    "Cyber Security": [
        "Network Security",
        "Ethical Hacking Basics",
        "Risk Assessment",
        "Security Monitoring"
    ],
    "MySQL": [
        "Database Design",
        "SQL Queries",
        "Data Management",
        "Indexing & Optimization"
    ],
    "Web Designing": [
        "HTML & CSS",
        "Responsive Design",
        "UI/UX Basics",
        "Web Accessibility"
    ],
    "Data Science": [
        "Python for Data Science",
        "Data Analysis",
        "Statistics",
        "Data Visualization"
    ],
    "Machine Learning": [
        "Supervised Learning",
        "Feature Engineering",
        "Model Evaluation",
        "ML Algorithms"
    ],
    "Networking": [
        "Network Fundamentals",
        "Routing & Switching",
        "Troubleshooting",
        "Protocols & Standards"
    ]
}

# ------------------------------------------
# STREAMLIT UI
# ------------------------------------------
st.title("🎓 Student Course Recommendation System")

student_name = st.text_input("👤 Student Name")
cgpa = st.number_input("📊 CGPA", 0.0, 10.0, step=0.1)

degree = st.selectbox("🎓 Degree", encoders["degree"].classes_)
major = st.selectbox(
    "📘 Major",
    DEGREE_MAJOR_MAP.get(degree, encoders["major"].classes_)
)

interest_areas = st.multiselect(
    "💡 Select Interest Area(s)",
    explanation_df["recommended_course"].unique()
)

skill_options = ["NONE"] + list(mlb.classes_)
skills_selected = st.multiselect("🛠 Select Skills", skill_options)

if "NONE" in skills_selected:
    coding_level = st.radio(
        "💻 Select Coding Level",
        ["no_coding", "low_coding", "high_coding"],
        horizontal=True
    )
    skills_selected = []
else:
    coding_level = "low_coding"

# ------------------------------------------
# RECOMMENDATION BUTTON
# ------------------------------------------
if st.button("🚀 Get Recommendations"):

    if not student_name:
        st.warning("Please enter student name.")
        st.stop()

    if not skills_selected:
        mapped_courses = coding_based_courses.get(coding_level, []).copy()
        random.shuffle(mapped_courses)
        top_courses = mapped_courses[:3]
    else:
        input_row = {
            "degree": encoders["degree"].transform([degree])[0],
            "major": encoders["major"].transform([major])[0],
            "coding_level": encoders["coding_level"].transform([coding_level])[0],
            "cgpa": cgpa
        }

        for skill in mlb.classes_:
            input_row[skill] = 1 if skill in skills_selected else 0

        input_df = pd.DataFrame([input_row])
        input_df = input_df.reindex(columns=feature_columns, fill_value=0)

        probs = model.predict_proba(input_df)[0]
        courses = model.classes_

        course_scores = list(zip(courses, probs))
        course_scores.sort(key=lambda x: x[1], reverse=True)
        top_courses = [c for c, _ in course_scores[:3]]

    st.success(f"🎯 Top Course Recommendations for {student_name}")

    for i, course in enumerate(top_courses, 1):
        st.markdown(f"### {i}. {course}")

        matched = explanation_df[
            explanation_df["recommended_course"] == course
        ]

        if not matched.empty:
            row = matched.iloc[0]
            st.markdown(f"**Why Should Prefer:** {row['why_should_prefer']}")
        else:
            st.markdown("**Why Should Prefer:** Recommended based on selected preferences.")

        skills_acquired = COURSE_SKILLS_MAP.get(
            course,
            ["Core fundamentals", "Practical knowledge", "Industry-relevant skills"]
        )
        st.markdown("**Skills Acquired:** " + ", ".join(skills_acquired))

        if not matched.empty:
            st.markdown(
                "**Career Roles After Course:** "
                + ", ".join(row["career_roles_after_course"].split(","))
            )
        else:
            st.markdown(
                "**Career Roles After Course:** Entry-level roles related to this domain."
            )

        st.markdown("---")

    st.subheader("🌟 Motivation for You")
    st.info(random.choice(MOTIVATIONAL_QUOTES))