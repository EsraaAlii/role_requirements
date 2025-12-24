# simple_app.py
import streamlit as st
import requests
import pandas as pd

# Configuration
API_URL = "http://localhost:5000"

# Skills list (truncated for simplicity)
ALL_SKILLS = [
'APL',
 'Assembly',
 'Bash/Shell',
 'C',
 'C#',
 'C++',
 'COBOL',
 'Clojure',
 'Crystal',
 'Dart',
 'Delphi',
 'Elixir',
 'Erlang',
 'F#',
 'Go',
 'Groovy',
 'HTML/CSS',
 'Haskell',
 'Java',
 'JavaScript',
 'Julia',
 'Kotlin',
 'LISP',
 'Matlab',
 'Node.js',
 'Objective-C',
 'PHP',
 'Perl',
 'PowerShell',
 'Python',
 'R',
 'Ruby',
 'Rust',
 'SQL',
 'Scala',
 'Swift',
 'TypeScript',
 'VBA',
 'Cassandra',
 'Couchbase',
 'DynamoDB',
 'Elasticsearch',
 'Firebase',
 'IBM DB2',
 'MariaDB',
 'Microsoft SQL Server',
 'MongoDB',
 'MySQL',
 'Oracle',
 'PostgreSQL',
 'Redis',
 'SQLite',
 'AWS',
 'DigitalOcean',
 'Google Cloud Platform',
 'Heroku',
 'IBM Cloud or Watson',
 'Microsoft Azure',
 'Oracle Cloud Infrastructure',
 'ASP.NET',
 'ASP.NET Core ',
 'Angular',
 'Angular.js',
 'Django',
 'Drupal',
 'Express',
 'FastAPI',
 'Flask',
 'Gatsby',
 'Laravel',
 'React.js',
 'Ruby on Rails',
 'Spring',
 'Svelte',
 'Symfony',
 'Vue.js',
 'jQuery',
 '.NET Core / .NET 5',
 '.NET Framework',
 'Apache Spark',
 'Cordova',
 'Flutter',
 'Hadoop',
 'Keras',
 'NumPy',
 'Pandas',
 'Qt',
 'React Native',
 'TensorFlow',
 'Torch/PyTorch',
 'Ansible',
 'Chef',
 'Deno',
 'Docker',
 'Flow',
 'Git',
 'Kubernetes',
 'Pulumi',
 'Puppet',
 'Terraform',
 'Unity 3D',
 'Unreal Engine',
 'Xamarin',
 'Yarn',
 'Android Studio',
 'Atom',
 'Eclipse',
 'Emacs',
 'IPython/Jupyter',
 'IntelliJ',
 'Neovim',
 'NetBeans',
 'Notepad++',
 'PHPStorm',
 'PyCharm',
 'RStudio',
 'Rider',
 'RubyMine',
 'Sublime Text',
 'TextMate',
 'Vim',
 'Visual Studio',
 'Visual Studio Code',
 'Webstorm',
 'Xcode',
]

# Job titles
JOB_TITLES = [
    'Academic researcher',
    'Data or business analyst',
    'Data scientist or machine learning specialist',
    'Database administrator',
    'DevOps specialist',
    'Developer, QA or test',
    'Developer, back-end',
    'Developer, desktop or enterprise applications',
    'Developer, embedded applications or devices',
    'Developer, front-end',
    'Developer, full-stack',
    'Developer, game or graphics',
    'Developer, mobile',
    'Engineer, data',
    'Scientist',
    'System administrator'
]

st.set_page_config(page_title="Job Predictor", page_icon="💼", layout="wide")
st.title("💼 Job Prediction System")

# Create two columns
col1, col2 = st.columns(2)

with col1:
    st.header("Your Skills")
    selected_skills = st.multiselect(
        "Select your current skills:",
        options=sorted(ALL_SKILLS),
        help="Choose all skills you have"
    )
    
    if selected_skills:
        st.write(f"**Selected:** {', '.join(selected_skills)}")
    

with col2:
    st.header("Analysis Options")
    
    option = st.radio(
        "What would you like to do?",
        ["Predict Suitable Jobs", "Get Skill Recommendations"],
        index=0
    )
    
    if option == "Predict Suitable Jobs":
        if st.button("🔮 Predict Jobs", type="primary", use_container_width=True):
            if not selected_skills:
                st.warning("Please select your skills first!")
            else:
                with st.spinner("Predicting jobs..."):
                    try:
                        response = requests.post(
                            f"{API_URL}/predict_jobs_probs",
                            json=selected_skills
                        )
                        if response.status_code == 200:
                            predictions = response.json()
                            
                            # Filter to only show our predefined job titles
                            filtered_predictions = {}
                            for job, prob in predictions.items():
                                for job_title in JOB_TITLES:
                                    if job_title.lower() in job.lower() or job.lower() in job_title.lower():
                                        filtered_predictions[job_title] = prob
                                        break
                            
                            if not filtered_predictions:
                                filtered_predictions = predictions
                            
                            df = pd.DataFrame(filtered_predictions.items(), 
                                            columns=["Job", "Probability"])
                            df = df.sort_values("Probability", ascending=False)
                            
                            st.subheader("Top Job Predictions")
                            
                            # Display top 5
                            for _, row in df.head(5).iterrows():
                                job, prob = row["Job"], row["Probability"]
                                st.progress(prob, text=f"{job}: {prob:.1%}")
                            
                            # Show all in expander
                            with st.expander("View All Predictions"):
                                st.dataframe(df)
                            
                            # Top recommendation
                            top_job = df.iloc[0]
                            st.success(f"**Top Match:** {top_job['Job']} ({top_job['Probability']:.1%})")
                    except:
                        st.error("Could not connect to API")
    
    else:  # Skill Recommendations
        target_job = st.selectbox(
            "Select target job:",
            options=[""] + JOB_TITLES,
            format_func=lambda x: "Choose a job..." if x == "" else x
        )
        
        if st.button("🎯 Get Recommendations", type="primary", use_container_width=True):
            if not selected_skills:
                st.warning("Please select your skills first!")
            elif not target_job:
                st.warning("Please select a target job!")
            else:
                with st.spinner(f"Finding skills for {target_job}..."):
                    try:
                        payload = {
                            "available_skills": selected_skills,
                            "target_job": target_job
                        }
                        response = requests.post(
                            f"{API_URL}/recommend_new_skills",
                            json=payload
                        )
                        if response.status_code == 200:
                            recommendations = response.json()
                            df = pd.DataFrame(recommendations.items(), 
                                            columns=["Skill", "Importance"])
                            df = df.sort_values("Importance", ascending=False)
                            
                            st.subheader(f"Skills for {target_job}")
                            
                            # Separate existing vs needed skills
                            current_set = set(selected_skills)
                            
                            st.write("**Skills you need to learn:**")
                            for skill, importance in df.itertuples(index=False):
                                if skill not in current_set:
                                    st.write(f"- **{skill}** (priority: {importance:.1%})")
                            
                            st.write("**Skills you already have:**")
                            existing = [s for s in df["Skill"] if s in current_set]
                            if existing:
                                for skill in existing:
                                    st.write(f"- ✅ {skill}")
                            else:
                                st.write("None yet")
                            
                            # Progress stats
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.metric("Total Required", len(df))
                            with col_b:
                                progress = len(existing)/len(df) if len(df) > 0 else 0
                                st.metric("Progress", f"{progress:.0%}")
                    except:
                        st.error("Could not connect to API")

# Footer
st.markdown("---")
st.caption("Select your skills and choose an analysis option above")