from dotenv import load_dotenv
import os
import cerebras
from langchain_cerebras import ChatCerebras
import streamlit as st
from langchain.prompts import ChatPromptTemplate
from serpapi import GoogleSearch


load_dotenv()
cerebras_api = st.secrets["CEREBRAS_API"]
serp_api = st.secrets["SERP_API"]
llm = ChatCerebras(model="llama3.1-8b", api_key=cerebras_api)


def get_survey_responses():
    responses = {}
    survey_questions = {
        "Rate your proficiency with Python programming (1-10)": {
            "question_type": "scale",
            "options": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "additional_text": "Please elaborate on your experience with Python programming."
        },
        "How comfortable are you with libraries like Pandas, NumPy, and Matplotlib? (1-10)": {
            "question_type": "scale",
            "options": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "additional_text": "Please describe any specific data visualization you've worked on with these libraries."
        },
        "Have you worked with machine learning algorithms such as linear regression or decision trees? (Yes/No)": {
            "question_type": "yes/no",
            "options": ["Yes", "No"],
            "additional_text": "If yes, describe your experience with training one."
        },
        "Rate your knowledge in deep learning (1-10)": {
            "question_type": "scale",
            "options": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "additional_text": "Please explain your experience with CNNs."
        },
        "Can you explain what overfitting means in machine learning?": {
            "question_type": "open_ended",
            "options": None,
            "additional_text": "Provide any examples you've encountered regarding overfitting."
        },
        "How do you stay updated with the latest trends in machine learning and AI?": {
            "question_type": "open_ended",
            "options": None,
            "additional_text": "Share any resources or platforms you use to stay updated."
        }
    }

    for question, details in survey_questions.items():
        if details["question_type"] == "scale":
            response = st.slider(question, min_value=1, max_value=10)
        elif details["question_type"] == "yes/no":
            response = st.radio(question, options=details["options"])
        elif details["question_type"] == "open_ended":
            response = st.text_area(question)
        
        additional_response = st.text_area(details["additional_text"], key=question+"_extra")
        responses[question] = {
            "answer": response,
            "additional_text": additional_response
        }
    return responses


def process_survey_responses(responses):
    formatted_responses = "\n".join([
        f"{question}:\nAnswer: {response['answer']}\nAdditional Input: {response['additional_text']}" 
        for question, response in responses.items()
    ])

    prompt_template = ChatPromptTemplate.from_messages([  
        ("system", "You are an AI assistant helping a user assess their proficiency in machine learning and Python."),
        ("user", f"The user has answered the following survey questions about their proficiency:\n{formatted_responses}\nBased on these answers, provide structured recommendations on how they can improve and get started with learning machine learning, Python, and related fields.")
    ])
    
    messages = prompt_template.format_messages()

    try:
        ai_response = llm.invoke(messages) 
        st.subheader("LLM Recommendations:")
        st.markdown(ai_response.content)
    except Exception as e:
        st.error(f"Error during LLM response generation: {e}")

    st.write("Searching for additional resources and articles on the web...")

    search_results = []
    for question, response in responses.items():
        # Constructing a query from the response
        query = f"{question}: {response['answer']} {response['additional_text']}".strip()
        if query:
            # Searching with SerpAPI
            params = {
                "q": query,
                "location": "Austin, Texas, United States",  
                "hl": "en",
                "gl": "us",
                "google_domain": "google.com",
                "api_key": serp_api
            }

            search = GoogleSearch(params)
            results = search.get_dict()
            search_results.append(results)


    for result in search_results:
        if result.get("organic_results"):
            st.write("### Top Resources from Search:")
            for item in result["organic_results"]:
                st.write(f"[{item['title']}]({item['link']})")
                st.write(f"Description: {item['snippet']}")
                st.write("\n---")

    st.write("### Recommended Courses & Practice Material:")
    st.write("[Coursera - Machine Learning by Andrew Ng](https://www.coursera.org/learn/machine-learning)")
    st.write("[Kaggle - Python Tutorial](https://www.kaggle.com/learn/python)")
    st.write("[DeepLearning.AI](https://www.deeplearning.ai/)")

    st.write("### LinkedIn Machine Learning Communities:")
    st.write("- [Machine Learning Group](https://www.linkedin.com/groups/4821567/)")
    st.write("- [Artificial Intelligence and Machine Learning](https://www.linkedin.com/groups/3132736/)")
    st.write("- [Data Science & Machine Learning Network](https://www.linkedin.com/groups/8032546/)")

    st.write("### Active Reddit Threads in Machine Learning:")
    st.write("- [r/MachineLearning](https://www.reddit.com/r/MachineLearning/)")
    st.write("- [r/LearnMachineLearning](https://www.reddit.com/r/LearnMachineLearning/)")
    st.write("- [r/Artificial](https://www.reddit.com/r/Artificial/)")
    st.write("- [r/MLQuestions](https://www.reddit.com/r/MLQuestions/)")

st.title("Machine Learning and Python Proficiency Survey")

responses = get_survey_responses()


if st.button("Submit"):
    st.write("Processing responses...")
    process_survey_responses(responses)
