# import os
# from openai import OpenAI
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# # Retrieve the API key
# api_key = os.getenv("OPENAI_API_KEY")

# if not api_key:
#     print("❌ API Key Not Found! Check your .env file.")
# else:
#     print(f"✅ API Key Loaded: {api_key[:5]}...{api_key[-5:]}")

#     # Initialize OpenAI client
#     client = OpenAI(api_key=api_key)

#     try:
#         # Make a request to the OpenAI API using gpt-3.5-turbo
#         response = client.chat.completions.create(
#             model="gpt-3.5-turbo",  # Specify the model
#             messages=[              # Provide the messages as input
#                 {"role": "system", "content": "You are a helpful assistant."},
#                 {"role": "user", "content": "Say hello!"}
#             ]
#         )

#         # Correct way to extract the response content
#         print("✅ API is working! Response:", response.choices[0].message.content)

#     except Exception as e:
#         print(f"❌ API Request Failed! {e}")

