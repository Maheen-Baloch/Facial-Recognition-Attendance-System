from pyngrok import ngrok
import uvicorn

# Paste your token here
ngrok.set_auth_token("39cnFhTRHwKYJT60UTc7q6FGFt4_7GhpmfDsyeCQ3k18qLZfp")

public_url = ngrok.connect(8000)
print("Public URL:", public_url)

uvicorn.run("app:app", host="0.0.0.0", port=8000)
