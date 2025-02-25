import os
import dotenv
from typing import Dict
from fastapi import FastAPI
import resend
from datetime import datetime

# Load environment variables from .env file
dotenv.load_dotenv()

# Initialize Resend API key from environment variable
resend.api_key = os.environ["RESEND_API_KEY"]
cal_url = "https://cal.com/mehulc/30min"
app = FastAPI()


@app.post("/email/welcome")
def send_welcome_email(name: str, email: str) -> Dict:
    params: resend.Emails.SendParams = {
        "from": "Chunkr <team@chunkr.ai>", 
        "to": [email],
        "subject": "Welcome to Chunkr",
        "html": f"""
            <h1>Hello {name}!</h1>
            <p>Welcome to Chunkr. Thank you for signing up. We're excited to have you on board. 
            To get you started with parsing your first document, you could use our UI at https://chunkr.ai/, or our API via curl, or our python SDK.</p>
            <p>You can find your API key at https://chunkr.ai/dashboard.</p>
            <h3>Python SDK Example:</h3>
            <pre>
from chunkr_ai import Chunkr

chunkr = Chunkr(api_key="your_api_key")

# Start instantly with our default configurations
task = chunkr.upload("/path/to/your/file")

# Export HTML of document
task.html(output_file="output.html")

# Export markdown of document
task.markdown(output_file="output.md")
            </pre>
            <h3>cURL Example:</h3>
            <pre>
curl -X POST https://api.chunkr.ai/api/v1/task \\
    -H "Content-Type: multipart/form-data" \\
    -H "Authorization: YOUR_API_KEY" \\
    -F "file=@/path/to/your/file"
            </pre>
            <p>You can adjust the configuration options further. To learn more about the configuration options, visit our documentation at docs.chunkr.ai</p>
            <p>We'd love to learn more about your use case! <a href="{cal_url}">Book a call with us</a> to discuss how we can help you get the most out of Chunkr.</p>
        """
    }
    return resend.Emails.send(params)

@app.post("/email/upgrade")
def send_upgrade_email(name: str, email: str, tier: str) -> Dict:
    tier_features = {
        "Starter": {
            "pages": "5,000",
            "overage": "$0.01",
            "features": [
                "5,000 pages included monthly",
                "$0.01 per page overage after 5,000 pages"
            ]
        },
        "Dev": {
            "pages": "25,000", 
            "overage": "$0.008",
            "features": [
                "25,000 pages included monthly",
                "$0.008 per page overage after 25,000 pages"
            ]
        },
        "Growth": {
            "pages": "100,000",
            "overage": "$0.005", 
            "features": [
                "100,000 pages included monthly",
                "$0.005 per page overage after 100,000 pages"
            ]
        }
    }

    tier_info = tier_features.get(tier, tier_features["Starter"])
    
    params: resend.Emails.SendParams = {
        "from": "Chunkr <team@chunkr.ai>",
        "to": [email],
        "subject": f"Welcome to Chunkr {tier}!",
        "html": f"""
            <h1>Thanks for upgrading, {name}!</h1>
            <p>Welcome to Chunkr-{tier}. You now have access to the following:</p>
            <ul>
                {"".join(f"<li>{feature}</li>" for feature in tier_info['features'])}
            </ul>
            <p>Let us know if you need any help getting started with your new features. We're here to help!</p>
        """
    }
    return resend.Emails.send(params)

        
        
@app.post("/email/reactivate")
def send_reactivation_email(name: str, email: str, cal_url: str) -> Dict:
    params: resend.Emails.SendParams = {
        "from": "Chunkr <team@chunkr.ai>",
        "to": [email],
        "subject": "How has your Chunkr experience been?",
        "html": f"""
            <h1>Hi {name},</h1>
            <p>We hope you are enjoying Chunkr! Thank you for being a part of our community.</p>
            <p>We'd love to understand your experience better.</p>
            <p>What aspects of Chunkr worked well for you? What could we improve to better serve your needs?</p>
            <p>Your feedback is incredibly valuable in helping us build a better product.</p>
            <p>I'd love to hear your thoughts in a quick chat:</p>
            <a href="{cal_url}">Share Your Feedback</a>
        """
    }
    return resend.Emails.send(params)

# @app.post("/email/invoice")
# def send_invoice_email(name: str, email: str, invoice_url: str) -> Dict:
#     params: resend.Emails.SendParams = {
#         "from": "Chunkr Billing <billing@resend.dev>",
#         "to": [email],
#         "subject": "Your Chunkr Invoice",
#         "html": f"""
#             <h1>Hi {name},</h1>
#             <p>Your invoice is ready. Please click below to view and pay:</p>
#             <a href="{invoice_url}">View Invoice</a>
#         """
#     }
#     return resend.Emails.send(params)

# @app.post("/email/payment-failed")
# def send_payment_failed_email(name: str, email: str, retry_url: str) -> Dict:
#     params: resend.Emails.SendParams = {
#         "from": "Chunkr Billing <billing@resend.dev>",
#         "to": [email],
#         "subject": "Payment Failed - Action Required",
#         "html": f"""
#             <h1>Hi {name},</h1>
#             <p>We were unable to process your recent payment.</p>
#             <p>Please update your payment information:</p>
#             <a href="{retry_url}">Update Payment Method</a>
#         """
#     }
#     return resend.Emails.send(params)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
