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

app = FastAPI()


@app.post("/email/welcome")
def send_welcome_email(name: str, email: str) -> Dict:
    params: resend.Emails.SendParams = {
        "from": "Chunkr <team@chunkr.ai>", 
        "to": [email],
        "subject": "Welcome to Chunkr",
        "html": f"""
            <h1>Hello {name}!</h1>
            <p>Welcome to Chunkr. Here's everything you need to know to get started...</p>
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
                "$0.01 per page overage"
            ]
        },
        "Dev": {
            "pages": "25,000", 
            "overage": "$0.008",
            "features": [
                "25,000 pages included monthly",
                "$0.008 per page overage"
            ]
        },
        "Growth": {
            "pages": "100,000",
            "overage": "$0.005", 
            "features": [
                "100,000 pages included monthly",
                "$0.005 per page overage"
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
            <p>Welcome to Chunkr-{tier}. You now have access to these great features:</p>
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
