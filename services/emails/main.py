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
twitter_url = "https://x.com/chunkrai"
github_url = "https://github.com/lumina-ai-inc/chunkr"
linkedin_url = "https://www.linkedin.com/company/chunkr"
discord_url = "https://discord.gg/XzKWFByKzW"
chunkr_url = "https://chunkr.ai"

@app.post("/email/welcome")
def send_welcome_email(name: str, email: str) -> Dict:
    cal_url = "https://cal.com/mehulc/30min"

    params: resend.Emails.SendParams = {
        "from": "Chunkr <team@chunkr.ai>", 
        "to": [email],
        "reply_to": ["ishaan@chunkr.ai", "mehul@chunkr.ai"],
        "subject": "Welcome to Chunkr",
        "scheduledAt": "tomorrow at 9am",
        "html": f"""
<div style="font-family: Arial, sans-serif;">
    <p>Hello {name},</p>

    <p>Welcome to Chunkr! Thank you for signing up. We're excited to have you on board.</p>

    <p>To get you started with parsing your first document, you can use our UI at <a href="https://chunkr.ai/">https://chunkr.ai/</a>, our API via curl, or our python SDK.</p>

    <p>You can find your API key at <a href="https://chunkr.ai/dashboard">https://chunkr.ai/dashboard</a>.</p>

    <p>Here's a Python SDK example:</p>

    <pre style="background-color: #f5f5f5; padding: 15px; border-radius: 5px;">
from chunkr_ai import Chunkr

chunkr = Chunkr(api_key="your_api_key")

# Start instantly with our default configurations
task = chunkr.upload("/path/to/your/file")

# Export HTML of document
task.html(output_file="output.html")

# Export markdown of document
task.markdown(output_file="output.md")</pre>

    <p>And here's a cURL example:</p>

    <pre style="background-color: #f5f5f5; padding: 15px; border-radius: 5px;">
curl -X POST https://api.chunkr.ai/api/v1/task \\
    -H "Content-Type: multipart/form-data" \\
    -H "Authorization: YOUR_API_KEY" \\
    -F "file=@/path/to/your/file"</pre>

    <p>You can adjust the configuration options further. To learn more about the configuration options, visit our documentation at <a href="https://docs.chunkr.ai">https://docs.chunkr.ai</a></p>

    <p>We'd love to learn more about your use case! Book a call with us at {cal_url} to discuss how we can help you get the most out of Chunkr.</p>

    <p>Best,<br>Team Chunkr<br>
    <a href="{chunkr_url}">Chunkr</a> | <a href="{twitter_url}">Twitter</a> | <a href="{github_url}">GitHub</a> | <a href="{linkedin_url}">LinkedIn</a> | <a href="{discord_url}">Discord</a></p>
</div>
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
    
    email_template = ""
    if tier == "Starter":
        email_template = f"""
        <div style="font-family: Arial, sans-serif;">
            <h1>Thanks for upgrading, {name}!</h1>
            <p>Welcome to Chunkr-{tier}. You now have access to the following:</p>
            <ul>
                {"".join(f"<li>{feature}</li>" for feature in tier_info['features'])}
            </ul>
            <p>Let us know if you need any help with getting started or if you run into any issues. We're here to help!</p>
            <p>Best,<br>Team Chunkr<br>
            <a href="{chunkr_url}">Chunkr</a> | <a href="{twitter_url}">Twitter</a> | <a href="{github_url}">GitHub</a> | <a href="{linkedin_url}">LinkedIn</a> | <a href="{discord_url}">Discord</a></p>
        </div>
        """
    else:
        email_template = f"""
        <div style="font-family: Arial, sans-serif;">
            <h1>Thanks for upgrading, {name}!</h1>
            <p>Welcome to Chunkr-{tier}. You now have access to the following:</p>
            <ul>
                {"".join(f"<li>{feature}</li>" for feature in tier_info['features'])}
            </ul>
            <p>If you'd like to setup your dedicated support channels - reply to this email with a small description of how you're using chunkr.</p>
            <p>Let us know if you need any help with getting started or if you run into any issues. We're here to help!</p>
            <p>Best,<br>Team Chunkr<br>
            <a href="{chunkr_url}">Chunkr</a> | <a href="{twitter_url}">Twitter</a> | <a href="{github_url}">GitHub</a> | <a href="{linkedin_url}">LinkedIn</a> | <a href="{discord_url}">Discord</a></p>
        </div>
        """
    
    params: resend.Emails.SendParams = {
        "from": "Chunkr <team@chunkr.ai>",
        "to": [email],
        "reply_to": ["ishaan@chunkr.ai", "mehul@chunkr.ai"],
        "subject": f"Welcome to Chunkr {tier}!",
        "html": email_template
    }
    return resend.Emails.send(params)

@app.post("/email/free-pages")
def send_free_pages_email(name: str, email: str) -> Dict:
    params: resend.Emails.SendParams = {
        "from": "Chunkr <team@chunkr.ai>",
        "to": [email],
        "reply_to": ["ishaan@chunkr.ai", "mehul@chunkr.ai"],
        "subject": "Chunkr: Page limit reached",
        "html": f"""
        <div style="font-family: Arial, sans-serif;">
            <h1>Hi {name},</h1>
            <p>We hope you're enjoying Chunkr!</p>
            <p>We're reaching out to notify you that your free pages for this cycle have been used up. You can upgrade and get access to unlimited pages here by heading to the "Usage" pane in our dashboard and upgrading your plan.</p>
            <p>Let us know if you need any help with getting started or if you run into any issues. We're here to help!</p>
            <p><b>Team Chunkr</b><br>
            <a href="{chunkr_url}">Chunkr</a> | <a href="{twitter_url}">Twitter</a> | <a href="{github_url}">GitHub</a> | <a href="{linkedin_url}">LinkedIn</a> | <a href="{discord_url}">Discord</a></p>
        </div>
        """
    }
    return resend.Emails.send(params)

@app.post("/email/unpaid_invoice")
def send_unpaid_invoice_email(name: str, email: str) -> Dict:
    params: resend.Emails.SendParams = {
        "from": "Chunkr <team@chunkr.ai>",
        "to": [email],
        "reply_to": ["ishaan@chunkr.ai", "mehul@chunkr.ai"],
        "subject": "Chunkr: Action Required - Unpaid Invoice",
        "html": f"""
        <div style="font-family: Arial, sans-serif;">
            <h1>Hi {name},</h1>
            <p>We hope you're doing well!</p>
            <p>We wanted to let you know that your Chunkr usage is currently blocked due to an outstanding invoice that hasn't been paid yet.</p>
            <p>To resolve this issue and resume your service, please follow these steps:</p>
            <ol>
                <li>Go to your profile card in the Chunkr dashboard</li>
                <li>Click on "Manage Billing"</li>
                <li>View and pay any open invoices listed on the billing page</li>
            </ol>
            <p>Once your payment is processed, your service will be automatically restored.</p>
            <p>If you have any questions or need assistance with the payment process, please don't hesitate to reply to this email.</p>
            <p>Thank you for your prompt attention to this matter.</p>
            <p>Best regards,<br>Team Chunkr<br>
            <a href="{chunkr_url}">Chunkr</a> | <a href="{twitter_url}">Twitter</a> | <a href="{github_url}">GitHub</a> | <a href="{linkedin_url}">LinkedIn</a> | <a href="{discord_url}">Discord</a></p>
        </div>
        """
    }
    return resend.Emails.send(params)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
