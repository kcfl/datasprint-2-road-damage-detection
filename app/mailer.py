"""SMTP Email Dispatcher module for Civic Road Damage Reports.

Sends official grievance emails with attached annotated evidence via Gmail SMTP (TLS).
Gracefully falls back to simulation mode if credentials are not configured.
"""

from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import os
import smtplib
from typing import Any, Dict, Optional
from dotenv import load_dotenv

# Load env variables from root .env or app/.env
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_EMAIL = os.getenv("SMTP_EMAIL", "").strip()
SMTP_APP_PASSWORD = os.getenv("SMTP_APP_PASSWORD", "").strip()


def is_smtp_configured() -> bool:
    """Checks whether actual non-placeholder SMTP credentials exist."""
    if not SMTP_EMAIL or not SMTP_APP_PASSWORD:
        return False
    if "your_gmail" in SMTP_EMAIL or "your_16_char" in SMTP_APP_PASSWORD:
        return False
    return True


def send_grievance_email(
    to_email: str,
    subject: str,
    body_text: str,
    annotated_image_bytes: Optional[bytes] = None,
    cc_citizen_email: Optional[str] = None,
    citizen_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Sends official grievance email to municipal corporation with attached AI-annotated image.

    Returns dict with dispatch status, receipt id, and mode.
    """
    to_email = to_email.strip()
    if not to_email:
        return {
            "success": False,
            "simulated": False,
            "message": "Error: Recipient municipal grievance email cannot be empty.",
        }

    # If SMTP credentials are not configured, simulate delivery safely
    if not is_smtp_configured():
        return {
            "success": True,
            "simulated": True,
            "message": (
                f"Report dispatched in SIMULATION mode to '{to_email}'. "
                "To enable live Gmail delivery, add SMTP_EMAIL and SMTP_APP_PASSWORD to .env."
            ),
            "recipient": to_email,
            "cc": cc_citizen_email or "None",
            "subject": subject,
        }

    msg = MIMEMultipart("mixed")
    msg["Subject"] = subject
    msg["From"] = f"Civic Road Reporter <{SMTP_EMAIL}>"
    msg["To"] = to_email

    recipients = [to_email]
    if cc_citizen_email and "@" in cc_citizen_email:
        msg["Cc"] = cc_citizen_email
        recipients.append(cc_citizen_email)

    # Email body (plain text fallback)
    body_part = MIMEText(body_text, "plain", "utf-8")
    msg.attach(body_part)

    # Attach annotated image
    if annotated_image_bytes:
        img_part = MIMEImage(annotated_image_bytes, _subtype="jpeg")
        img_part.add_header(
            "Content-Disposition",
            "attachment",
            filename="AI_Annotated_Road_Damage_Report.jpg",
        )
        msg.attach(img_part)

    try:
        server = smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20.0)
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(SMTP_EMAIL, SMTP_APP_PASSWORD)
        server.sendmail(SMTP_EMAIL, recipients, msg.as_string())
        server.quit()

        return {
            "success": True,
            "simulated": False,
            "message": f"Official grievance report successfully emailed to {to_email} with photographic evidence.",
            "recipient": to_email,
            "cc": cc_citizen_email or "None",
            "subject": subject,
        }
    except smtplib.SMTPAuthenticationError:
        return {
            "success": False,
            "simulated": False,
            "message": (
                "SMTP Authentication failed. Ensure you are using a 16-character Google App Password "
                "(https://myaccount.google.com/apppasswords), NOT your regular Gmail password."
            ),
        }
    except Exception as e:
        return {
            "success": False,
            "simulated": False,
            "message": f"Failed to deliver email over SMTP: {str(e)}",
        }
