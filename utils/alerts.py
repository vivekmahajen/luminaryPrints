import smtplib
import os
from email.mime.text import MIMEText
from utils.logger import get_logger

logger = get_logger(__name__)


def send_failure_alert(subject: str, body: str):
    alert_email = os.getenv("ALERT_EMAIL", "")
    smtp_host = os.getenv("SMTP_HOST", "")
    smtp_user = os.getenv("SMTP_USER", "")
    smtp_pass = os.getenv("SMTP_PASS", "")

    if not all([alert_email, smtp_host, smtp_user, smtp_pass]):
        logger.warning("Alert email not configured — skipping email alert.")
        logger.error(f"ALERT: {subject}\n{body}")
        return

    try:
        msg = MIMEText(body)
        msg["Subject"] = f"[ArtPilot] {subject}"
        msg["From"] = smtp_user
        msg["To"] = alert_email

        with smtplib.SMTP_SSL(smtp_host, 465) as server:
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, [alert_email], msg.as_string())

        logger.info(f"Alert email sent to {alert_email}")
    except Exception as e:
        logger.error(f"Failed to send alert email: {e}")
