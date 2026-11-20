import os
import time

import requests

from . import config as C

_last_sent = {}


def send_telegram(status, detail):
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return False, "TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set"
    now = time.time()
    if now - _last_sent.get(status, 0) < C.ALERT_COOLDOWN_S:
        return False, "cooldown"
    try:
        r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          data={"chat_id": chat_id, "text": f"Machine health alert: {status}\n{detail}"}, timeout=5)
    except requests.RequestException as e:
        return False, str(e)
    if r.status_code == 200:
        _last_sent[status] = now
        return True, "sent"
    return False, f"HTTP {r.status_code}"
