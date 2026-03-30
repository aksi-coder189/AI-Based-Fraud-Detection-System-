"""
alerts.py
---------
Alert generation, logging, and retrieval logic.
Alerts are stored in a local JSON file (no DB required for demo).
Swap log_alert() with a MySQL INSERT for production.
"""

import os
import json
import uuid
from datetime import datetime
from typing import Optional

ALERT_LOG_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'alert_log.json')


def _load_alerts() -> list:
    """Load existing alerts from the JSON log file."""
    if not os.path.exists(ALERT_LOG_PATH):
        return []
    with open(ALERT_LOG_PATH, 'r') as f:
        return json.load(f)


def _save_alerts(alerts: list):
    """Persist alert list to JSON file."""
    os.makedirs(os.path.dirname(ALERT_LOG_PATH), exist_ok=True)
    with open(ALERT_LOG_PATH, 'w') as f:
        json.dump(alerts, f, indent=2, default=str)


def log_alert(transaction_id: str, prediction: dict, transaction: dict) -> Optional[dict]:
    """
    Create and persist a fraud alert if the transaction is flagged as fraud.

    Parameters
    ----------
    transaction_id : Unique ID for the transaction (str or int)
    prediction     : Output dict from predict.predict_single()
    transaction    : Original transaction feature dict

    Returns
    -------
    alert dict if fraud detected, else None
    """
    if not prediction.get('is_fraud', False):
        return None

    alert = {
        'alert_id':          str(uuid.uuid4())[:8].upper(),
        'transaction_id':    str(transaction_id),
        'fraud_probability': prediction['fraud_probability'],
        'risk_level':        prediction['risk_level'],
        'model_used':        prediction['model_used'],
        'amount':            transaction.get('Amount', 'N/A'),
        'status':            'OPEN',
        'created_at':        datetime.now().isoformat(),
        'resolved_at':       None,
        'resolved_by':       None,
    }

    alerts = _load_alerts()
    alerts.append(alert)
    _save_alerts(alerts)

    print(f"[ALERT] 🚨 Fraud detected! Alert ID: {alert['alert_id']} | "
          f"Amount: {alert['amount']} | Risk: {alert['risk_level']} | "
          f"P(fraud)={alert['fraud_probability']:.2%}")

    return alert


def get_all_alerts() -> list:
    """Return all logged alerts sorted by newest first."""
    alerts = _load_alerts()
    return sorted(alerts, key=lambda a: a['created_at'], reverse=True)


def get_open_alerts() -> list:
    """Return only unresolved (OPEN) alerts."""
    return [a for a in get_all_alerts() if a['status'] == 'OPEN']


def resolve_alert(alert_id: str, resolved_by: str = 'Admin') -> bool:
    """
    Mark an alert as RESOLVED.

    Returns True if found and updated, False otherwise.
    """
    alerts = _load_alerts()
    for alert in alerts:
        if alert['alert_id'] == alert_id:
            alert['status']      = 'RESOLVED'
            alert['resolved_at'] = datetime.now().isoformat()
            alert['resolved_by'] = resolved_by
            _save_alerts(alerts)
            print(f"[INFO] Alert {alert_id} resolved by {resolved_by}.")
            return True
    print(f"[WARN] Alert {alert_id} not found.")
    return False


def get_summary_stats() -> dict:
    """Return summary statistics for the dashboard."""
    alerts = _load_alerts()
    total   = len(alerts)
    open_   = sum(1 for a in alerts if a['status'] == 'OPEN')
    resolved= sum(1 for a in alerts if a['status'] == 'RESOLVED')
    high    = sum(1 for a in alerts if a['risk_level'] == 'HIGH')
    medium  = sum(1 for a in alerts if a['risk_level'] == 'MEDIUM')

    return {
        'total_alerts':    total,
        'open_alerts':     open_,
        'resolved_alerts': resolved,
        'high_risk':       high,
        'medium_risk':     medium,
    }


if __name__ == '__main__':
    # Demo
    fake_prediction = {
        'fraud_probability': 0.87,
        'is_fraud': True,
        'risk_level': 'HIGH',
        'model_used': 'random_forest',
    }
    fake_transaction = {'Amount': 1250.00}
    alert = log_alert('TXN-001', fake_prediction, fake_transaction)
    print(json.dumps(alert, indent=2))
    print("\nSummary:", get_summary_stats())
