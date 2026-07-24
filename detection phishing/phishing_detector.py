#!/usr/bin/env python3
"""
PhishAI Sentinel
Professional AI-powered phishing detection desktop application.

Features:
- Modern PyQt6 dark cybersecurity GUI
- Real machine learning model using scikit-learn RandomForestClassifier
- In-code training dataset of safe/phishing URLs
- Saves trained model to disk and reuses it
- URL feature extraction
- Hybrid AI + heuristic risk score
- Optional VirusTotal API integration
- AI phishing probability and confidence
- Feature importance explanation
- Scan history
- Progress bar and loading spinner

Important:
This project uses a small demo dataset for educational/prototype use.
For production use, train on a large, verified, continuously updated URL dataset.
"""

import os
import sys
import json
import time
import html as html_lib
import ipaddress
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse, urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

try:
    import numpy as np
    from joblib import dump, load
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
except ImportError:
    print("Missing ML dependencies. Install with:")
    print("pip install scikit-learn joblib numpy")
    raise

try:
    from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
    from PyQt6.QtGui import QColor, QBrush
    from PyQt6.QtWidgets import (
        QApplication,
        QMainWindow,
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QGridLayout,
        QLabel,
        QLineEdit,
        QPushButton,
        QTextEdit,
        QProgressBar,
        QFrame,
        QCheckBox,
        QTableWidget,
        QTableWidgetItem,
        QHeaderView,
        QMessageBox,
        QAbstractItemView,
        QSizePolicy,
    )
except ImportError:
    print("Missing GUI dependency. Install with:")
    print("pip install PyQt6")
    raise


# ============================================================
# Application configuration
# ============================================================

APP_NAME = "PhishAI Sentinel"

APP_DIR = Path.home() / ".phishai_sentinel"
APP_DIR.mkdir(parents=True, exist_ok=True)

MODEL_VERSION = "demo-random-forest-v1"
MODEL_PATH = APP_DIR / "url_phishing_random_forest.joblib"
HISTORY_PATH = APP_DIR / "scan_history.json"

SUSPICIOUS_KEYWORDS = ("login", "verify", "secure", "update")

FEATURE_DEFINITIONS = [
    ("url_length", "URL length"),
    ("has_ip", "Uses IP address"),
    ("dot_count", "Number of dots"),
    ("keyword_count", "Suspicious keyword count"),
    ("uses_https", "Uses HTTPS"),
    ("contains_at", "Contains @ symbol"),
    ("hyphen_count", "Hyphen count"),
    ("digit_count", "Digit count"),
    ("path_depth", "Path depth"),
]

FEATURE_KEYS = [item[0] for item in FEATURE_DEFINITIONS]
FEATURE_NAMES = {key: name for key, name in FEATURE_DEFINITIONS}

VT_SUBMIT_URL = "https://www.virustotal.com/api/v3/urls"
VT_ANALYSIS_URL = "https://www.virustotal.com/api/v3/analyses/{}"

COLOR_SAFE = "#00E676"
COLOR_SUSPICIOUS = "#FFB020"
COLOR_PHISHING = "#FF4D5E"
COLOR_CYAN = "#00E5FF"
COLOR_MUTED = "#9FB7D0"


# ============================================================
# Utility functions
# ============================================================

def clamp(value, lower=0, upper=100):
    """Clamp numeric value to a range."""
    return max(lower, min(upper, int(round(value))))


def html_escape(value):
    """HTML escape helper."""
    return html_lib.escape(str(value))


def normalize_url(raw_url):
    """
    Normalize user input into a parseable URL.

    If the user enters example.com, this becomes http://example.com.
    Returns:
        normalized_url, scheme_was_missing
    """
    raw = (raw_url or "").strip()

    if not raw:
        raise ValueError("Please enter a URL.")

    if raw.startswith("//"):
        return "http:" + raw, True

    if "://" not in raw:
        return "http://" + raw, True

    return raw, False


def is_ip_hostname(hostname):
    """Return True if hostname is IPv4 or IPv6."""
    if not hostname:
        return False

    try:
        ipaddress.ip_address(hostname.strip("[]"))
        return True
    except ValueError:
        return False


def get_subdomain_counts(hostname):
    """
    Simple dependency-free subdomain counter.

    Note:
    This is not public-suffix-list aware. For production accuracy,
    consider using the tldextract package.
    """
    if not hostname or is_ip_hostname(hostname):
        return 0, 0, []

    labels = [part for part in hostname.strip(".").lower().split(".") if part]

    if len(labels) <= 2:
        return 0, 0, []

    subdomain_labels = labels[:-2]
    raw_count = len(subdomain_labels)

    risk_labels = list(subdomain_labels)

    # Treat leading www as low-risk.
    if risk_labels and risk_labels[0] == "www":
        risk_labels = risk_labels[1:]

    return raw_count, len(risk_labels), subdomain_labels


def classify_risk(score):
    """Classify final risk score into Safe, Suspicious, or Phishing."""
    score = clamp(score)

    if score < 30:
        return {
            "label": "Safe",
            "color": COLOR_SAFE,
            "background": "#0F3328",
        }

    if score < 60:
        return {
            "label": "Suspicious",
            "color": COLOR_SUSPICIOUS,
            "background": "#3A2A0A",
        }

    return {
        "label": "Phishing",
        "color": COLOR_PHISHING,
        "background": "#3A1017",
    }


def make_finding(check, status, risk, details):
    """Create a standardized heuristic finding."""
    return {
        "check": check,
        "status": status,
        "risk": int(risk),
        "details": details,
    }


def finding_color(finding):
    """Return color for a heuristic finding."""
    status = finding.get("status", "")
    risk = int(finding.get("risk", 0))

    if status == "Passed":
        return COLOR_SAFE

    if status == "Skipped":
        return COLOR_MUTED

    if status == "High Risk" or risk >= 18:
        return COLOR_PHISHING

    if status in {"Warning", "Error"} or risk > 0:
        return COLOR_SUSPICIOUS

    return COLOR_MUTED


def feature_value_to_text(key, value):
    """Pretty display for feature values."""
    if key in {"has_ip", "uses_https", "contains_at"}:
        return "Yes" if int(value) == 1 else "No"

    return str(value)


# ============================================================
# Demo ML training dataset
# ============================================================

def get_training_dataset():
    """
    Small educational training dataset.

    Label:
        0 = safe
        1 = phishing

    This intentionally uses many synthetic phishing examples instead of
    live malicious URLs.
    """

    safe_urls = [
        "https://www.google.com",
        "https://www.wikipedia.org",
        "https://github.com/login",
        "https://www.microsoft.com/security",
        "https://support.google.com/accounts",
        "https://www.apple.com/apple-id",
        "https://www.amazon.com",
        "https://www.paypal.com/signin",
        "https://www.bankofamerica.com",
        "https://www.chase.com",
        "https://accounts.google.com/signin/v2/identifier",
        "https://www.linkedin.com/login",
        "https://www.dropbox.com/login",
        "https://www.cloudflare.com",
        "https://www.mozilla.org/firefox",
        "https://docs.python.org/3/",
        "https://pypi.org/project/scikit-learn/",
        "https://www.reddit.com",
        "https://stackoverflow.com/questions",
        "https://www.netflix.com/login",
        "https://help.instagram.com",
        "https://www.office.com",
        "https://login.microsoftonline.com",
        "https://mail.google.com/mail/u/0/",
        "https://www.salesforce.com/products/",
        "https://www.adobe.com/account.html",
        "https://www.spotify.com/account/overview/",
        "https://www.ebay.com/signin",
        "https://twitter.com/i/flow/login",
        "https://www.irs.gov/refunds",
        "https://www.gov.uk/check-tax",
        "https://www.nhs.uk/conditions/",
        "https://ubuntu.com/download",
        "https://www.cisco.com/c/en/us/support/index.html",
        "https://www.verisign.com",
        "https://www.digicert.com/help/",
        "https://learn.microsoft.com/en-us/security/",
        "https://www.cloudflare.com/learning/",
        "https://example.com/contact",
        "https://www.example.org/about/team",
        "http://example.com/public-info",
        "http://neverssl.com",
        "https://www.python.org/downloads/",
        "https://developer.mozilla.org/en-US/",
    ]

    phishing_urls = [
        "http://192.168.1.55/login/verify/account",
        "http://10.0.0.5/secure-update",
        "http://172.16.12.9/account/login",
        "http://paypal.com.secure-login.verify-account.example.net",
        "http://secure-paypal-login.account-verify.example.com/session",
        "http://appleid.apple.com.verify-secure.example.net/login",
        "http://microsoft-account-update.login-security.example.org/verify",
        "http://amazon.account.verify-login.example.co/update",
        "https://login.verify.secure.account.example-phish.test",
        "http://bankofamerica.secure-update.example-login.net",
        "http://chase.verify.account-update.example-security.com",
        "http://www.paypal.com@evil-login.example.net/secure",
        "http://google.com@192.168.1.99/login",
        "http://account-update-secure-login.example.com",
        "http://verify-login-update-account.example.net",
        "http://support-login.secure-verify-account.example.org",
        "http://security-update.account-login.example.com/verify",
        "http://secure-login-example.com.verify-account.example.net",
        "http://example.com/login/secure/verify/update/account",
        "http://update-billing-secure-login.example.info",
        "http://mail.google.com.account.verify.example.net/login",
        "http://icloud.com.secure.verify-login.example.org",
        "http://facebook.com.login.verify-account.example.net",
        "http://office365.verify-update-login.example.com",
        "http://dhl.delivery-update.secure-login.example.net",
        "http://webmail.verify-login.account.example.org",
        "http://confirm-account-login-update.example.com",
        "http://secure.verify.account.login.update.example.net",
        "http://www.bank-login-secure-update.example.ru/verify",
        "http://myaccount.verify-secure-login.example.cn",
        "https://auth-secure-update-login.example-login.com/account",
        "http://www.paypaI.com/login/verify",
        "https://www.paypal.com.security-check.example.com/login",
        "http://login-account-verification.example-security.org",
        "http://secure-update-account-verification.example.co",
        "http://verify-your-account-now.example.net/login",
        "http://online-banking-secure-login.example.info/update",
        "http://free-gift-card-login-verify.example.xyz/account",
        "http://bit.ly/login-verify-account?target=paypal",
        "http://tinyurl.com/secure-account-update",
        "https://login.update.verify.secure.account.example.com",
        "https://secure.example.com@malicious.example.net/login",
        "http://verify-login.secure-update.account.example.net/session",
        "http://192.168.100.25/update/secure/login",
    ]

    dataset = [(url, 0) for url in safe_urls] + [(url, 1) for url in phishing_urls]
    return dataset


# ============================================================
# URL feature extraction
# ============================================================

def extract_url_features(raw_url):
    """
    Extract numeric ML features from a URL.

    Features:
    - URL length
    - Presence of IP address
    - Number of dots
    - Suspicious keyword count
    - HTTPS usage
    - Presence of @ symbol
    - Hyphen count
    - Digit count
    - Path depth
    """

    normalized_url, scheme_missing = normalize_url(raw_url)
    parsed = urlparse(normalized_url)

    try:
        hostname = (parsed.hostname or "").lower()
    except ValueError:
        hostname = ""

    scheme = (parsed.scheme or "").lower()
    netloc = parsed.netloc or ""
    path = parsed.path or ""
    query = parsed.query or ""

    lower_url = normalized_url.lower()

    matched_keywords = [
        keyword
        for keyword in SUSPICIOUS_KEYWORDS
        if keyword in lower_url
    ]

    has_ip = int(is_ip_hostname(hostname))
    dot_count = hostname.count(".") if hostname else 0
    keyword_count = len(matched_keywords)
    uses_https = int(scheme == "https")
    contains_at = int("@" in normalized_url)
    hyphen_count = normalized_url.count("-")
    digit_count = sum(1 for char in normalized_url if char.isdigit())
    path_depth = len([segment for segment in path.split("/") if segment])

    raw_subdomains, risk_subdomains, subdomain_labels = get_subdomain_counts(hostname)

    values = {
        "url_length": len(normalized_url),
        "has_ip": has_ip,
        "dot_count": dot_count,
        "keyword_count": keyword_count,
        "uses_https": uses_https,
        "contains_at": contains_at,
        "hyphen_count": hyphen_count,
        "digit_count": digit_count,
        "path_depth": path_depth,
    }

    vector = [float(values[key]) for key in FEATURE_KEYS]

    return {
        "normalized_url": normalized_url,
        "scheme_missing": bool(scheme_missing),
        "scheme": scheme,
        "hostname": hostname,
        "netloc": netloc,
        "path": path,
        "query": query,
        "matched_keywords": matched_keywords,
        "raw_subdomain_count": raw_subdomains,
        "risk_subdomain_count": risk_subdomains,
        "subdomain_labels": subdomain_labels,
        "values": values,
        "vector": vector,
    }


# ============================================================
# ML model class
# ============================================================

class URLAIDetector:
    """
    Handles:
    - Training the Random Forest model
    - Saving and loading the model
    - Predicting phishing probability
    - Producing feature importance explanations
    """

    def __init__(self, model_path=MODEL_PATH):
        self.model_path = Path(model_path)
        self.payload = self.load_or_train_model()
        self.model = self.payload["model"]
        self.metrics = self.payload.get("metrics", {})

    def load_or_train_model(self):
        """Load saved model if compatible; otherwise train a new one."""
        if self.model_path.exists():
            try:
                payload = load(self.model_path)

                if (
                    payload.get("version") == MODEL_VERSION
                    and payload.get("feature_keys") == FEATURE_KEYS
                    and hasattr(payload.get("model"), "predict_proba")
                ):
                    return payload

            except Exception:
                pass

        return self.train_and_save_model()

    def train_and_save_model(self):
        """Train RandomForestClassifier and save it to disk."""
        dataset = get_training_dataset()

        X = np.array(
            [extract_url_features(url)["vector"] for url, label in dataset],
            dtype=float,
        )
        y = np.array([label for url, label in dataset], dtype=int)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=42,
            stratify=y,
        )

        model = RandomForestClassifier(
            n_estimators=220,
            max_depth=7,
            random_state=42,
            class_weight="balanced",
        )

        model.fit(X_train, y_train)

        train_accuracy = float(model.score(X_train, y_train))
        test_accuracy = float(model.score(X_test, y_test))

        payload = {
            "version": MODEL_VERSION,
            "algorithm": "RandomForestClassifier",
            "feature_keys": FEATURE_KEYS,
            "feature_names": FEATURE_NAMES,
            "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": model,
            "metrics": {
                "samples": int(len(dataset)),
                "train_samples": int(len(X_train)),
                "test_samples": int(len(X_test)),
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
            },
        }

        dump(payload, self.model_path)
        return payload

    def predict_from_features(self, feature_info):
        """Predict phishing probability using already extracted features."""
        X = np.array([feature_info["vector"]], dtype=float)

        probabilities = self.model.predict_proba(X)[0]
        classes = list(self.model.classes_)

        phishing_index = classes.index(1) if 1 in classes else 1
        safe_index = classes.index(0) if 0 in classes else 0

        phishing_probability = float(probabilities[phishing_index] * 100.0)
        safe_probability = float(probabilities[safe_index] * 100.0)

        ai_prediction = "Phishing" if phishing_probability >= 50.0 else "Safe"
        confidence = phishing_probability if ai_prediction == "Phishing" else safe_probability

        feature_explanations, global_importances = self.explain_features(feature_info)

        return {
            "ai_prediction": ai_prediction,
            "phishing_probability": phishing_probability,
            "safe_probability": safe_probability,
            "confidence": float(confidence),
            "feature_explanations": feature_explanations,
            "global_importances": global_importances,
        }

    def explain_features(self, feature_info):
        """
        Explain model behavior using Random Forest feature importances
        weighted by the current URL's risk signals.

        This is an interpretable approximation, not SHAP/LIME.
        """
        importances = getattr(self.model, "feature_importances_", None)

        if importances is None or len(importances) != len(FEATURE_KEYS):
            importances = np.ones(len(FEATURE_KEYS), dtype=float) / len(FEATURE_KEYS)
        else:
            importances = np.array(importances, dtype=float)

        if importances.sum() > 0:
            importances = importances / importances.sum()

        values = feature_info["values"]

        active_explanations = []

        for key, importance in zip(FEATURE_KEYS, importances):
            value = values[key]
            signal, explanation = risk_signal_for_feature(key, value, feature_info)
            contribution = float(importance) * float(signal)

            if contribution > 0.001:
                active_explanations.append(
                    {
                        "key": key,
                        "feature": FEATURE_NAMES[key],
                        "value": value,
                        "importance": float(importance),
                        "risk_signal": float(signal),
                        "contribution": float(contribution),
                        "explanation": explanation,
                    }
                )

        active_explanations.sort(
            key=lambda item: item["contribution"],
            reverse=True,
        )

        global_importances = []

        for key, importance in zip(FEATURE_KEYS, importances):
            global_importances.append(
                {
                    "key": key,
                    "feature": FEATURE_NAMES[key],
                    "importance": float(importance),
                }
            )

        global_importances.sort(
            key=lambda item: item["importance"],
            reverse=True,
        )

        return active_explanations[:6], global_importances[:6]


def risk_signal_for_feature(key, value, feature_info):
    """
    Convert a raw feature into a risk signal between 0 and 1.

    This is used for explanation only.
    The actual ML prediction comes directly from the Random Forest model.
    """

    if key == "url_length":
        length = int(value)
        signal = min(max((length - 70) / 130, 0), 1)

        if signal > 0:
            return signal, f"URL length is {length}, which may hide misleading paths or parameters."

        return 0.0, f"URL length is {length}, which is not unusually long."

    if key == "has_ip":
        if int(value) == 1:
            return 1.0, "The hostname is a raw IP address, often used to hide identity."

        return 0.0, "The URL uses a domain name instead of a raw IP address."

    if key == "dot_count":
        dots = int(value)
        signal = min(max((dots - 1) / 5, 0), 1)

        if signal > 0:
            return signal, f"The hostname contains {dots} dots, indicating deeper subdomain nesting."

        return 0.0, f"The hostname contains {dots} dots."

    if key == "keyword_count":
        count = int(value)
        signal = min(count / 4, 1)

        if count > 0:
            keywords = ", ".join(feature_info.get("matched_keywords", []))
            return signal, f"Suspicious keyword count is {count}: {keywords}."

        return 0.0, "No configured suspicious keywords were found."

    if key == "uses_https":
        if int(value) == 0:
            return 1.0, "The URL does not use HTTPS."

        return 0.0, "The URL uses HTTPS."

    if key == "contains_at":
        if int(value) == 1:
            return 1.0, "The URL contains an @ symbol, which can disguise the true destination."

        return 0.0, "The URL does not contain an @ symbol."

    if key == "hyphen_count":
        count = int(value)
        signal = min(count / 6, 1)

        if signal > 0:
            return signal, f"The URL contains {count} hyphen characters."

        return 0.0, "The URL contains no hyphen characters."

    if key == "digit_count":
        count = int(value)
        signal = min(count / 25, 1)

        if signal > 0:
            return signal, f"The URL contains {count} digits."

        return 0.0, "The URL contains no digits."

    if key == "path_depth":
        depth = int(value)
        signal = min(depth / 6, 1)

        if signal > 0:
            return signal, f"The URL path depth is {depth}."

        return 0.0, "The URL has no deep path structure."

    return 0.0, "No explanation available."


# ============================================================
# Heuristic analysis
# ============================================================

def run_heuristic_analysis(feature_info):
    """
    Rule-based phishing analysis.

    Produces a heuristic score from 0 to 100 and detailed findings.
    """
    findings = []
    score = 0

    values = feature_info["values"]
    normalized_url = feature_info["normalized_url"]
    hostname = feature_info["hostname"]
    scheme = feature_info["scheme"]
    netloc = feature_info["netloc"]

    def add(check, status, risk, details):
        nonlocal score
        score += int(risk)
        findings.append(make_finding(check, status, risk, details))

    # URL parsing
    if not hostname:
        add(
            "URL Parsing",
            "Warning",
            12,
            "No hostname could be parsed. Malformed URLs are suspicious.",
        )
    else:
        add(
            "URL Parsing",
            "Passed",
            0,
            f"Hostname parsed successfully: {hostname}",
        )

    # URL length
    length = int(values["url_length"])

    if length <= 75:
        add(
            "URL Length Analysis",
            "Passed",
            0,
            f"URL length is {length} characters, which is within a normal range.",
        )
    elif length <= 100:
        add(
            "URL Length Analysis",
            "Warning",
            8,
            f"URL length is {length} characters. Longer URLs can obscure the destination.",
        )
    elif length <= 150:
        add(
            "URL Length Analysis",
            "Warning",
            15,
            f"URL length is {length} characters. This is unusually long.",
        )
    else:
        add(
            "URL Length Analysis",
            "High Risk",
            22,
            f"URL length is {length} characters. Extremely long URLs are common in phishing.",
        )

    # IP address
    if int(values["has_ip"]) == 1:
        add(
            "Presence of IP Address",
            "High Risk",
            25,
            f"The hostname is an IP address: {hostname}. Phishing often uses raw IP hosts.",
        )
    elif hostname:
        add(
            "Presence of IP Address",
            "Passed",
            0,
            "The URL uses a domain name instead of a raw IP address.",
        )
    else:
        add(
            "Presence of IP Address",
            "Skipped",
            0,
            "No hostname was available for IP analysis.",
        )

    # Suspicious keywords
    keyword_count = int(values["keyword_count"])
    matched_keywords = feature_info.get("matched_keywords", [])

    if keyword_count > 0:
        risk = min(25, keyword_count * 8)

        if keyword_count >= 3:
            risk = min(25, risk + 4)

        status = "High Risk" if risk >= 18 else "Warning"

        add(
            "Suspicious Keywords",
            status,
            risk,
            "Suspicious keyword(s) found: "
            + ", ".join(matched_keywords)
            + ". These are often used in credential-harvesting URLs.",
        )
    else:
        add(
            "Suspicious Keywords",
            "Passed",
            0,
            "No configured suspicious keywords were found.",
        )

    # HTTPS
    if feature_info["scheme_missing"]:
        add(
            "HTTPS vs HTTP",
            "Warning",
            10,
            "No scheme was provided. The app treated it as HTTP for analysis.",
        )
    elif scheme == "https":
        add(
            "HTTPS vs HTTP",
            "Passed",
            0,
            "The URL uses HTTPS. Note: HTTPS does not guarantee legitimacy.",
        )
    elif scheme == "http":
        add(
            "HTTPS vs HTTP",
            "Warning",
            15,
            "The URL uses HTTP instead of HTTPS.",
        )
    else:
        add(
            "HTTPS vs HTTP",
            "Warning",
            12,
            f"The URL uses an uncommon scheme: {scheme}",
        )

    # @ symbol
    if int(values["contains_at"]) == 1:
        if "@" in netloc:
            add(
                "Presence of @ Symbol",
                "High Risk",
                25,
                "The authority section contains '@'. Text before '@' can disguise the true host.",
            )
        else:
            add(
                "Presence of @ Symbol",
                "Warning",
                8,
                "The URL contains '@'. This can be benign, but is also common in deceptive URLs.",
            )
    else:
        add(
            "Presence of @ Symbol",
            "Passed",
            0,
            "No '@' symbol was found.",
        )

    # Dots and subdomains
    dot_count = int(values["dot_count"])
    raw_subdomain_count = int(feature_info["raw_subdomain_count"])
    risk_subdomain_count = int(feature_info["risk_subdomain_count"])
    subdomain_labels = feature_info.get("subdomain_labels", [])

    subdomain_text = ".".join(subdomain_labels) if subdomain_labels else "none"

    if not hostname:
        add(
            "Number of Dots/Subdomains",
            "Skipped",
            0,
            "No hostname was available for subdomain analysis.",
        )
    elif risk_subdomain_count <= 1:
        add(
            "Number of Dots/Subdomains",
            "Passed",
            0,
            f"Hostname has {dot_count} dots and {risk_subdomain_count} risk-bearing subdomains.",
        )
    elif risk_subdomain_count == 2:
        add(
            "Number of Dots/Subdomains",
            "Warning",
            8,
            f"Hostname has {dot_count} dots and {risk_subdomain_count} risk-bearing subdomains: {subdomain_text}.",
        )
    elif risk_subdomain_count <= 4:
        add(
            "Number of Dots/Subdomains",
            "Warning",
            14,
            f"Hostname has {dot_count} dots and {risk_subdomain_count} risk-bearing subdomains: {subdomain_text}.",
        )
    else:
        add(
            "Number of Dots/Subdomains",
            "High Risk",
            22,
            f"Hostname has {dot_count} dots and {risk_subdomain_count} risk-bearing subdomains: {subdomain_text}.",
        )

    return clamp(score), findings


# ============================================================
# Optional VirusTotal API integration
# ============================================================

def run_virustotal_lookup(url, api_key, progress_callback=None):
    """
    Submit a URL to VirusTotal and poll for analysis.

    Returns:
        dict with stats and risk_delta
    """

    result = {
        "enabled": True,
        "ok": False,
        "skipped": False,
        "message": "",
        "stats": {},
        "risk_delta": 0,
        "analysis_id": "",
    }

    def emit(value, message):
        if progress_callback:
            progress_callback(value, message)

    if not api_key:
        result["skipped"] = True
        result["message"] = (
            "VirusTotal lookup was enabled, but no API key was provided. "
            "Paste a key or set VIRUSTOTAL_API_KEY."
        )
        return result

    headers = {
        "x-apikey": api_key,
        "Accept": "application/json",
    }

    try:
        emit(72, "Submitting URL to VirusTotal")

        post_data = urlencode({"url": url}).encode("utf-8")

        submit_request = Request(
            VT_SUBMIT_URL,
            data=post_data,
            headers={
                **headers,
                "Content-Type": "application/x-www-form-urlencoded",
            },
            method="POST",
        )

        with urlopen(submit_request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))

        analysis_id = payload.get("data", {}).get("id", "")

        if not analysis_id:
            result["message"] = "VirusTotal did not return an analysis ID."
            return result

        result["analysis_id"] = analysis_id

        last_payload = None

        for attempt in range(1, 8):
            time.sleep(1.5)

            emit(
                min(94, 74 + attempt * 3),
                f"Waiting for VirusTotal analysis {attempt}/7",
            )

            analysis_request = Request(
                VT_ANALYSIS_URL.format(analysis_id),
                headers=headers,
                method="GET",
            )

            with urlopen(analysis_request, timeout=20) as response:
                analysis_payload = json.loads(response.read().decode("utf-8"))

            last_payload = analysis_payload

            attributes = analysis_payload.get("data", {}).get("attributes", {})
            status = attributes.get("status", "")

            if status == "completed":
                stats = attributes.get("stats", {}) or {}

                malicious = int(stats.get("malicious", 0) or 0)
                suspicious = int(stats.get("suspicious", 0) or 0)
                harmless = int(stats.get("harmless", 0) or 0)
                undetected = int(stats.get("undetected", 0) or 0)

                risk_delta = min(35, malicious * 12 + suspicious * 6)

                result.update(
                    {
                        "ok": True,
                        "stats": {
                            "malicious": malicious,
                            "suspicious": suspicious,
                            "harmless": harmless,
                            "undetected": undetected,
                        },
                        "risk_delta": risk_delta,
                    }
                )

                if malicious or suspicious:
                    result["message"] = (
                        f"VirusTotal detections: {malicious} malicious, "
                        f"{suspicious} suspicious, {harmless} harmless, "
                        f"{undetected} undetected."
                    )
                else:
                    result["message"] = (
                        f"VirusTotal reported no malicious or suspicious detections. "
                        f"Harmless: {harmless}, undetected: {undetected}."
                    )

                return result

        if last_payload:
            attributes = last_payload.get("data", {}).get("attributes", {})
            result["stats"] = attributes.get("stats", {}) or {}

        result["message"] = (
            "VirusTotal analysis did not complete within the timeout. "
            "Try again later or check API quota."
        )

        return result

    except HTTPError as error:
        if error.code in {401, 403}:
            result["message"] = "VirusTotal authentication failed. Check your API key."
        elif error.code == 429:
            result["message"] = "VirusTotal API rate limit reached."
        else:
            result["message"] = f"VirusTotal HTTP error {error.code}: {error.reason}"

        return result

    except URLError as error:
        result["message"] = f"VirusTotal network error: {error.reason}"
        return result

    except Exception as error:
        result["message"] = f"VirusTotal unexpected error: {error}"
        return result


# ============================================================
# Hybrid scanner
# ============================================================

def build_final_reason(result):
    """Create final human-readable verdict reason."""
    ai_result = result["ai_result"]
    heuristic_score = result["heuristic_score"]
    hybrid_base_score = result["hybrid_base_score"]
    vt_delta = result["virustotal"].get("risk_delta", 0)
    final_verdict = result["final_verdict"]

    top_factors = ai_result.get("feature_explanations", [])[:3]

    if top_factors:
        factor_text = ", ".join(item["feature"] for item in top_factors)
    else:
        factor_text = "no strong active ML risk signals"

    formula = (
        f"Hybrid calculation: 65% AI probability "
        f"({ai_result['phishing_probability']:.1f}) + 35% heuristic score "
        f"({heuristic_score}) = {hybrid_base_score}. "
        f"VirusTotal adjustment: +{vt_delta}. "
    )

    if final_verdict == "Safe":
        return (
            formula
            + "The final score is below the suspicious threshold. "
            + f"The model found {factor_text}."
        )

    if final_verdict == "Suspicious":
        return (
            formula
            + "The final score is in the suspicious range. "
            + f"Review carefully. Main AI factors: {factor_text}."
        )

    return (
        formula
        + "The final score is in the phishing range. "
        + f"High-risk indicators were observed. Main AI factors: {factor_text}."
    )


def run_full_scan(raw_url, detector, use_virustotal=False, vt_api_key="", progress_callback=None):
    """
    Full scanning pipeline:
    - Feature extraction
    - ML prediction
    - Heuristic analysis
    - Optional VirusTotal
    - Hybrid final score
    """

    def emit(value, message):
        if progress_callback:
            progress_callback(clamp(value), message)

    def pause(seconds=0.12):
        time.sleep(seconds)

    emit(3, "Normalizing target URL")
    pause()

    feature_info = extract_url_features(raw_url)

    emit(15, "Extracting machine learning features")
    pause()

    emit(28, "Running Random Forest AI model")
    ai_result = detector.predict_from_features(feature_info)
    pause()

    emit(44, "Running heuristic detection engine")
    heuristic_score, heuristic_findings = run_heuristic_analysis(feature_info)
    pause()

    emit(58, "Combining AI and heuristic scores")
    ml_score = float(ai_result["phishing_probability"])
    hybrid_base_score = clamp((ml_score * 0.65) + (heuristic_score * 0.35))
    pause()

    if use_virustotal:
        emit(70, "Starting VirusTotal reputation lookup")
        vt_result = run_virustotal_lookup(
            feature_info["normalized_url"],
            vt_api_key,
            progress_callback=emit,
        )
    else:
        emit(78, "VirusTotal lookup disabled")
        pause()
        vt_result = {
            "enabled": False,
            "ok": False,
            "skipped": True,
            "message": "VirusTotal lookup was disabled.",
            "stats": {},
            "risk_delta": 0,
            "analysis_id": "",
        }

    emit(95, "Calculating final verdict")
    pause()

    final_score = clamp(hybrid_base_score + int(vt_result.get("risk_delta", 0) or 0))
    classification = classify_risk(final_score)

    result = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_url": raw_url.strip(),
        "normalized_url": feature_info["normalized_url"],
        "hostname": feature_info["hostname"],
        "feature_info": feature_info,
        "ai_result": ai_result,
        "heuristic_score": heuristic_score,
        "heuristic_findings": heuristic_findings,
        "hybrid_base_score": hybrid_base_score,
        "risk_score": final_score,
        "final_verdict": classification["label"],
        "color": classification["color"],
        "background": classification["background"],
        "virustotal": vt_result,
        "model_info": {
            "algorithm": detector.payload.get("algorithm", "RandomForestClassifier"),
            "model_path": str(detector.model_path),
            "trained_at": detector.payload.get("trained_at", ""),
            "metrics": detector.metrics,
        },
    }

    result["reason"] = build_final_reason(result)

    emit(100, "Scan complete")
    return result


# ============================================================
# HTML report rendering
# ============================================================

def render_report_html(result):
    """Render full result into HTML for QTextEdit."""
    color = result.get("color", COLOR_MUTED)
    verdict = result.get("final_verdict", "Unknown")
    risk_score = int(result.get("risk_score", 0))

    ai_result = result.get("ai_result", {})
    feature_info = result.get("feature_info", {})
    model_info = result.get("model_info", {})
    vt_result = result.get("virustotal", {})

    # AI feature explanation rows
    feature_rows = ""

    for item in ai_result.get("feature_explanations", []):
        feature_rows += f"""
        <tr>
            <td><b>{html_escape(item["feature"])}</b></td>
            <td>{html_escape(feature_value_to_text(item["key"], item["value"]))}</td>
            <td>{item["importance"] * 100:.1f}%</td>
            <td>{item["contribution"] * 100:.1f}</td>
            <td class="small">{html_escape(item["explanation"])}</td>
        </tr>
        """

    if not feature_rows:
        feature_rows = """
        <tr>
            <td colspan="5" class="small">
                No strong active ML risk signals were found for this URL.
            </td>
        </tr>
        """

    # Global model importance rows
    global_rows = ""

    for item in ai_result.get("global_importances", []):
        global_rows += f"""
        <tr>
            <td>{html_escape(item["feature"])}</td>
            <td>{item["importance"] * 100:.1f}%</td>
        </tr>
        """

    # Heuristic rows
    heuristic_rows = ""

    for finding in result.get("heuristic_findings", []):
        status_color = finding_color(finding)
        risk = int(finding.get("risk", 0))
        risk_text = f"+{risk}" if risk > 0 else "0"

        heuristic_rows += f"""
        <tr>
            <td>
                <b>{html_escape(finding.get("check", ""))}</b><br>
                <span class="small">{html_escape(finding.get("details", ""))}</span>
            </td>
            <td style="color: {status_color}; font-weight: bold;">
                {html_escape(finding.get("status", ""))}
            </td>
            <td style="color: {status_color}; text-align: center; font-weight: bold;">
                {risk_text}
            </td>
        </tr>
        """

    # VirusTotal section
    if vt_result.get("enabled"):
        stats = vt_result.get("stats", {}) or {}

        if stats:
            vt_stats_html = f"""
            <ul>
                <li>Malicious: <b>{html_escape(stats.get("malicious", 0))}</b></li>
                <li>Suspicious: <b>{html_escape(stats.get("suspicious", 0))}</b></li>
                <li>Harmless: <b>{html_escape(stats.get("harmless", 0))}</b></li>
                <li>Undetected: <b>{html_escape(stats.get("undetected", 0))}</b></li>
            </ul>
            """
        else:
            vt_stats_html = ""

        vt_html = f"""
        <div class="panel">
            <h3>VirusTotal Reputation</h3>
            <p>{html_escape(vt_result.get("message", ""))}</p>
            <p><b>Risk Adjustment:</b> +{html_escape(vt_result.get("risk_delta", 0))}</p>
            {vt_stats_html}
        </div>
        """
    else:
        vt_html = """
        <div class="panel">
            <h3>VirusTotal Reputation</h3>
            <p>VirusTotal lookup was disabled for this scan.</p>
        </div>
        """

    metrics = model_info.get("metrics", {})

    return f"""
    <html>
    <head>
        <style>
            body {{
                background-color: #0B1626;
                color: #D9E6F2;
                font-family: Segoe UI, Arial, sans-serif;
                font-size: 13px;
            }}

            h2 {{
                color: {color};
                margin-bottom: 4px;
            }}

            h3 {{
                color: #E9F8FF;
                margin-bottom: 6px;
            }}

            .panel {{
                background-color: #101D2F;
                border: 1px solid #223149;
                border-radius: 10px;
                padding: 10px;
                margin-top: 10px;
            }}

            .small {{
                color: #9FB7D0;
                font-size: 12px;
            }}

            .verdict {{
                color: {color};
                font-weight: bold;
            }}

            code {{
                color: #7EE7FF;
            }}

            table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 8px;
            }}

            th {{
                background-color: #142236;
                color: #9FB7D0;
                padding: 8px;
                text-align: left;
                border-bottom: 1px solid #223149;
            }}

            td {{
                padding: 8px;
                border-bottom: 1px solid #223149;
                vertical-align: top;
            }}

            .summary td {{
                border-bottom: none;
                padding: 4px 8px 4px 0;
            }}
        </style>
    </head>

    <body>
        <h2>AI Threat Analysis Report</h2>

        <table class="summary">
            <tr>
                <td><b>Input URL:</b></td>
                <td><code>{html_escape(result.get("input_url", ""))}</code></td>
            </tr>
            <tr>
                <td><b>Normalized URL:</b></td>
                <td><code>{html_escape(result.get("normalized_url", ""))}</code></td>
            </tr>
            <tr>
                <td><b>Hostname:</b></td>
                <td>{html_escape(result.get("hostname", "") or "N/A")}</td>
            </tr>
            <tr>
                <td><b>Risk Score:</b></td>
                <td class="verdict">{risk_score}%</td>
            </tr>
            <tr>
                <td><b>Final Verdict:</b></td>
                <td class="verdict">{html_escape(verdict)}</td>
            </tr>
        </table>

        <div class="panel">
            <h3>Final Decision Reason</h3>
            <p>{html_escape(result.get("reason", ""))}</p>
            <p class="small">
                Thresholds: Safe = 0-29, Suspicious = 30-59, Phishing = 60-100.
                Hybrid score uses 65% AI model probability and 35% heuristic score,
                plus optional VirusTotal adjustment.
            </p>
        </div>

        <div class="panel">
            <h3>AI Model Result</h3>
            <table class="summary">
                <tr>
                    <td><b>AI Prediction:</b></td>
                    <td class="verdict">{html_escape(ai_result.get("ai_prediction", ""))}</td>
                </tr>
                <tr>
                    <td><b>AI Phishing Probability:</b></td>
                    <td>{ai_result.get("phishing_probability", 0):.1f}%</td>
                </tr>
                <tr>
                    <td><b>AI Safe Probability:</b></td>
                    <td>{ai_result.get("safe_probability", 0):.1f}%</td>
                </tr>
                <tr>
                    <td><b>AI Confidence:</b></td>
                    <td>{ai_result.get("confidence", 0):.1f}%</td>
                </tr>
                <tr>
                    <td><b>Model:</b></td>
                    <td>{html_escape(model_info.get("algorithm", ""))}</td>
                </tr>
                <tr>
                    <td><b>Model File:</b></td>
                    <td><code>{html_escape(model_info.get("model_path", ""))}</code></td>
                </tr>
                <tr>
                    <td><b>Trained At:</b></td>
                    <td>{html_escape(model_info.get("trained_at", ""))}</td>
                </tr>
                <tr>
                    <td><b>Dataset Size:</b></td>
                    <td>{html_escape(metrics.get("samples", 0))} demo URLs</td>
                </tr>
                <tr>
                    <td><b>Demo Test Accuracy:</b></td>
                    <td>{float(metrics.get("test_accuracy", 0)) * 100:.1f}%</td>
                </tr>
            </table>
        </div>

        <div class="panel">
            <h3>AI Feature Importance Explanation</h3>
            <p class="small">
                The table below combines Random Forest feature importance with the current URL's active risk signals.
                This explains why the model considered the URL safe or risky.
            </p>

            <table>
                <tr>
                    <th>Feature</th>
                    <th>Value</th>
                    <th>Model Importance</th>
                    <th>Risk Contribution</th>
                    <th>Explanation</th>
                </tr>
                {feature_rows}
            </table>
        </div>

        <div class="panel">
            <h3>Top Global Model Importances</h3>
            <table>
                <tr>
                    <th>Feature</th>
                    <th>Importance</th>
                </tr>
                {global_rows}
            </table>
        </div>

        <div class="panel">
            <h3>Heuristic Detection Results</h3>
            <p><b>Heuristic Score:</b> {html_escape(result.get("heuristic_score", 0))}%</p>

            <table>
                <tr>
                    <th>Detection Technique</th>
                    <th>Status</th>
                    <th>Risk</th>
                </tr>
                {heuristic_rows}
            </table>
        </div>

        {vt_html}

        <div class="panel">
            <h3>Raw Extracted Features</h3>
            <table>
                <tr>
                    <th>Feature</th>
                    <th>Value</th>
                </tr>
                {"".join(
                    f"<tr><td>{html_escape(FEATURE_NAMES[key])}</td>"
                    f"<td>{html_escape(feature_value_to_text(key, feature_info.get('values', {}).get(key, '')))}</td></tr>"
                    for key in FEATURE_KEYS
                )}
            </table>
        </div>
    </body>
    </html>
    """


def welcome_html(detector):
    """Initial information panel."""
    metrics = detector.metrics

    return f"""
    <html>
    <body style="color: #D9E6F2; font-family: Segoe UI, Arial;">
        <h2 style="color: #00E5FF;">PhishAI Sentinel Ready</h2>

        <p>
            Enter a URL and click <b>Scan URL</b>.
            The system will analyze the URL using:
        </p>

        <ul>
            <li><b>Real AI/ML:</b> scikit-learn Random Forest classifier</li>
            <li><b>Heuristics:</b> URL length, IP usage, HTTPS, @ symbol, subdomains, keywords</li>
            <li><b>Optional reputation:</b> VirusTotal API lookup</li>
        </ul>

        <div style="background-color: #101D2F; border: 1px solid #223149; border-radius: 10px; padding: 10px;">
            <h3 style="color: #E9F8FF;">AI Model Status</h3>
            <p><b>Model file:</b> <code>{html_escape(MODEL_PATH)}</code></p>
            <p><b>Training samples:</b> {html_escape(metrics.get("samples", 0))}</p>
            <p><b>Demo test accuracy:</b> {float(metrics.get("test_accuracy", 0)) * 100:.1f}%</p>
        </div>

        <p style="color: #9FB7D0;">
            Note: This is a professional prototype using a small built-in dataset.
            For production use, retrain with a much larger verified phishing/safe URL corpus.
        </p>
    </body>
    </html>
    """


# ============================================================
# Worker thread
# ============================================================

class ScanWorker(QThread):
    """Run scanning in a background thread."""

    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, url, detector, use_virustotal, vt_api_key):
        super().__init__()
        self.url = url
        self.detector = detector
        self.use_virustotal = use_virustotal
        self.vt_api_key = vt_api_key

    def run(self):
        try:
            result = run_full_scan(
                raw_url=self.url,
                detector=self.detector,
                use_virustotal=self.use_virustotal,
                vt_api_key=self.vt_api_key,
                progress_callback=self.progress.emit,
            )
            self.finished.emit(result)

        except Exception as error:
            self.failed.emit(str(error))


# ============================================================
# PyQt6 GUI
# ============================================================

class PhishAISentinelApp(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()

        self.detector = URLAIDetector()

        self.history = []
        self.worker = None

        self.loading_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.loading_index = 0
        self.current_status_message = "Idle"

        self.loading_timer = QTimer(self)
        self.loading_timer.timeout.connect(self.update_loading_animation)

        self.init_ui()
        self.load_history()

    # --------------------------------------------------------
    # UI construction
    # --------------------------------------------------------

    def init_ui(self):
        self.setWindowTitle(f"{APP_NAME} - AI URL Threat Scanner")
        self.resize(1240, 840)

        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(24, 20, 24, 20)
        main_layout.setSpacing(14)

        # Header
        header_layout = QHBoxLayout()

        title_box = QVBoxLayout()
        title_box.setSpacing(2)

        title = QLabel("PHISHAI SENTINEL")
        title.setObjectName("Title")

        subtitle = QLabel("AI-Powered URL Threat Intelligence Scanner")
        subtitle.setObjectName("Subtitle")

        title_box.addWidget(title)
        title_box.addWidget(subtitle)

        samples = self.detector.metrics.get("samples", 0)
        test_acc = float(self.detector.metrics.get("test_accuracy", 0)) * 100

        badge = QLabel(f"RANDOM FOREST AI  |  {samples} TRAINING URLS  |  TEST {test_acc:.1f}%")
        badge.setObjectName("HeaderBadge")
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)

        header_layout.addLayout(title_box, 1)
        header_layout.addWidget(badge)

        main_layout.addLayout(header_layout)

        # Input card
        input_card = self.create_card()
        input_layout = QVBoxLayout(input_card)
        input_layout.setContentsMargins(18, 16, 18, 16)
        input_layout.setSpacing(12)

        input_title = QLabel("Target URL")
        input_title.setObjectName("SectionTitle")
        input_layout.addWidget(input_title)

        url_row = QHBoxLayout()
        url_row.setSpacing(12)

        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("Enter URL, e.g. https://example.com/login")
        self.url_input.setClearButtonEnabled(True)
        self.url_input.returnPressed.connect(self.start_scan)

        self.scan_button = QPushButton("SCAN URL")
        self.scan_button.setObjectName("ScanButton")
        self.scan_button.clicked.connect(self.start_scan)

        url_row.addWidget(self.url_input, 1)
        url_row.addWidget(self.scan_button)

        input_layout.addLayout(url_row)

        options_row = QHBoxLayout()
        options_row.setSpacing(12)

        self.vt_checkbox = QCheckBox("Use VirusTotal reputation lookup")
        self.vt_checkbox.setChecked(bool(os.getenv("VIRUSTOTAL_API_KEY", "").strip()))

        self.vt_key_input = QLineEdit()
        self.vt_key_input.setPlaceholderText("VirusTotal API key, optional and not saved")
        self.vt_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.vt_key_input.setClearButtonEnabled(True)
        self.vt_key_input.setText(os.getenv("VIRUSTOTAL_API_KEY", "").strip())

        options_row.addWidget(self.vt_checkbox)
        options_row.addWidget(self.vt_key_input, 1)

        input_layout.addLayout(options_row)
        main_layout.addWidget(input_card)

        # Progress card
        progress_card = self.create_card()
        progress_layout = QHBoxLayout(progress_card)
        progress_layout.setContentsMargins(18, 12, 18, 12)
        progress_layout.setSpacing(12)

        self.status_label = QLabel("Idle")
        self.status_label.setObjectName("StatusLabel")
        self.status_label.setMinimumWidth(280)

        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("ScanProgress")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")

        progress_layout.addWidget(self.status_label)
        progress_layout.addWidget(self.progress_bar, 1)

        main_layout.addWidget(progress_card)

        # Main content
        content_layout = QHBoxLayout()
        content_layout.setSpacing(14)

        left_column = QVBoxLayout()
        left_column.setSpacing(14)

        summary_card = self.build_summary_card()
        history_card = self.build_history_card()
        details_card = self.build_details_card()

        left_column.addWidget(summary_card)
        left_column.addWidget(history_card, 1)

        content_layout.addLayout(left_column, 1)
        content_layout.addWidget(details_card, 2)

        main_layout.addLayout(content_layout, 1)

        self.apply_styles()

        self.update_risk_bar_style("#41536B")
        self.set_chip_style(self.verdict_chip, "WAITING", COLOR_MUTED, "#142236")
        self.set_chip_style(self.ai_chip, "AI IDLE", COLOR_MUTED, "#142236")
        self.details_text.setHtml(welcome_html(self.detector))

    def create_card(self):
        card = QFrame()
        card.setObjectName("Card")
        card.setFrameShape(QFrame.Shape.NoFrame)
        return card

    def create_metric_box(self, title, initial_value):
        frame = QFrame()
        frame.setObjectName("MetricBox")

        layout = QVBoxLayout(frame)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(4)

        caption = QLabel(title)
        caption.setObjectName("MetricCaption")

        value = QLabel(initial_value)
        value.setObjectName("MetricValue")

        layout.addWidget(caption)
        layout.addWidget(value)

        return frame, value

    def build_summary_card(self):
        card = self.create_card()

        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)

        title = QLabel("Risk Overview")
        title.setObjectName("SectionTitle")
        layout.addWidget(title)

        self.score_label = QLabel("—")
        self.score_label.setObjectName("ScoreLabel")
        self.score_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.score_label)

        self.risk_bar = QProgressBar()
        self.risk_bar.setObjectName("RiskBar")
        self.risk_bar.setRange(0, 100)
        self.risk_bar.setValue(0)
        self.risk_bar.setTextVisible(True)
        self.risk_bar.setFormat("0%")
        layout.addWidget(self.risk_bar)

        chip_grid = QGridLayout()
        chip_grid.setHorizontalSpacing(12)
        chip_grid.setVerticalSpacing(8)

        verdict_caption = QLabel("Final Verdict")
        verdict_caption.setObjectName("Caption")

        ai_caption = QLabel("AI Prediction")
        ai_caption.setObjectName("Caption")

        self.verdict_chip = QLabel("WAITING")
        self.verdict_chip.setMinimumHeight(42)
        self.verdict_chip.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.ai_chip = QLabel("AI IDLE")
        self.ai_chip.setMinimumHeight(42)
        self.ai_chip.setAlignment(Qt.AlignmentFlag.AlignCenter)

        chip_grid.addWidget(verdict_caption, 0, 0)
        chip_grid.addWidget(ai_caption, 0, 1)
        chip_grid.addWidget(self.verdict_chip, 1, 0)
        chip_grid.addWidget(self.ai_chip, 1, 1)

        layout.addLayout(chip_grid)

        metrics_grid = QGridLayout()
        metrics_grid.setHorizontalSpacing(10)
        metrics_grid.setVerticalSpacing(10)

        ai_prob_box, self.ai_prob_value = self.create_metric_box("AI phishing probability", "—")
        ai_conf_box, self.ai_confidence_value = self.create_metric_box("AI confidence", "—")
        heuristic_box, self.heuristic_value = self.create_metric_box("Heuristic score", "—")
        vt_box, self.vt_value = self.create_metric_box("VirusTotal adjustment", "—")

        metrics_grid.addWidget(ai_prob_box, 0, 0)
        metrics_grid.addWidget(ai_conf_box, 0, 1)
        metrics_grid.addWidget(heuristic_box, 1, 0)
        metrics_grid.addWidget(vt_box, 1, 1)

        layout.addLayout(metrics_grid)

        return card

    def build_details_card(self):
        card = self.create_card()
        card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)

        title = QLabel("Detailed Analysis Report")
        title.setObjectName("SectionTitle")
        layout.addWidget(title)

        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        self.details_text.setMinimumHeight(500)

        layout.addWidget(self.details_text, 1)

        return card

    def build_history_card(self):
        card = self.create_card()

        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(10)

        header = QHBoxLayout()

        title = QLabel("Scan History")
        title.setObjectName("SectionTitle")

        clear_button = QPushButton("Clear")
        clear_button.setObjectName("ClearButton")
        clear_button.clicked.connect(self.clear_history)

        header.addWidget(title, 1)
        header.addWidget(clear_button)

        layout.addLayout(header)

        self.history_table = QTableWidget(0, 5)
        self.history_table.setHorizontalHeaderLabels(
            ["Time", "URL", "Risk", "Verdict", "AI P(phish)"]
        )
        self.history_table.verticalHeader().setVisible(False)
        self.history_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.history_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.history_table.setMinimumHeight(230)
        self.history_table.cellClicked.connect(self.load_history_scan)

        header_view = self.history_table.horizontalHeader()
        header_view.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header_view.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header_view.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header_view.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header_view.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)

        layout.addWidget(self.history_table)

        return card

    # --------------------------------------------------------
    # Styles
    # --------------------------------------------------------

    def apply_styles(self):
        self.setStyleSheet(
            """
            QMainWindow {
                background-color: #050B14;
            }

            QWidget {
                color: #D9E6F2;
                font-family: Segoe UI, Inter, Arial, sans-serif;
                font-size: 13px;
            }

            QLabel#Title {
                color: #E9F8FF;
                font-size: 30px;
                font-weight: 900;
                letter-spacing: 1px;
            }

            QLabel#Subtitle {
                color: #00E5FF;
                font-size: 13px;
                font-weight: 700;
            }

            QLabel#HeaderBadge {
                background-color: #0B1626;
                border: 1px solid #00E5FF;
                border-radius: 14px;
                color: #00E5FF;
                padding: 10px 16px;
                font-weight: 900;
            }

            QLabel#SectionTitle {
                color: #E9F8FF;
                font-size: 16px;
                font-weight: 900;
            }

            QLabel#Caption {
                color: #9FB7D0;
                font-size: 12px;
                font-weight: 800;
            }

            QLabel#ScoreLabel {
                color: #9FB7D0;
                font-size: 56px;
                font-weight: 900;
            }

            QLabel#StatusLabel {
                color: #9FB7D0;
                font-weight: 800;
            }

            QLabel#MetricCaption {
                color: #9FB7D0;
                font-size: 11px;
                font-weight: 800;
            }

            QLabel#MetricValue {
                color: #E9F8FF;
                font-size: 16px;
                font-weight: 900;
            }

            QFrame#Card {
                background-color: #0D1624;
                border: 1px solid #223149;
                border-radius: 16px;
            }

            QFrame#MetricBox {
                background-color: #0B1626;
                border: 1px solid #223149;
                border-radius: 12px;
            }

            QLineEdit {
                background-color: #08111F;
                border: 1px solid #2D405E;
                border-radius: 11px;
                padding: 11px;
                color: #F1F7FF;
                selection-background-color: #00BCD4;
            }

            QLineEdit:focus {
                border: 1px solid #00E5FF;
            }

            QPushButton {
                background-color: #142236;
                color: #D9E6F2;
                border: 1px solid #2D405E;
                border-radius: 11px;
                padding: 10px 16px;
                font-weight: 900;
            }

            QPushButton:hover {
                background-color: #1B304D;
                border: 1px solid #00E5FF;
            }

            QPushButton:disabled {
                background-color: #1A2433;
                color: #607084;
                border: 1px solid #223149;
            }

            QPushButton#ScanButton {
                background-color: #00E5FF;
                color: #041018;
                border: none;
                border-radius: 12px;
                padding: 12px 24px;
                font-weight: 900;
            }

            QPushButton#ScanButton:hover {
                background-color: #65F3FF;
            }

            QPushButton#ClearButton {
                padding: 7px 12px;
                background-color: #1A2433;
            }

            QCheckBox {
                color: #D9E6F2;
                font-weight: 700;
            }

            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border-radius: 4px;
                border: 1px solid #2D405E;
                background-color: #08111F;
            }

            QCheckBox::indicator:checked {
                background-color: #00E5FF;
                border: 1px solid #00E5FF;
            }

            QProgressBar#ScanProgress {
                background-color: #08111F;
                border: 1px solid #223149;
                border-radius: 8px;
                text-align: center;
                color: #D9E6F2;
                height: 16px;
                font-weight: 800;
            }

            QProgressBar#ScanProgress::chunk {
                background-color: #00E5FF;
                border-radius: 8px;
            }

            QProgressBar#RiskBar {
                background-color: #08111F;
                border: 1px solid #223149;
                border-radius: 10px;
                text-align: center;
                color: #F1F7FF;
                height: 24px;
                font-weight: 900;
            }

            QTextEdit {
                background-color: #08111F;
                border: 1px solid #223149;
                border-radius: 12px;
                padding: 10px;
                color: #D9E6F2;
            }

            QTableWidget {
                background-color: #08111F;
                border: 1px solid #223149;
                border-radius: 10px;
                gridline-color: #223149;
                color: #D9E6F2;
                selection-background-color: #16395C;
                selection-color: #FFFFFF;
            }

            QHeaderView::section {
                background-color: #142236;
                color: #9FB7D0;
                padding: 7px;
                border: none;
                font-weight: 900;
            }

            QTableWidget::item {
                padding: 6px;
                border-bottom: 1px solid #162238;
            }

            QTableWidget::item:selected {
                background-color: #16395C;
                color: #FFFFFF;
            }
            """
        )

    def update_risk_bar_style(self, color):
        self.risk_bar.setStyleSheet(
            f"""
            QProgressBar#RiskBar {{
                background-color: #08111F;
                border: 1px solid #223149;
                border-radius: 10px;
                text-align: center;
                color: #F1F7FF;
                height: 24px;
                font-weight: 900;
            }}

            QProgressBar#RiskBar::chunk {{
                background-color: {color};
                border-radius: 10px;
            }}
            """
        )

    def set_chip_style(self, label, text, color, background):
        label.setText(text)
        label.setStyleSheet(
            f"""
            QLabel {{
                color: {color};
                background-color: {background};
                border: 1px solid {color};
                border-radius: 12px;
                padding: 8px;
                font-weight: 900;
            }}
            """
        )

    # --------------------------------------------------------
    # Scan flow
    # --------------------------------------------------------

    def start_scan(self):
        if self.worker and self.worker.isRunning():
            return

        url = self.url_input.text().strip()

        if not url:
            QMessageBox.warning(self, "Missing URL", "Please enter a URL to scan.")
            return

        self.scan_button.setEnabled(False)
        self.scan_button.setText("SCANNING...")
        self.progress_bar.setValue(0)

        self.current_status_message = "Starting scan"
        self.status_label.setText("Starting scan")

        self.loading_index = 0
        self.loading_timer.start(120)

        self.worker = ScanWorker(
            url=url,
            detector=self.detector,
            use_virustotal=self.vt_checkbox.isChecked(),
            vt_api_key=self.vt_key_input.text().strip(),
        )

        self.worker.progress.connect(self.on_scan_progress)
        self.worker.finished.connect(self.on_scan_finished)
        self.worker.failed.connect(self.on_scan_failed)
        self.worker.start()

    def on_scan_progress(self, value, message):
        self.progress_bar.setValue(clamp(value))
        self.current_status_message = message

    def on_scan_finished(self, result):
        self.loading_timer.stop()
        self.status_label.setText("✓ Scan complete")
        self.progress_bar.setValue(100)

        self.scan_button.setEnabled(True)
        self.scan_button.setText("SCAN URL")

        self.display_result(result)
        self.add_history_entry(result)

    def on_scan_failed(self, message):
        self.loading_timer.stop()
        self.status_label.setText("✕ Scan failed")
        self.progress_bar.setValue(0)

        self.scan_button.setEnabled(True)
        self.scan_button.setText("SCAN URL")

        QMessageBox.critical(self, "Scan Failed", message)

    def update_loading_animation(self):
        frame = self.loading_frames[self.loading_index % len(self.loading_frames)]
        self.loading_index += 1
        self.status_label.setText(f"{frame} {self.current_status_message}")

    # --------------------------------------------------------
    # Display result
    # --------------------------------------------------------

    def display_result(self, result):
        score = int(result.get("risk_score", 0))
        color = result.get("color", COLOR_MUTED)
        background = result.get("background", "#142236")
        verdict = result.get("final_verdict", "Unknown")

        ai_result = result.get("ai_result", {})
        ai_probability = float(ai_result.get("phishing_probability", 0))
        ai_confidence = float(ai_result.get("confidence", 0))
        ai_prediction = ai_result.get("ai_prediction", "Unknown")

        ai_class = classify_risk(ai_probability)

        self.score_label.setText(f"{score}%")
        self.score_label.setStyleSheet(
            f"""
            QLabel#ScoreLabel {{
                color: {color};
                font-size: 56px;
                font-weight: 900;
            }}
            """
        )

        self.risk_bar.setValue(score)
        self.risk_bar.setFormat(f"{score}%")
        self.update_risk_bar_style(color)

        self.set_chip_style(
            self.verdict_chip,
            verdict.upper(),
            color,
            background,
        )

        self.set_chip_style(
            self.ai_chip,
            f"AI {ai_prediction.upper()}",
            ai_class["color"],
            ai_class["background"],
        )

        self.ai_prob_value.setText(f"{ai_probability:.1f}%")
        self.ai_confidence_value.setText(f"{ai_confidence:.1f}%")
        self.heuristic_value.setText(f"{result.get('heuristic_score', 0)}%")

        vt = result.get("virustotal", {})

        if vt.get("enabled"):
            self.vt_value.setText(f"+{int(vt.get('risk_delta', 0))}%")
        else:
            self.vt_value.setText("Disabled")

        self.details_text.setHtml(render_report_html(result))

    # --------------------------------------------------------
    # History
    # --------------------------------------------------------

    def add_history_entry(self, result):
        self.history.insert(0, result)
        self.history = self.history[:100]
        self.save_history()
        self.refresh_history_table()

    def refresh_history_table(self):
        self.history_table.setRowCount(0)

        for result in self.history:
            row = self.history_table.rowCount()
            self.history_table.insertRow(row)

            score = int(result.get("risk_score", 0))
            classification = classify_risk(score)
            color = result.get("color", classification["color"])

            ai_result = result.get("ai_result", {})
            ai_probability = float(ai_result.get("phishing_probability", 0))

            values = [
                result.get("timestamp", ""),
                result.get("input_url") or result.get("normalized_url", ""),
                f"{score}%",
                result.get("final_verdict", ""),
                f"{ai_probability:.1f}%",
            ]

            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))

                if column == 0:
                    item.setData(Qt.ItemDataRole.UserRole, result)

                if column in {2, 3, 4}:
                    item.setForeground(QBrush(QColor(color)))
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

                if column == 1:
                    item.setToolTip(str(value))

                self.history_table.setItem(row, column, item)

    def load_history_scan(self, row, column):
        item = self.history_table.item(row, 0)

        if not item:
            return

        result = item.data(Qt.ItemDataRole.UserRole)

        if isinstance(result, dict):
            self.display_result(result)
            self.status_label.setText("Historical scan loaded")

    def clear_history(self):
        if not self.history:
            return

        reply = QMessageBox.question(
            self,
            "Clear Scan History",
            "Remove all saved scan history?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.history = []
            self.save_history()
            self.refresh_history_table()
            self.status_label.setText("Scan history cleared")

    def load_history(self):
        try:
            if HISTORY_PATH.exists():
                data = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))

                if isinstance(data, list):
                    self.history = data[:100]

        except Exception:
            self.history = []

        self.refresh_history_table()

    def save_history(self):
        try:
            HISTORY_PATH.write_text(
                json.dumps(self.history[:100], indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass


# ============================================================
# Entrypoint
# ============================================================

def main():
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setStyle("Fusion")

    window = PhishAISentinelApp()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

    