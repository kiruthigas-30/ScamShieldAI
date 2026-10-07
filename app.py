from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
)

from flask_sqlalchemy import SQLAlchemy

from flask_login import (
    LoginManager,
    UserMixin,
    login_user,
    logout_user,
    login_required,
    current_user,
)

from werkzeug.security import (
    generate_password_hash,
    check_password_hash,
)

from werkzeug.utils import secure_filename

import joblib
from feature_extractor import get_features

import numpy as np
import os
import re
import time

import pytesseract
import cv2

from PIL import Image


# =========================================================
# FLASK APP
# =========================================================

app = Flask(__name__)


# =========================================================
# CONFIGURATION
# =========================================================

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///database.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SECRET_KEY"] = "scamshield-secret-key"


# =========================================================
# UPLOAD CONFIGURATION
# =========================================================

UPLOAD_FOLDER = "uploads"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg"
}


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# =========================================================
# TESSERACT OCR
# =========================================================

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


# =========================================================
# DATABASE
# =========================================================

db = SQLAlchemy(app)


# =========================================================
# LOAD URL ML MODEL
# =========================================================

phishing_model = joblib.load(
    "models/phishing_detection_model.pkl"
)

scaler = joblib.load(
    "models/scaler.pkl"
)


# =========================================================
# LOAD SMS SPAM NLP MODEL
# =========================================================

spam_model = joblib.load(
    "models/spam_model.pkl"
)

tfidf_vectorizer = joblib.load(
    "models/tfidf_vectorizer.pkl"
)


print("Phishing ML model loaded successfully!")
print("Scaler loaded successfully!")
print("Spam NLP model loaded successfully!")
print("TF-IDF vectorizer loaded successfully!")


# =========================================================
# USER MODEL
# =========================================================

class User(UserMixin, db.Model):

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    username = db.Column(
        db.String(100),
        unique=True,
        nullable=False
    )

    password = db.Column(
        db.String(200),
        nullable=False
    )


# =========================================================
# COMMUNITY REPORT MODEL
# =========================================================

class ScamReport(db.Model):

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    user_id = db.Column(
        db.Integer,
        db.ForeignKey("user.id"),
        nullable=False
    )

    report_type = db.Column(
        db.String(50),
        nullable=False
    )

    title = db.Column(
        db.String(200),
        nullable=False
    )

    description = db.Column(
        db.Text,
        nullable=False
    )

    related_url = db.Column(
        db.String(500),
        nullable=True
    )

    severity = db.Column(
        db.String(30),
        nullable=False
    )

    created_at = db.Column(
        db.DateTime,
        server_default=db.func.now(),
        nullable=False
    )

    user = db.relationship(
        "User",
        backref=db.backref(
            "scam_reports",
            lazy=True
        )
    )


# =========================================================
# LOGIN MANAGER
# =========================================================

login_manager = LoginManager()

login_manager.init_app(app)

login_manager.login_view = "login"


@login_manager.user_loader
def load_user(user_id):

    return db.session.get(
        User,
        int(user_id)
    )


# =========================================================
# COMMUNITY REPORTS
# =========================================================

def get_recent_reports():

    return (
        ScamReport.query
        .order_by(
            ScamReport.created_at.desc()
        )
        .limit(10)
        .all()
    )


# =========================================================
# SESSION SCAN HELPERS
# =========================================================

def get_scan_results():

    data = session.get(
        "scan_results",
        {}
    )

    if isinstance(data, dict):
        return data

    return {}


def save_scan_result(
    scanner_name,
    data
):

    scan_results = get_scan_results()

    scan_results[scanner_name] = data

    session["scan_results"] = scan_results

    session.modified = True


def clear_scan_results():

    session.pop(
        "scan_results",
        None
    )

    session.modified = True


# =========================================================
# OVERALL RISK
# =========================================================

def calculate_overall_risk():

    scan_results = get_scan_results()

    scores = []

    for data in scan_results.values():

        if not isinstance(data, dict):
            continue

        score = data.get("risk_score")

        if score is None:
            continue

        try:

            scores.append(
                int(score)
            )

        except (
            TypeError,
            ValueError
        ):

            continue

    if not scores:
        return 0

    return max(scores)


# =========================================================
# DASHBOARD CONTEXT
# =========================================================

def dashboard_context():

    scan_results = get_scan_results()

    return {

        "url_scan":
            scan_results.get("url"),

        "screenshot_scan":
            scan_results.get("screenshot"),

        "seller_scan":
            scan_results.get("seller"),

        "overall_risk_score":
            calculate_overall_risk(),

        "reports":
            get_recent_reports(),

    }


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =========================================================
# REGISTER
# =========================================================

@app.route(
    "/register",
    methods=[
        "GET",
        "POST"
    ]
)
def register():

    if request.method == "POST":

        username = request.form.get(
            "username",
            ""
        ).strip()

        password = request.form.get(
            "password",
            ""
        )

        if not username or not password:

            return (
                "Username and password are required."
            )

        existing_user = (
            User.query
            .filter_by(
                username=username
            )
            .first()
        )

        if existing_user:

            return (
                "Username already exists!"
            )

        user = User(

            username=username,

            password=generate_password_hash(
                password
            )

        )

        db.session.add(
            user
        )

        db.session.commit()

        return redirect(
            url_for("login")
        )

    return render_template(
        "register.html"
    )


# =========================================================
# LOGIN
# =========================================================

@app.route(
    "/login",
    methods=[
        "GET",
        "POST"
    ]
)
def login():

    if request.method == "POST":

        username = request.form.get(
            "username",
            ""
        ).strip()

        password = request.form.get(
            "password",
            ""
        )

        user = (
            User.query
            .filter_by(
                username=username
            )
            .first()
        )

        if (
            user
            and
            check_password_hash(
                user.password,
                password
            )
        ):

            login_user(user)

            return redirect(
                url_for("dashboard")
            )

        return (
            "Invalid username or password!"
        )

    return render_template(
        "login.html"
    )


# =========================================================
# DASHBOARD
# =========================================================

@app.route("/dashboard")
@login_required
def dashboard():

    return render_template(
        "dashboard.html",
        **dashboard_context()
    )


# =========================================================
# URL ANALYSIS
# =========================================================

def analyze_url(url):

    # -----------------------------------------------------
    # FEATURE EXTRACTION
    # -----------------------------------------------------

    features = get_features(url)

    print(
        "FEATURE COUNT:",
        len(features)
    )

    if len(features) != 50:

        raise ValueError(
            "Feature extraction error. "
            "Expected 50 features."
        )

    feature_array = np.array(
        features,
        dtype=float
    ).reshape(
        1,
        -1
    )

    # -----------------------------------------------------
    # SCALE FEATURES
    # -----------------------------------------------------

    scaled_features = scaler.transform(
        feature_array
    )

    # -----------------------------------------------------
    # ML PREDICTION
    # -----------------------------------------------------

    prediction = phishing_model.predict(
        scaled_features
    )[0]

    probabilities = phishing_model.predict_proba(
        scaled_features
    )[0]

    # -----------------------------------------------------
    # PHISHING PROBABILITY
    # -----------------------------------------------------

    phishing_index = list(
        phishing_model.classes_
    ).index(1)

    phishing_probability = float(
        probabilities[
            phishing_index
        ]
    )

    # -----------------------------------------------------
    # URL INDICATORS
    # -----------------------------------------------------

    suspicious_words = [

        "verify",
        "verification",
        "account-verify",
        "secure-login",
        "security",
        "update-account",
        "confirm-account",
        "login-verification",
        "password",
        "bank-login",
        "wallet",
        "payment",

    ]

    url_lower = url.lower()

    suspicious_keyword = any(
        word in url_lower
        for word in suspicious_words
    )

    suspicious_domain = (

        ".example" in url_lower
        or
        ".xyz" in url_lower
        or
        ".top" in url_lower
        or
        ".click" in url_lower

    )

    uses_http = url_lower.startswith(
        "http://"
    )

    has_at_symbol = (
        "@" in url_lower
    )

    many_hyphens = (
        url_lower.count("-") >= 3
    )

    ip_pattern = (
        r"^(https?://)?"
        r"\d{1,3}"
        r"(\.\d{1,3}){3}"
    )

    uses_ip = bool(
        re.search(
            ip_pattern,
            url_lower
        )
    )

    # -----------------------------------------------------
    # RISK SCORE
    # -----------------------------------------------------

    risk_score = int(
        round(
            phishing_probability * 100
        )
    )

    if prediction == 1:
        risk_score += 20

    if suspicious_keyword:
        risk_score += 15

    if suspicious_domain:
        risk_score += 25

    risk_score = min(
        risk_score,
        100
    )

    # -----------------------------------------------------
    # DETAILS
    # -----------------------------------------------------

    details = []

    if uses_http:

        details.append(
            "The website uses an insecure HTTP connection."
        )

    if suspicious_keyword:

        details.append(
            "The website address contains wording commonly "
            "seen in login, verification, payment, or phishing links."
        )

    if suspicious_domain:

        details.append(
            "The domain uses an unusual extension that can "
            "also appear in deceptive or suspicious links."
        )

    if uses_ip:

        details.append(
            "The website uses an IP address instead of "
            "a normal domain name."
        )

    if has_at_symbol:

        details.append(
            "The URL contains an unusual @ symbol that "
            "can sometimes disguise the actual destination."
        )

    if many_hyphens:

        details.append(
            "The domain has an unusually complex structure "
            "with several hyphens."
        )

    # -----------------------------------------------------
    # RESULT
    # -----------------------------------------------------

    if risk_score >= 70:

        result = (
            "🚨 Phishing URL Detected!"
        )

        reason = (
            f'The website "{url}" appears highly suspicious '
            "because its URL shows characteristics commonly "
            "associated with phishing websites."
        )

        risk_explanation = (
            "The detected signals suggest that this website "
            "may be attempting to mislead users or obtain "
            "sensitive information."
        )

        safety_advice = (
            "Do not enter passwords, OTPs, banking details, "
            "card information, or payment information on this website."
        )

    elif risk_score >= 40:

        result = (
            "⚠️ Suspicious URL"
        )

        reason = (
            f'The website "{url}" appears suspicious because '
            "its address contains characteristics that can be "
            "associated with phishing or deceptive websites."
        )

        risk_explanation = (
            "The available signals suggest that this website "
            "should be treated with caution."
        )

        safety_advice = (
            "Verify the website address carefully before "
            "entering personal or financial information."
        )

    else:

        result = (
            "✅ Legitimate URL"
        )

        reason = (
            f'The website "{url}" appears low-risk because '
            "its URL does not show strong phishing indicators "
            "and the machine-learning model estimates a low phishing risk."
        )

        risk_explanation = (
            "Based on the available URL signals, no strong "
            "evidence of phishing was detected."
        )

        safety_advice = (
            "Even when a website appears legitimate, verify "
            "the address before entering sensitive information."
        )

    # -----------------------------------------------------
    # DEFAULT DETAIL
    # -----------------------------------------------------

    if not details:

        if prediction == 1:

            details.append(
                "The machine-learning model detected "
                "characteristics associated with phishing."
            )

        else:

            details.append(
                "No major suspicious URL characteristics were detected."
            )

    return {

        "type":
            "URL",

        "result":
            result,

        "probability":
            phishing_probability,

        "risk_score":
            risk_score,

        "url":
            url,

        "reason":
            reason,

        "risk_explanation":
            risk_explanation,

        "safety_advice":
            safety_advice,

        "details":
            details,

    }


# =========================================================
# URL SCANNER
# =========================================================

@app.route(
    "/check-link",
    methods=["POST"]
)
@login_required
def check_link():

    url = request.form.get(
        "url",
        ""
    ).strip()

    if not url:

        return redirect(
            url_for("dashboard")
        )

    try:

        analysis = analyze_url(
            url
        )

        save_scan_result(
            "url",
            analysis
        )

        print(
            "\n========== URL SCAN =========="
        )

        print(
            "URL:",
            url
        )

        print(
            "Probability:",
            analysis["probability"]
        )

        print(
            "Risk Score:",
            analysis["risk_score"]
        )

        print(
            "Result:",
            analysis["result"]
        )

        print(
            "==============================\n"
        )

        return render_template(
            "dashboard.html",
            **dashboard_context()
        )

    except Exception as e:

        print(
            "URL Prediction Error:",
            e
        )

        return render_template(
            "dashboard.html",
            **dashboard_context()
        )


# =========================================================
# NLP SCAM ANALYSIS
# =========================================================

def analyze_text_with_nlp(text):

    vector = (
        tfidf_vectorizer
        .transform(
            [text]
        )
    )

    prediction = (
        spam_model
        .predict(
            vector
        )[0]
    )

    probabilities = (
        spam_model
        .predict_proba(
            vector
        )[0]
    )

    spam_index = (
        list(
            spam_model.classes_
        )
        .index(1)
    )

    spam_probability = float(
        probabilities[
            spam_index
        ]
    )

    risk_score = int(
        round(
            spam_probability * 100
        )
    )

    text_lower = text.lower()

    scam_signals = []

    # -----------------------------------------------------
    # SCAM KEYWORDS
    # -----------------------------------------------------

    scam_keywords = {

        "otp":
            "The message asks for an OTP or one-time password.",

        "one time password":
            "The message asks for an OTP or one-time password.",

        "password":
            "The message contains a request involving a password.",

        "pin":
            "The message contains a request involving a PIN.",

        "cvv":
            "The message refers to sensitive card information such as CVV.",

        "bank account":
            "The message refers to a bank account or banking information.",

        "account blocked":
            "The message creates urgency by claiming that an account is blocked.",

        "account suspended":
            "The message creates urgency by claiming that an account is suspended.",

        "verify your account":
            "The message asks the user to verify an account.",

        "click here":
            "The message asks the user to click a link.",

        "claim now":
            "The message asks the user to claim something immediately.",

        "urgent":
            "The message uses urgent language to pressure the user.",

        "immediately":
            "The message pressures the user to take immediate action.",

        "limited time":
            "The message creates pressure using a limited-time offer.",

        "winner":
            "The message contains a prize or winner claim.",

        "you won":
            "The message claims that the user has won something.",

        "congratulations":
            "The message contains a reward or prize claim.",

        "prize":
            "The message mentions a prize or reward.",

        "reward":
            "The message mentions a reward.",

        "refund":
            "The message contains a refund-related claim.",

        "cashback":
            "The message contains a cashback-related claim.",

        "lottery":
            "The message contains a lottery-related claim.",

        "gift voucher":
            "The message offers a gift voucher.",

        "free gift":
            "The message offers a free gift.",

        "investment":
            "The message contains an investment-related offer.",

        "double your money":
            "The message promises unusually high financial returns.",

        "guaranteed profit":
            "The message promises guaranteed profit.",

        "send money":
            "The message asks the user to send money.",

        "pay now":
            "The message asks the user to make an immediate payment.",

        "payment required":
            "The message claims that payment is required.",

        "processing fee":
            "The message requests a processing fee.",

        "registration fee":
            "The message requests a registration fee.",

        "security deposit":
            "The message requests a security deposit.",

        "remote access":
            "The message asks for remote access to a device.",

        "screen sharing":
            "The message asks the user to share their screen.",

    }

    for keyword, explanation in scam_keywords.items():

        if keyword in text_lower:

            scam_signals.append(
                explanation
            )

    # -----------------------------------------------------
    # URL IN MESSAGE
    # -----------------------------------------------------

    url_pattern = (
        r"(https?://\S+|www\.\S+|"
        r"\b[a-zA-Z0-9-]+\."
        r"(com|net|org|xyz|top|click)\b)"
    )

    if re.search(
        url_pattern,
        text_lower
    ):

        scam_signals.append(
            "The message contains a website link that "
            "should be verified before opening."
        )

    # -----------------------------------------------------
    # MONEY
    # -----------------------------------------------------

    money_pattern = (
        r"(₹\s?\d[\d,]*|"
        r"\$\s?\d[\d,]*|"
        r"\brs\.?\s?\d[\d,]*)"
    )

    if re.search(
        money_pattern,
        text_lower
    ):

        scam_signals.append(
            "The message contains a money amount or financial claim."
        )

    # -----------------------------------------------------
    # URGENCY
    # -----------------------------------------------------

    urgency_patterns = [

        "act now",
        "hurry",
        "today only",
        "within 24 hours",
        "expires today",
        "last chance",
        "do not delay",

    ]

    for pattern in urgency_patterns:

        if pattern in text_lower:

            scam_signals.append(
                "The message uses urgency to pressure "
                "the user into acting quickly."
            )

    scam_signals = list(
        dict.fromkeys(
            scam_signals
        )
    )

    # -----------------------------------------------------
    # RESULT
    # -----------------------------------------------------

    if prediction == 1:

        if risk_score >= 70:

            result = (
                "🚨 Potential Scam Detected"
            )

            reason = (
                "This screenshot appears suspicious because "
                "the message contains patterns commonly found "
                "in scam or phishing messages."
            )

            risk_explanation = (
                "The message may be attempting to create urgency, "
                "promote an unexpected reward, request sensitive "
                "information, or direct the user toward a risky action."
            )

            safety_advice = (
                "Do not click suspicious links or share OTPs, "
                "passwords, PINs, banking details, or payment information."
            )

        else:

            result = (
                "⚠️ Suspicious Message"
            )

            reason = (
                "This screenshot contains some characteristics "
                "that can be associated with scam messages."
            )

            risk_explanation = (
                "The message should be treated carefully because "
                "its content may be attempting to influence the "
                "user to take an unexpected action."
            )

            safety_advice = (
                "Verify the sender and message before clicking "
                "links or sharing personal information."
            )

    else:

        if risk_score >= 40:

            result = (
                "⚠️ Suspicious Message"
            )

            reason = (
                "This screenshot contains some characteristics "
                "that may be associated with suspicious content."
            )

            risk_explanation = (
                "Although the model did not strongly classify "
                "the message as spam, some risk signals were detected."
            )

            safety_advice = (
                "Verify the sender before responding or sharing "
                "any sensitive information."
            )

        else:

            result = (
                "✅ No Obvious Scam Detected"
            )

            reason = (
                "The text extracted from this screenshot does "
                "not show strong characteristics commonly associated "
                "with scam messages."
            )

            risk_explanation = (
                "Based on the available text and model prediction, "
                "no strong evidence of a scam was detected."
            )

            safety_advice = (
                "Continue to verify unexpected messages before "
                "clicking links or sharing sensitive information."
            )

    if not scam_signals:

        if prediction == 1:

            scam_signals.append(
                "The machine-learning model detected "
                "patterns associated with scam messages."
            )

        else:

            scam_signals.append(
                "No major scam-related text indicators were detected."
            )

    return {

        "type":
            "Message",

        "result":
            result,

        "probability":
            spam_probability,

        "risk_score":
            risk_score,

        "reason":
            reason,

        "risk_explanation":
            risk_explanation,

        "safety_advice":
            safety_advice,

        "details":
            scam_signals,

    }


# =========================================================
# SCREENSHOT / QR SCANNER
# =========================================================

@app.route(
    "/scan-screenshot",
    methods=["POST"]
)
@login_required
def scan_screenshot():

    if "screenshot" not in request.files:

        return redirect(
            url_for("dashboard")
        )

    file = request.files[
        "screenshot"
    ]

    if file.filename == "":

        return redirect(
            url_for("dashboard")
        )

    if not allowed_file(
        file.filename
    ):

        return redirect(
            url_for("dashboard")
        )

    try:

        # -------------------------------------------------
        # SAVE IMAGE
        # -------------------------------------------------

        filename = secure_filename(
            file.filename
        )

        filename = (
            str(
                int(
                    time.time() * 1000
                )
            )
            + "_"
            + filename
        )

        filepath = os.path.join(
            app.config["UPLOAD_FOLDER"],
            filename
        )

        file.save(
            filepath
        )

        print(
            "Screenshot saved:",
            filepath
        )

        # =================================================
        # QR DETECTION
        # =================================================

        cv_image = cv2.imread(
            filepath
        )

        qr_detector = cv2.QRCodeDetector()

        qr_text = ""

        if cv_image is not None:

            decoded_text, points, _ = (
                qr_detector.detectAndDecode(
                    cv_image
                )
            )

            if decoded_text:

                qr_text = (
                    decoded_text.strip()
                )

        # =================================================
        # QR FOUND
        # =================================================

        if qr_text:

            print(
                "QR CODE DETECTED:",
                qr_text
            )

            # -------------------------------------------------
            # QR URL
            # -------------------------------------------------

            url_match = re.search(
                r"(https?://\S+|www\.\S+)",
                qr_text,
                re.IGNORECASE
            )

            if url_match:

                decoded_url = (
                    url_match
                    .group(0)
                    .rstrip(
                        ".,);]"
                    )
                )

                if decoded_url.lower().startswith(
                    "www."
                ):

                    decoded_url = (
                        "https://"
                        + decoded_url
                    )

                analysis = analyze_url(
                    decoded_url
                )

                analysis["type"] = "QR"

                analysis["qr_text"] = qr_text

                save_scan_result(
                    "screenshot",
                    analysis
                )

                return render_template(
                    "dashboard.html",
                    **dashboard_context()
                )

            # -------------------------------------------------
            # QR TEXT
            # -------------------------------------------------

            analysis = (
                analyze_text_with_nlp(
                    qr_text
                )
            )

            data = {

                "type":
                    "QR",

                "result":
                    analysis["result"],

                "risk_score":
                    analysis["risk_score"],

                "probability":
                    analysis["probability"],

                "qr_text":
                    qr_text,

                "reason":
                    "The QR code contains text that was analyzed "
                    "for scam-related characteristics.",

                "risk_explanation":
                    analysis["risk_explanation"],

                "safety_advice":
                    analysis["safety_advice"],

                "details":
                    analysis["details"],

            }

            save_scan_result(
                "screenshot",
                data
            )

            return render_template(
                "dashboard.html",
                **dashboard_context()
            )

        # =================================================
        # OCR
        # =================================================

        image = Image.open(
            filepath
        )

        extracted_text = (
            pytesseract
            .image_to_string(
                image
            )
            .strip()
        )

        print(
            "\n========== OCR TEXT =========="
        )

        print(
            extracted_text
        )

        print(
            "==============================\n"
        )

        if not extracted_text:

            data = {

                "type":
                    "Screenshot",

                "result":
                    (
                        "No QR code or text "
                        "could be detected."
                    ),

                "risk_score":
                    0,

                "probability":
                    0,

                "ocr_text":
                    "",

            }

            save_scan_result(
                "screenshot",
                data
            )

            return render_template(
                "dashboard.html",
                **dashboard_context()
            )

        # -------------------------------------------------
        # NLP ANALYSIS
        # -------------------------------------------------

        analysis = (
            analyze_text_with_nlp(
                extracted_text
            )
        )

        stored_text = (
            extracted_text[:3000]
        )

        data = {

            "type":
                "Screenshot",

            "result":
                analysis["result"],

            "risk_score":
                analysis["risk_score"],

            "probability":
                analysis["probability"],

            "ocr_text":
                stored_text,

            "reason":
                analysis["reason"],

            "risk_explanation":
                analysis["risk_explanation"],

            "safety_advice":
                analysis["safety_advice"],

            "details":
                analysis["details"],

        }

        save_scan_result(
            "screenshot",
            data
        )

        print(
            "Spam Probability:",
            analysis["probability"]
        )

        print(
            "Risk Score:",
            analysis["risk_score"]
        )

        print(
            "Result:",
            analysis["result"]
        )

        return render_template(
            "dashboard.html",
            **dashboard_context()
        )

    except Exception as e:

        print(
            "Screenshot / QR Error:",
            e
        )

        data = {

            "type":
                "Screenshot",

            "result":
                (
                    "Screenshot analysis error: "
                    + str(e)
                ),

            "risk_score":
                0,

            "probability":
                None,

        }

        save_scan_result(
            "screenshot",
            data
        )

        return render_template(
            "dashboard.html",
            **dashboard_context()
        )


# =========================================================
# SELLER ANALYSIS
# =========================================================

def analyze_seller(
    seller_name,
    seller_url,
    phone
):

    score = 0

    signals = []

    name_lower = (
        seller_name
        .lower()
        .strip()
    )

    url_lower = (
        seller_url
        .lower()
        .strip()
    )

    phone_clean = (
        phone
        .strip()
    )

    suspicious_name_keywords = [

        "deal",
        "discount",
        "cheap",
        "winner",
        "prize",
        "offer",
        "loan",
        "investment",
        "crypto",
        "free",

    ]

    # -----------------------------------------------------
    # SELLER NAME
    # -----------------------------------------------------

    for keyword in suspicious_name_keywords:

        if keyword in name_lower:

            score += 5

            signals.append(
                "The seller name contains wording commonly "
                "used in promotional or suspicious offers."
            )

    # -----------------------------------------------------
    # DOMAIN
    # -----------------------------------------------------

    suspicious_domains = [

        ".xyz",
        ".top",
        ".click",
        ".shop",
        ".info",
        ".buzz",

    ]

    for domain in suspicious_domains:

        if domain in url_lower:

            score += 15

            signals.append(
                "The seller website uses an unusual domain "
                "extension that can also appear in deceptive websites."
            )

    # -----------------------------------------------------
    # HTTP
    # -----------------------------------------------------

    if url_lower.startswith(
        "http://"
    ):

        score += 15

        signals.append(
            "The seller website does not use a secure HTTPS connection."
        )

    # -----------------------------------------------------
    # @ SYMBOL
    # -----------------------------------------------------

    if "@" in url_lower:

        score += 15

        signals.append(
            "The seller website address contains an unusual "
            "@ symbol that can sometimes disguise a destination."
        )

    # -----------------------------------------------------
    # HYPHENS
    # -----------------------------------------------------

    if url_lower.count("-") >= 3:

        score += 10

        signals.append(
            "The seller website has an unusually complex domain structure."
        )

    # -----------------------------------------------------
    # IP ADDRESS
    # -----------------------------------------------------

    ip_pattern = (
        r"^(https?://)?"
        r"\d{1,3}"
        r"(\.\d{1,3}){3}"
    )

    if re.search(
        ip_pattern,
        url_lower
    ):

        score += 25

        signals.append(
            "The seller website uses an IP address instead "
            "of a normal domain name."
        )

    # -----------------------------------------------------
    # MISSING WEBSITE
    # -----------------------------------------------------

    if not url_lower:

        score += 10

        signals.append(
            "The seller did not provide a website, making "
            "the seller harder to verify."
        )

    # -----------------------------------------------------
    # PHONE
    # -----------------------------------------------------

    if phone_clean:

        phone_digits = re.sub(
            r"\D",
            "",
            phone_clean
        )

        if len(phone_digits) < 10:

            score += 15

            signals.append(
                "The seller phone number appears unusually "
                "short or may be invalid."
            )

    else:

        score += 5

        signals.append(
            "No phone number was provided for seller verification."
        )

    score = min(
        score,
        100
    )

    # =====================================================
    # RESULT
    # =====================================================

    if score >= 70:

        result = (
            "🚨 High Risk Seller"
        )

        reason = (
            f'The seller "{seller_name}" appears highly '
            "suspicious because several seller or website "
            "characteristics suggest that the seller may not "
            "be trustworthy."
        )

        risk_explanation = (
            "The available information indicates that extra "
            "caution is needed before purchasing from this "
            "seller or sharing personal or payment information."
        )

        safety_advice = (
            "Do not make a payment until the seller, website, "
            "contact details, and business information have "
            "been independently verified."
        )

    elif score >= 40:

        result = (
            "⚠️ Suspicious Seller"
        )

        reason = (
            f'The seller "{seller_name}" appears suspicious '
            "because some of the provided seller or website "
            "details show characteristics commonly associated "
            "with risky sellers."
        )

        risk_explanation = (
            "The available information is not sufficient to "
            "establish that the seller is trustworthy, so the "
            "seller should be treated with caution."
        )

        safety_advice = (
            "Verify the seller and website independently "
            "before making a payment or sharing personal information."
        )

    else:

        result = (
            "✅ Low Risk Seller"
        )

        reason = (
            f'The seller "{seller_name}" appears low-risk '
            "based on the seller information provided and "
            "the absence of strong suspicious indicators."
        )

        risk_explanation = (
            "No major warning signs were detected from the "
            "provided seller details, but this does not guarantee "
            "that the seller is completely trustworthy."
        )

        safety_advice = (
            "Continue to verify the seller, website, and payment "
            "details before making a purchase."
        )

    signals = list(
        dict.fromkeys(
            signals
        )
    )

    return {

        "type":
            "Seller",

        "result":
            result,

        "score":
            score,

        "risk_score":
            score,

        "signals":
            signals,

        "reason":
            reason,

        "risk_explanation":
            risk_explanation,

        "safety_advice":
            safety_advice,

    }


# =========================================================
# SELLER CHECKER
# =========================================================

@app.route(
    "/check-seller",
    methods=["POST"]
)
@login_required
def check_seller():

    seller_name = request.form.get(
        "seller_name",
        ""
    ).strip()

    seller_url = request.form.get(
        "seller_url",
        ""
    ).strip()

    phone = request.form.get(
        "phone",
        ""
    ).strip()

    if not seller_name:

        return redirect(
            url_for("dashboard")
        )

    try:

        analysis = analyze_seller(
            seller_name,
            seller_url,
            phone
        )

        data = {

            "type":
                "Seller",

            "result":
                analysis["result"],

            "risk_score":
                analysis["risk_score"],

            "seller_name":
                seller_name,

            "seller_url":
                seller_url,

            "seller_phone":
                phone,

            "seller_signals":
                analysis["signals"],

            "reason":
                analysis["reason"],

            "risk_explanation":
                analysis["risk_explanation"],

            "safety_advice":
                analysis["safety_advice"],

        }

        save_scan_result(
            "seller",
            data
        )

        return render_template(
            "dashboard.html",
            **dashboard_context()
        )

    except Exception as e:

        print(
            "Seller Checker Error:",
            e
        )

        return render_template(
            "dashboard.html",
            **dashboard_context()
        )


# =========================================================
# COMMUNITY REPORT
# =========================================================

@app.route(
    "/report-scam",
    methods=["POST"]
)
@login_required
def report_scam():

    report_type = request.form.get(
        "report_type",
        ""
    ).strip()

    title = request.form.get(
        "title",
        ""
    ).strip()

    description = request.form.get(
        "description",
        ""
    ).strip()

    related_url = request.form.get(
        "related_url",
        ""
    ).strip()

    severity = request.form.get(
        "severity",
        ""
    ).strip()

    if (
        not report_type
        or
        not title
        or
        not description
    ):

        return redirect(
            url_for("dashboard")
        )

    if severity not in [
        "Low",
        "Medium",
        "High"
    ]:

        severity = "Medium"

    report = ScamReport(

        user_id=current_user.id,

        report_type=report_type,

        title=title,

        description=description,

        related_url=(
            related_url
            if related_url
            else None
        ),

        severity=severity,

    )

    db.session.add(
        report
    )

    db.session.commit()

    print(
        "Community report saved successfully!"
    )

    return redirect(
        url_for("dashboard")
    )


# =========================================================
# DELETE COMMUNITY REPORT
# =========================================================
# =========================================================
# DELETE COMMUNITY REPORT
# =========================================================

@app.route(
    "/delete-report/<int:report_id>",
    methods=["POST"]
)
@login_required
def delete_report(report_id):

    report = ScamReport.query.get_or_404(
        report_id
    )

    db.session.delete(
        report
    )

    db.session.commit()

    print(
        "Community report deleted successfully!"
    )

    return redirect(
        url_for("dashboard")
    )

# =========================================================
# RESET ALL SCAN RESULTS
# =========================================================

@app.route(
    "/reset-risk",
    methods=["POST"]
)
@login_required
def reset_risk():

    clear_scan_results()

    return redirect(
        url_for("dashboard")
    )


# =========================================================
# LOGOUT
# =========================================================

@app.route(
    "/logout"
)
@login_required
def logout():

    logout_user()

    session.pop(
        "scan_results",
        None
    )

    session.modified = True

    return redirect(
        url_for("login")
    )


# =========================================================
# START APPLICATION
# =========================================================

with app.app_context():
    db.create_all()


if __name__ == "__main__":
    app.run(
        debug=True
    )