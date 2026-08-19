from pathlib import Path
import hashlib, hmac, json, os, secrets

AUTH_FILE = Path(__file__).resolve().parents[1] / "admin_auth.json"


def _hash(password: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000).hex()


def setup_admin(email: str, password: str) -> None:
    email = email.strip().lower()
    if not email or "@" not in email:
        raise ValueError("Enter a valid admin email address.")
    if len(password) < 10:
        raise ValueError("Admin password must contain at least 10 characters.")
    salt = secrets.token_bytes(16)
    AUTH_FILE.write_text(json.dumps({"email": email, "salt": salt.hex(), "password_hash": _hash(password, salt)}), encoding="utf-8")
    try:
        os.chmod(AUTH_FILE, 0o600)
    except OSError:
        pass


def authenticate(email: str, password: str) -> bool:
    if not AUTH_FILE.exists():
        return False
    try:
        data = json.loads(AUTH_FILE.read_text(encoding="utf-8"))
        salt = bytes.fromhex(data["salt"])
        candidate = _hash(password, salt)
        return hmac.compare_digest(email.strip().lower(), data["email"]) and hmac.compare_digest(candidate, data["password_hash"])
    except (KeyError, ValueError, OSError):
        return False
