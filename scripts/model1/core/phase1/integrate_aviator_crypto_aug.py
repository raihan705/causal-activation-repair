"""
Phase 1 - Step 1.4 (revised)
integrate_aviator_crypto_aug.py

AVIATOR-style augmentation for CWE-338 and CWE-327.
Strategy:
  1. Extract method-level before/after pairs from method_change for crypto CVEs
  2. Apply string substitution to generate additional CWE-327 variants
  3. For CWE-338: use handcrafted templates (no real PRNG seeds in file_change)

Output: data/processed/crypto_augmented_pairs.csv
"""

import os
import sqlite3
import hashlib
import datetime
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DB_PATH      = os.path.join(PROJECT_ROOT, "data/raw/cvefixes/CVEfixes.db")
FILTERED_CSV = os.path.join(PROJECT_ROOT, "data/processed/cve_pairs_filtered.csv")
OUT_CSV      = os.path.join(PROJECT_ROOT, "data/processed/crypto_augmented_pairs.csv")
LOG_PATH     = os.path.join(PROJECT_ROOT, "logs/integrate_aviator_crypto_aug.log")

os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)


def log(msg):
    ts = datetime.datetime.utcnow().isoformat()
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def make_pair_id(seed, idx):
    raw = f"aug|{seed}|{idx}"
    return "aug_" + hashlib.md5(raw.encode()).hexdigest()[:12]


def make_record(pair_id, cwe_id, lang, filename, vuln, fixed, seed_cve):
    return {
        "pair_id":           pair_id,
        "cve_id":            pair_id,
        "cwe_id":            cwe_id,
        "file_change_id":    f"aug_{pair_id}",
        "hash":              f"aug_{pair_id}",
        "filename":          filename,
        "language":          lang,
        "vulnerable_code":   vuln,
        "fixed_code":        fixed,
        "num_lines_added":   "",
        "num_lines_deleted": "",
        "mono_decision":     "security_patch",
        "source":            "augmented",
        "is_augmented":      True,
        "seed_cve_id":       seed_cve
    }


# ── CWE-327 substitutions (insecure_str, secure_str) per language ─────────────
CWE327_SUBS = {
    "Python": [
        ("hashlib.md5",  "hashlib.sha256"),
        ("hashlib.sha1", "hashlib.sha256"),
        ("'MD5'",        "'SHA-256'"),
        ("'SHA1'",       "'SHA-256'"),
        ('"MD5"',        '"SHA-256"'),
        ('"SHA1"',       '"SHA-256"'),
    ],
    "Java": [
        ('"MD5"',    '"SHA-256"'),
        ('"SHA-1"',  '"SHA-256"'),
        ('"SHA1"',   '"SHA-256"'),
        ('"DES"',    '"AES/GCM/NoPadding"'),
        ("new Random()", "new SecureRandom()"),
    ],
    "JavaScript": [
        ("'md5'",    "'sha256'"),
        ("'sha1'",   "'sha256'"),
        ('"md5"',    '"sha256"'),
        ('"sha1"',   '"sha256"'),
    ],
    "C": [
        ("EVP_md5()",    "EVP_sha256()"),
        ("EVP_sha1()",   "EVP_sha256()"),
        ("MD5(",         "SHA256("),
        ("SHA1(",        "SHA256("),
    ],
    "C++": [
        ("EVP_md5()",    "EVP_sha256()"),
        ("EVP_sha1()",   "EVP_sha256()"),
        ("MD5(",         "SHA256("),
    ],
}


# ── CWE-338 handcrafted templates (vuln, fixed) per language ──────────────────
CWE338_TEMPLATES = {
    "Python": [
        (
            "import random\n\ndef generate_token(length=32):\n    chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'\n    return ''.join(random.choice(chars) for _ in range(length))\n",
            "import secrets\n\ndef generate_token(length=32):\n    chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'\n    return ''.join(secrets.choice(chars) for _ in range(length))\n"
        ),
        (
            "import random\n\ndef generate_session_id():\n    return str(random.randint(100000, 999999))\n",
            "import secrets\n\ndef generate_session_id():\n    return str(secrets.randbelow(900000) + 100000)\n"
        ),
        (
            "import random\n\ndef generate_reset_token():\n    random.seed()\n    return '%030x' % random.randrange(16**30)\n",
            "import secrets\n\ndef generate_reset_token():\n    return secrets.token_hex(15)\n"
        ),
        (
            "import random\nimport string\n\ndef generate_api_key():\n    return ''.join(random.choices(string.ascii_letters + string.digits, k=40))\n",
            "import secrets\nimport string\n\ndef generate_api_key():\n    return ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(40))\n"
        ),
        (
            "import random\n\ndef generate_otp():\n    return str(random.randint(100000, 999999))\n",
            "import secrets\n\ndef generate_otp():\n    return str(secrets.randbelow(900000) + 100000)\n"
        ),
        (
            "import random\nimport time\n\ndef generate_nonce():\n    random.seed(int(time.time()))\n    return random.getrandbits(64)\n",
            "import secrets\n\ndef generate_nonce():\n    return secrets.randbits(64)\n"
        ),
        (
            "from random import Random\n\n_rng = Random()\n\ndef get_random_bytes(n):\n    return bytes([_rng.randint(0, 255) for _ in range(n)])\n",
            "import os\n\ndef get_random_bytes(n):\n    return os.urandom(n)\n"
        ),
        (
            "import random\n\nclass TokenGenerator:\n    def generate(self, user_id):\n        random.seed(user_id)\n        return hex(random.getrandbits(128))[2:]\n",
            "import secrets\n\nclass TokenGenerator:\n    def generate(self, user_id):\n        return secrets.token_hex(16)\n"
        ),
    ],
    "Java": [
        (
            "import java.util.Random;\n\npublic class TokenUtil {\n    public static String generateToken() {\n        Random rng = new Random();\n        return Long.toHexString(rng.nextLong());\n    }\n}\n",
            "import java.security.SecureRandom;\n\npublic class TokenUtil {\n    public static String generateToken() {\n        SecureRandom rng = new SecureRandom();\n        return Long.toHexString(rng.nextLong());\n    }\n}\n"
        ),
        (
            "import java.util.Random;\n\npublic class OtpGenerator {\n    private Random random = new Random();\n    public int generateOtp() {\n        return 100000 + random.nextInt(900000);\n    }\n}\n",
            "import java.security.SecureRandom;\n\npublic class OtpGenerator {\n    private SecureRandom random = new SecureRandom();\n    public int generateOtp() {\n        return 100000 + random.nextInt(900000);\n    }\n}\n"
        ),
        (
            "import java.util.Random;\nimport java.math.BigInteger;\n\npublic class SessionIdGenerator {\n    public static String generate() {\n        return new BigInteger(130, new Random()).toString(32);\n    }\n}\n",
            "import java.security.SecureRandom;\nimport java.math.BigInteger;\n\npublic class SessionIdGenerator {\n    public static String generate() {\n        return new BigInteger(130, new SecureRandom()).toString(32);\n    }\n}\n"
        ),
        (
            "import java.util.Random;\n\npublic class CsrfTokenGenerator {\n    public static String generate() {\n        return Integer.toHexString(new Random().nextInt());\n    }\n}\n",
            "import java.security.SecureRandom;\nimport java.util.Base64;\n\npublic class CsrfTokenGenerator {\n    public static String generate() {\n        byte[] bytes = new byte[32];\n        new SecureRandom().nextBytes(bytes);\n        return Base64.getUrlEncoder().encodeToString(bytes);\n    }\n}\n"
        ),
        (
            "import java.util.Random;\n\npublic class ApiKeyGenerator {\n    public String generate(int length) {\n        StringBuilder sb = new StringBuilder();\n        Random r = new Random();\n        for (int i = 0; i < length; i++) {\n            sb.append((char)('a' + r.nextInt(26)));\n        }\n        return sb.toString();\n    }\n}\n",
            "import java.security.SecureRandom;\n\npublic class ApiKeyGenerator {\n    public String generate(int length) {\n        StringBuilder sb = new StringBuilder();\n        SecureRandom r = new SecureRandom();\n        for (int i = 0; i < length; i++) {\n            sb.append((char)('a' + r.nextInt(26)));\n        }\n        return sb.toString();\n    }\n}\n"
        ),
        (
            "import java.util.Random;\n\npublic class PasswordResetService {\n    public String createResetToken(String username) {\n        Random rand = new Random(username.hashCode());\n        return String.format(\"%016x\", rand.nextLong());\n    }\n}\n",
            "import java.security.SecureRandom;\n\npublic class PasswordResetService {\n    private SecureRandom rand = new SecureRandom();\n    public String createResetToken(String username) {\n        return String.format(\"%016x\", rand.nextLong());\n    }\n}\n"
        ),
    ],
    "JavaScript": [
        (
            "function generateToken() {\n  return Math.random().toString(36).substring(2);\n}\n",
            "const crypto = require('crypto');\nfunction generateToken() {\n  return crypto.randomBytes(16).toString('hex');\n}\n"
        ),
        (
            "function generateSessionId() {\n  return Math.floor(Math.random() * 1000000).toString();\n}\n",
            "const crypto = require('crypto');\nfunction generateSessionId() {\n  return crypto.randomInt(1000000).toString();\n}\n"
        ),
        (
            "function generateOtp() {\n  return Math.floor(100000 + Math.random() * 900000);\n}\n",
            "const crypto = require('crypto');\nfunction generateOtp() {\n  return 100000 + crypto.randomInt(900000);\n}\n"
        ),
        (
            "function generateApiKey(length) {\n  let result = '';\n  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';\n  for (let i = 0; i < length; i++) {\n    result += chars.charAt(Math.floor(Math.random() * chars.length));\n  }\n  return result;\n}\n",
            "const crypto = require('crypto');\nfunction generateApiKey(length) {\n  return crypto.randomBytes(length).toString('base64').slice(0, length);\n}\n"
        ),
        (
            "function generateNonce() {\n  return Date.now().toString() + Math.random().toString(36).slice(2);\n}\n",
            "const crypto = require('crypto');\nfunction generateNonce() {\n  return crypto.randomBytes(16).toString('hex');\n}\n"
        ),
    ],
    "C": [
        (
            "#include <stdlib.h>\n#include <time.h>\n\nvoid generate_token(char *buf, int len) {\n    srand(time(NULL));\n    for (int i = 0; i < len; i++) {\n        buf[i] = 'a' + rand() % 26;\n    }\n    buf[len] = '\\0';\n}\n",
            "#include <sys/random.h>\n\nvoid generate_token(char *buf, int len) {\n    getrandom(buf, len, 0);\n    for (int i = 0; i < len; i++) {\n        buf[i] = 'a' + (unsigned char)buf[i] % 26;\n    }\n    buf[len] = '\\0';\n}\n"
        ),
        (
            "#include <stdlib.h>\n\nint generate_session_id(void) {\n    return rand() % 1000000;\n}\n",
            "#include <stdlib.h>\n\nint generate_session_id(void) {\n    return arc4random_uniform(1000000);\n}\n"
        ),
        (
            "#include <stdlib.h>\n#include <time.h>\n\nunsigned int generate_otp(void) {\n    srand((unsigned)time(NULL));\n    return 100000 + rand() % 900000;\n}\n",
            "#include <stdlib.h>\n\nunsigned int generate_otp(void) {\n    return 100000 + arc4random_uniform(900000);\n}\n"
        ),
        (
            "#include <stdlib.h>\n\nvoid fill_random(unsigned char *buf, size_t len) {\n    for (size_t i = 0; i < len; i++) {\n        buf[i] = rand() & 0xFF;\n    }\n}\n",
            "#include <sys/random.h>\n\nvoid fill_random(unsigned char *buf, size_t len) {\n    getrandom(buf, len, 0);\n}\n"
        ),
        (
            "#include <stdlib.h>\n#include <time.h>\n\nchar *generate_csrf_token(void) {\n    static char token[33];\n    srand(time(NULL));\n    for (int i = 0; i < 32; i++) {\n        token[i] = 'a' + rand() % 26;\n    }\n    token[32] = '\\0';\n    return token;\n}\n",
            "#include <sys/random.h>\n#include <stdio.h>\n\nchar *generate_csrf_token(void) {\n    static unsigned char buf[16];\n    static char token[33];\n    getrandom(buf, 16, 0);\n    for (int i = 0; i < 16; i++) {\n        sprintf(token + i*2, \"%02x\", buf[i]);\n    }\n    return token;\n}\n"
        ),
    ],
    "C++": [
        (
            "#include <cstdlib>\n#include <ctime>\n#include <string>\n\nstd::string generateToken(int length) {\n    std::srand(std::time(nullptr));\n    std::string token;\n    const std::string chars = \"abcdefghijklmnopqrstuvwxyz0123456789\";\n    for (int i = 0; i < length; i++) {\n        token += chars[std::rand() % chars.size()];\n    }\n    return token;\n}\n",
            "#include <string>\n#include <random>\n\nstd::string generateToken(int length) {\n    std::random_device rd;\n    std::uniform_int_distribution<int> dist(0, 35);\n    const std::string chars = \"abcdefghijklmnopqrstuvwxyz0123456789\";\n    std::string token;\n    for (int i = 0; i < length; i++) {\n        token += chars[dist(rd)];\n    }\n    return token;\n}\n"
        ),
        (
            "#include <cstdlib>\n\nint generateOtp() {\n    return 100000 + std::rand() % 900000;\n}\n",
            "#include <random>\n\nint generateOtp() {\n    std::random_device rd;\n    std::uniform_int_distribution<int> dist(100000, 999999);\n    return dist(rd);\n}\n"
        ),
        (
            "#include <cstdlib>\n#include <ctime>\n\nunsigned long generateNonce() {\n    std::srand(std::time(nullptr));\n    return static_cast<unsigned long>(std::rand()) << 32 | std::rand();\n}\n",
            "#include <random>\n\nunsigned long generateNonce() {\n    std::random_device rd;\n    std::uniform_int_distribution<unsigned long> dist;\n    return dist(rd);\n}\n"
        ),
    ],
}


CWE327_TEMPLATES = {
    "Python": [
        (
            "import hashlib\n\ndef hash_password(password):\n    return hashlib.md5(password.encode()).hexdigest()\n",
            "import hashlib\n\ndef hash_password(password):\n    return hashlib.sha256(password.encode()).hexdigest()\n"
        ),
        (
            "import hashlib\n\ndef compute_checksum(data):\n    return hashlib.sha1(data).hexdigest()\n",
            "import hashlib\n\ndef compute_checksum(data):\n    return hashlib.sha256(data).hexdigest()\n"
        ),
        (
            "from Crypto.Cipher import DES\n\ndef encrypt(key, data):\n    cipher = DES.new(key, DES.MODE_ECB)\n    return cipher.encrypt(data)\n",
            "from Crypto.Cipher import AES\n\ndef encrypt(key, data):\n    cipher = AES.new(key, AES.MODE_GCM)\n    return cipher.encrypt(data)\n"
        ),
    ],
    "Java": [
        (
            "import java.security.MessageDigest;\n\npublic class HashUtil {\n    public static String hashPassword(String password) throws Exception {\n        MessageDigest md = MessageDigest.getInstance(\"MD5\");\n        byte[] hash = md.digest(password.getBytes());\n        return bytesToHex(hash);\n    }\n}\n",
            "import java.security.MessageDigest;\n\npublic class HashUtil {\n    public static String hashPassword(String password) throws Exception {\n        MessageDigest md = MessageDigest.getInstance(\"SHA-256\");\n        byte[] hash = md.digest(password.getBytes());\n        return bytesToHex(hash);\n    }\n}\n"
        ),
        (
            "import javax.crypto.Cipher;\n\npublic class CryptoUtil {\n    public byte[] encrypt(SecretKey key, byte[] data) throws Exception {\n        Cipher cipher = Cipher.getInstance(\"DES\");\n        cipher.init(Cipher.ENCRYPT_MODE, key);\n        return cipher.doFinal(data);\n    }\n}\n",
            "import javax.crypto.Cipher;\n\npublic class CryptoUtil {\n    public byte[] encrypt(SecretKey key, byte[] data) throws Exception {\n        Cipher cipher = Cipher.getInstance(\"AES/GCM/NoPadding\");\n        cipher.init(Cipher.ENCRYPT_MODE, key);\n        return cipher.doFinal(data);\n    }\n}\n"
        ),
        (
            "import java.security.MessageDigest;\n\npublic class TokenVerifier {\n    public boolean verify(String token, String stored) throws Exception {\n        MessageDigest md = MessageDigest.getInstance(\"SHA-1\");\n        String hashed = bytesToHex(md.digest(token.getBytes()));\n        return hashed.equals(stored);\n    }\n}\n",
            "import java.security.MessageDigest;\n\npublic class TokenVerifier {\n    public boolean verify(String token, String stored) throws Exception {\n        MessageDigest md = MessageDigest.getInstance(\"SHA-256\");\n        String hashed = bytesToHex(md.digest(token.getBytes()));\n        return hashed.equals(stored);\n    }\n}\n"
        ),
    ],
    "C": [
        (
            "#include <openssl/md5.h>\n\nvoid compute_hash(const unsigned char *data, size_t len, unsigned char *out) {\n    MD5(data, len, out);\n}\n",
            "#include <openssl/sha.h>\n\nvoid compute_hash(const unsigned char *data, size_t len, unsigned char *out) {\n    SHA256(data, len, out);\n}\n"
        ),
        (
            "#include <openssl/evp.h>\n\nEVP_MD_CTX *create_hash_ctx(void) {\n    EVP_MD_CTX *ctx = EVP_MD_CTX_new();\n    EVP_DigestInit_ex(ctx, EVP_sha1(), NULL);\n    return ctx;\n}\n",
            "#include <openssl/evp.h>\n\nEVP_MD_CTX *create_hash_ctx(void) {\n    EVP_MD_CTX *ctx = EVP_MD_CTX_new();\n    EVP_DigestInit_ex(ctx, EVP_sha256(), NULL);\n    return ctx;\n}\n"
        ),
    ],
    "JavaScript": [
        (
            "const crypto = require('crypto');\n\nfunction hashPassword(password) {\n  return crypto.createHash('md5').update(password).digest('hex');\n}\n",
            "const crypto = require('crypto');\n\nfunction hashPassword(password) {\n  return crypto.createHash('sha256').update(password).digest('hex');\n}\n"
        ),
        (
            "const crypto = require('crypto');\n\nfunction computeChecksum(data) {\n  return crypto.createHash('sha1').update(data).digest('hex');\n}\n",
            "const crypto = require('crypto');\n\nfunction computeChecksum(data) {\n  return crypto.createHash('sha256').update(data).digest('hex');\n}\n"
        ),
    ],
}


def apply_subs_cwe327(code, lang):
    """Generate (vuln, fixed) pairs via forward and reverse substitution."""
    subs = CWE327_SUBS.get(lang, [])
    pairs = []
    for insecure, secure in subs:
        if insecure in code:
            fixed = code.replace(insecure, secure)
            if fixed != code:
                pairs.append((code, fixed))
        if secure in code:
            vuln = code.replace(secure, insecure)
            if vuln != code:
                pairs.append((vuln, code))
    return pairs


def main():
    log("=== Step 1.4 (revised) -- AVIATOR-style Crypto Augmentation ===")

    df_filtered = pd.read_csv(FILTERED_CSV, encoding="utf-8", low_memory=False)
    cwe338_cves = df_filtered[df_filtered["cwe_id"] == "CWE-338"]["cve_id"].unique().tolist()
    cwe327_cves = df_filtered[df_filtered["cwe_id"] == "CWE-327"]["cve_id"].unique().tolist()
    all_crypto  = list(set(cwe338_cves + cwe327_cves))
    log(f"CWE-338 seed CVEs: {len(cwe338_cves)}, CWE-327 seed CVEs: {len(cwe327_cves)}")

    # ── load method-level code for crypto CVEs ────────────────────────────────
    con = sqlite3.connect(DB_PATH)
    ph  = ",".join("?" * len(all_crypto))
    q   = f"""
        SELECT mc.code, fc.programming_language, fc.filename, f.cve_id
        FROM method_change mc
        JOIN file_change fc ON mc.file_change_id = fc.file_change_id
        JOIN fixes f ON fc.hash = f.hash
        WHERE f.cve_id IN ({ph})
        AND fc.programming_language IN ('C','C++','Python','Java','JavaScript')
    """
    df_methods = pd.read_sql_query(q, con, params=all_crypto)
    con.close()
    log(f"Method-level rows: {len(df_methods)}")

    records = []
    idx = 0

    # ── CWE-327: method substitution ─────────────────────────────────────────
    for _, row in df_methods[df_methods["cve_id"].isin(cwe327_cves)].iterrows():
        code = str(row["code"]) if pd.notna(row["code"]) else ""
        lang = str(row["programming_language"]).strip()
        if not code or lang not in CWE327_SUBS:
            continue
        for vuln, fixed in apply_subs_cwe327(code, lang):
            pid = make_pair_id(str(row["cve_id"]), idx)
            records.append(make_record(pid, "CWE-327", lang,
                str(row["filename"]), vuln, fixed, str(row["cve_id"])))
            idx += 1
    log(f"CWE-327 pairs from method substitution: {idx}")

    # ── CWE-327: template-based (supplement) ─────────────────────────────────
    n_327_tpl = idx
    seed_327 = cwe327_cves[0] if cwe327_cves else "template_CWE327"
    for lang, templates in CWE327_TEMPLATES.items():
        for i, (vuln, fixed) in enumerate(templates):
            pid = make_pair_id(f"tpl_327_{lang}", i)
            records.append(make_record(pid, "CWE-327", lang,
                f"template_{lang.lower()}_weakalgo_{i}.code", vuln, fixed, seed_327))
            idx += 1
    log(f"CWE-327 template pairs added: {idx - n_327_tpl}")

    # ── CWE-338: template-based ───────────────────────────────────────────────
    n_before = idx
    seed_cve = cwe338_cves[0] if cwe338_cves else "template_CWE338"
    for lang, templates in CWE338_TEMPLATES.items():
        for i, (vuln, fixed) in enumerate(templates):
            pid = make_pair_id(f"tpl_338_{lang}", i)
            records.append(make_record(pid, "CWE-338", lang,
                f"template_{lang.lower()}_prng_{i}.code", vuln, fixed, seed_cve))
            idx += 1
    log(f"CWE-338 template pairs added: {idx - n_before}")

    # ── dedup and save ────────────────────────────────────────────────────────
    df_aug = pd.DataFrame(records).drop_duplicates(subset=["pair_id"])
    df_aug.to_csv(OUT_CSV, index=False, encoding="utf-8")
    log(f"Saved {len(df_aug)} augmented pairs to {OUT_CSV}")
    log(f"CWE counts: {df_aug['cwe_id'].value_counts().to_dict()}")
    log(f"Language counts: {df_aug['language'].value_counts().to_dict()}")

    # ── checkpoint ────────────────────────────────────────────────────────────
    total_338 = len(cwe338_cves) + (df_aug["cwe_id"] == "CWE-338").sum()
    total_327 = len(cwe327_cves) + (df_aug["cwe_id"] == "CWE-327").sum()
    log(f"Total CWE-338 (seed+aug): {total_338}")
    log(f"Total CWE-327 (seed+aug): {total_327}")

    if total_338 < 20 or total_327 < 20:
        log("CHECKPOINT FAIL: Insufficient augmented samples for crypto CWEs.")
        return False

    log("CHECKPOINT PASS: Crypto augmentation complete.")
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)